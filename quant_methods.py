"""可选择的 INT4 量化方法：RTN / GPTQ / AWQ / AWQ+GPTQ。

三种方法共用同一套"fp -> INT4"语义（`quant.py` 的 `Quantizer`，
per-channel 或 per-group 的 affine 量化），差别只在量化前的权重预处理：

    rtn        : 无预处理，直接分组 RTN（round-to-nearest），不需要标定数据
    gptq       : 无预处理，Hessian 逆的逐列误差补偿（`gptq.py`，本仓库原有实现）
    awq        : AWQ 等价缩放 + 裁剪搜索（`awq.py`）后再做分组 RTN
    awq_gptq   : AWQ 等价变换 + GPTQ 列补偿（两者互补）

对外接口：
    add_method_args(parser)      给三个入口脚本注册统一的命令行参数
    resolve_method(args)         --nearest / --quant_method 归一化成方法名
    quantize_block(...)          逐层统一驱动（标定 -> 变换 -> 量化 -> 重算输出）
"""

import functools
import gc

import torch

from awq import AWQQuantState, apply_awq, quantize_weight
from gptq import GPTQ
from quant import Quantizer
from modelutils import find_layers
__all__ = [
    'METHODS', 'add_method_args', 'resolve_method', 'quantize_block',
    'check_save_supported', 'collect_input_features', 'pack_model',
]

METHODS = ('rtn', 'gptq', 'awq', 'awq_gptq')


# --------------------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------------------
def add_method_args(parser):
    """给入口脚本（opt.py / llama.py / bloom.py）注册量化方法相关参数。"""
    group = parser.add_argument_group('INT4 quantization method')
    group.add_argument(
        '--quant_method', type=str, default=None, choices=list(METHODS),
        help='INT4 quantization method: rtn (no calibration) / gptq (default) / '
             'awq (AWQ scaling + clipping, then RTN) / awq_gptq (AWQ + GPTQ).'
    )
    group.add_argument(
        '--awq-calib-samples', type=int, default=16,
        help='Number of calibration samples used by the AWQ scale/clip search.'
    )
    group.add_argument(
        '--awq-n-grid', type=int, default=20,
        help='Grid size for the AWQ scale search (alpha in [0, 1)).'
    )
    group.add_argument(
        '--awq-no-clip', action='store_true',
        help='Skip the AWQ weight-clipping search (scale search only).'
    )
    return parser


def resolve_method(args):
    """把 `--nearest` / `--quant_method` 归一化成方法名；wbits>=16 返回 None。"""
    if args.wbits >= 16:
        return None
    method = getattr(args, 'quant_method', None)
    nearest = getattr(args, 'nearest', False)
    if method is None:
        method = 'rtn' if nearest else 'gptq'
    elif nearest and method != 'rtn':
        print(f'[quant] `--nearest` is ignored because --quant_method={method}')
    if getattr(args, 'trits', False) and method != 'gptq':
        raise NotImplementedError('`--trits` is only supported with --quant_method gptq')
    print(f'[quant] INT4 quantization method: {method}')
    return method


def check_save_supported(args, method):
    """Validate the checkpoint format selected by --save."""
    if not getattr(args, 'save', ''):
        return
    if method == 'awq' and args.wbits == 4:
        if args.groupsize in (None, -1, 0):
            args.groupsize = -1  # resolved later by Quant4Linear
        return
    if args.wbits != 3 or args.groupsize not in (-1, 0):
        raise NotImplementedError(
            '--save supports AWQ 4-bit grouped checkpoints, or the legacy '
            '3-bit per-row format (--wbits 3 --groupsize -1).'
        )


# --------------------------------------------------------------------------------------
# 标定：收集每个 Linear 的输入激活（AWQ 用）
# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect_input_features(linears, inps, fwd, args):
    """跑少量标定样本，收集 `linears` 每个层的输入（保存在 CPU 上）。

    与 GPTQ 收集 Hessian 的那一趟相比，AWQ 只需要前 ~16 个样本就足够搜索了，
    这样显存/内存开销可控（每层 `n_samples * seqlen * hidden` fp16）。
    """
    n = min(inps.shape[0], max(1, getattr(args, 'awq_calib_samples', 16)))
    feats = {name: [] for name in linears}

    def hook(m, inp, out, name):
        feats[name].append(inp[0].detach().to('cpu'))

    handles = [
        m.register_forward_hook(functools.partial(hook, name=name))
        for name, m in linears.items()
    ]
    try:
        for j in range(n):
            fwd(inps[j].unsqueeze(0))
    finally:
        for h in handles:
            h.remove()

    return {name: torch.cat(v, dim=0) for name, v in feats.items() if v}


# --------------------------------------------------------------------------------------
# GPTQ：Hessian 收集
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _build_gptq(linears, names, inps, fwd, args):
    gptq = {}
    for name in names:
        g = GPTQ(linears[name])
        g.quantizer = Quantizer()
        g.quantizer.configure(
            args.wbits, perchannel=True, sym=args.sym, mse=False,
            trits=getattr(args, 'trits', False),
        )
        gptq[name] = g

    def hook(m, inp, out, name):
        gptq[name].add_batch(inp[0].data, out.data)

    handles = [
        linears[name].register_forward_hook(functools.partial(hook, name=name))
        for name in names
    ]
    try:
        for j in range(args.nsamples):
            fwd(inps[j].unsqueeze(0))
    finally:
        for h in handles:
            h.remove()
    return gptq


# --------------------------------------------------------------------------------------
# RTN / AWQ 的落地点：分组 round-to-nearest
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _rtn_quantize(module, args):
    """就地做分组 RTN 量化，返回一个填好 scale/zero 的 `Quantizer`（供 --save 使用）。"""
    W = module.weight.data
    state = quantize_weight(W, args.wbits, args.groupsize, args.sym)
    module.weight.data = state.dequantize(dtype=W.dtype)
    return state


# --------------------------------------------------------------------------------------
# 统一驱动
# --------------------------------------------------------------------------------------
def _subsets(linears, method, subsets):
    """把 block 内的 Linear 切成若干"顺序量化"的子集（--true-sequential）。

    只有 GPTQ 系的误差补偿真正受益于此，RTN/AWQ 直接整块处理。
    """
    names = list(linears.keys())
    if subsets and method in ('gptq', 'awq_gptq'):
        return [[n for n in sub if n in linears] for sub in subsets if any(n in linears for n in sub)]
    return [names]


@torch.no_grad()
def quantize_block(layer, inps, outs, fwd, method, args,
                   layer_idx=None, layer_kwargs=None, subsets=None):
    """对单个 Transformer block 执行 `method` 量化，并把量化后的输出写入 `outs`。

    参数
    ----
    layer        : 待量化的 block（已搬到 GPU）
    inps         : [nsamples, seqlen, hidden] 该 block 的输入
    outs         : [nsamples, seqlen, hidden] 输出缓冲（会写入量化后的 block 输出）
    fwd          : fwd(x) -> y，把一个 [1, seqlen, hidden] 张量喂给 block
    method       : 'rtn' / 'gptq' / 'awq' / 'awq_gptq'
    layer_kwargs : 调用子模块（attention/mlp）时需要的额外 kwargs（attention_mask 等）
    subsets      : --true-sequential 切分好的层名子集

    返回 layer 内每个 Linear 的 Quantizer 字典（key 是 block 内的局部名字）。
    """
    print(f'[quant] layer {layer_idx}: method={method}')
    linears = find_layers(layer)
    transform = method in ('awq', 'awq_gptq')
    use_gptq = method in ('gptq', 'awq_gptq')

    quantizers = {}

    # ---- 阶段 1: AWQ 等价变换（搜索 + 折叠），会就地修改 LN / 线性层权重 ----
    input_feat = None
    if transform:
        input_feat = collect_input_features(linears, inps, fwd, args)
        print(f'[quant] layer {layer_idx}: searching AWQ scales'
              f'{" + clips" if not getattr(args, "awq_no_clip", False) else ""} ...')
        apply_awq(
            layer, input_feat,
            wbits=args.wbits,
            groupsize=args.groupsize,
            sym=args.sym,
            n_grid=getattr(args, 'awq_n_grid', 20),
            clip=not getattr(args, 'awq_no_clip', False),
            kwargs=layer_kwargs,
        )

    # ---- 阶段 2: fp -> INT4 ----
    if use_gptq:
        for names in _subsets(linears, method, subsets):
            if not names:
                continue
            # 注意：Hessian 必须在 AWQ 变换 *之后* 收集，
            # 否则 H 对应的是变换前的激活分布，与变换后的权重不匹配。
            gptq = _build_gptq(linears, names, inps, fwd, args)
            for name in names:
                print(f'  quantizing {name} (gptq) ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize,
                    actorder=getattr(args, 'act_order', False),
                    static_groups=getattr(args, 'static_groups', False),
                )
                quantizers[name] = gptq[name].quantizer
                gptq[name].free()
            del gptq
            torch.cuda.empty_cache()
    else:
        for name, module in linears.items():
            print(f'  quantizing {name} ({method}) ...')
            quantizers[name] = _rtn_quantize(module, args)

    del input_feat
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 阶段 3: 用量化后的权重重新计算 block 输出（作为下一层的输入）----
    for j in range(inps.shape[0]):
        outs[j] = fwd(inps[j].unsqueeze(0))

    return quantizers

@torch.no_grad()
def pack_model(model, quantizers, args):
    """Replace quantized Linears with the appropriate packed inference layer."""
    from awq import AWQQuantState
    from modelutils import find_layers
    from quant import Quant3Linear, Quant4Linear, make_quant3, make_quant4

    layers = find_layers(model)
    layers = {name: layers[name] for name in quantizers}
    awq_states = {
        name: value.cpu() if isinstance(value, AWQQuantState) else value
        for name, value in quantizers.items()
    }
    use_awq4 = any(
        isinstance(value, AWQQuantState) and value.bits == 4
        for value in awq_states.values()
    )
    if use_awq4:
        groupsize = args.groupsize
        make_quant4(model, quantizers, groupsize=groupsize)
        qlayers = find_layers(model, [Quant4Linear])
        print('Packing AWQ INT4 ...')
        for name, module in qlayers.items():
            print(name)
            module.pack(layers[name], awq_states[name])
    else:
        make_quant3(model, quantizers, faster=getattr(args, 'faster_kernel', False))
        qlayers = find_layers(model, [Quant3Linear])
        print('Packing INT3 ...')
        for name, module in qlayers.items():
            print(name)
            quantizer = awq_states[name]
            module.pack(layers[name], quantizer.scale, quantizer.zero)
    print('Done.')
    return model
