"""AWQ 量化核心（从 mit-han-lab/llm-awq 迁移到本仓库）。

Paper: Lin et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM
Compression and Acceleration", MLSys 2024 (Best Paper).
https://arxiv.org/abs/2306.00978    参考实现: https://github.com/mit-han-lab/llm-awq

这里只迁移 AWQ 中"量化之前"的那一半，即两个正交的权重预处理：

  1. 等价缩放 (equivalent scaling)：对每个 Linear 的 *输入通道* 做 per-channel 缩放 s，
     同时把 s 折叠进前一个算子（LayerNorm / 上一个 Linear / 激活函数），
     数学上完全等价，但让"重要通道"（激活大、权重相对小）在分组量化时误差更小。
     缩放指数按 block 输出 MSE 做网格搜索（`search_scales`）。

  2. 权重裁剪 (weight clipping)：在 [|w|max, (1-max_shrink)|w|max] 内搜索每组的
     clamp 上限，用激活加权的输出误差挑选最优值（`search_clips`）。

真正的"fp -> INT4"这一步没有搬过来：本仓库沿用 `quant.py` 里的 `Quantizer`，
因此 RTN / GPTQ / AWQ 三种方法共用同一套 round 规则（见 `quant_methods.py`），
权重格式、packing、kernel 全部不变。

对外主要接口
------------
`group_quantize(w, wbits, groupsize, sym)`        分组 affine 量化（返回反量化权重 + scale/zero）
`find_scale_groups(block)`                        自动识别一个 Transformer block 里的缩放边界
`search_scales(block, input_feat, ...)`           搜索缩放系数（返回 scales_list）
`apply_scales(block, scales_list, input_feat)`    把缩放折叠进相邻算子，并同步修正 input_feat
`search_clips(block, input_feat, ...)`            搜索权重裁剪上限（返回 clip_list）
`apply_clips(block, clip_list)`                   应用裁剪
`apply_awq(block, input_feat, ...)`               上面四步的组合（搜索 + 就地应用）
"""

import inspect as _inspect
from collections import namedtuple
from dataclasses import dataclass

import torch
import torch.nn as nn

from quant import Quantizer, quantize

__all__ = [
    'ScaledActivation',
    'ScaleGroup',
    'find_scale_groups',
    'group_quantize',
    'AWQQuantState',
    'quantize_weight',
    'get_act_scale',
    'search_scales',
    'apply_scales',
    'search_clips',
    'apply_clips',
    'apply_awq',
    'get_op_by_name',
    'get_op_name',
    'set_op_by_name',
]


# --------------------------------------------------------------------------------------
# 在 block 内按名字定位 / 替换子模块（与 llm-awq/awq/utils/module.py 等价）
# --------------------------------------------------------------------------------------
def get_op_by_name(module, op_name):
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f'Cannot find op {op_name!r} in {type(module).__name__}')


def get_op_name(module, op):
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f'Cannot find op {type(op).__name__} in {type(module).__name__}')


def set_op_by_name(module, name, new_module):
    levels = name.split('.')
    mod = module
    for lv in levels[:-1]:
        mod = mod[int(lv)] if lv.isdigit() else getattr(mod, lv)
    setattr(mod, levels[-1], new_module)


def _is_norm(m):
    return isinstance(m, nn.LayerNorm) or 'norm' in type(m).__name__.lower()


def _is_activation(m):
    if isinstance(m, (nn.GELU, nn.SiLU, nn.ReLU, nn.LeakyReLU)):
        return True
    cls = type(m).__name__.lower()
    return any(h in cls for h in ('gelu', 'silu', 'swish', 'relu', 'activation'))


# 标定时绝不该传给子模块的 kwargs（KV cache / 输出开关）
_SKIP_KWARGS = ('past_key_values', 'past_key_value', 'use_cache', 'cache_position')


def _call_with_kwargs(module, x, kwargs):
    """调用 module(x, **kwargs)，自动丢掉签名里不接受的 kwargs，并补齐必填参数。

    这样做是为了让同一份搜索代码同时适配 OPTAttention / LlamaAttention /
    LlamaMLP / BloomAttention（后者的签名里有未使用的 `residual`）等。
    """
    kwargs = {k: v for k, v in (kwargs or {}).items() if k not in _SKIP_KWARGS}
    try:
        params = _inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        params = {}
    has_varkw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    if has_varkw or not params:
        call_kwargs = dict(kwargs)
    else:
        call_kwargs = {k: v for k, v in kwargs.items() if k in params}
        positional = [
            n for n, p in params.items()
            if n != 'self' and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        for name in positional[1:]:
            p = params[name]
            if p.default is p.empty and name not in call_kwargs:
                call_kwargs[name] = None
    out = module(x, **call_kwargs)
    return out[0] if isinstance(out, tuple) else out


class ScaledActivation(nn.Module):
    """包裹一个激活函数并在其后做 per-channel 缩放的包装层（对应 AWQ 的 act->fc 折叠）。"""

    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


# --------------------------------------------------------------------------------------
# 量化语义：显式整数 q + 反量化；保留旧 group_quantize 接口
# --------------------------------------------------------------------------------------
@dataclass
class AWQQuantState:
    """An explicit affine integer-quantization result for a weight tensor."""

    q: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    maxq: int
    group_size: int
    original_shape: torch.Size
    original_dtype: torch.dtype
    symmetric: bool = False
    bits: int = 4

    def dequantize(self, dtype=None):
        n_group = self.scales.shape[1]
        w2 = self.q.reshape(-1, n_group, self.group_size)
        scales = self.scales[:, :, None]
        zeros = self.zeros[:, :, None]
        w = scales * (w2.to(scales.dtype) - zeros)
        return w.reshape(self.original_shape).to(
            self.original_dtype if dtype is None else dtype
        )

    def cpu(self):
        return AWQQuantState(
            q=self.q.cpu(), scales=self.scales.cpu(), zeros=self.zeros.cpu(),
            maxq=self.maxq, group_size=self.group_size,
            original_shape=self.original_shape, original_dtype=self.original_dtype,
            symmetric=self.symmetric, bits=self.bits,
        )

    def to_legacy(self):
        """Return the historical group_quantize layout: (Wq, [n_group, out], ...)."""
        return (
            self.dequantize(),
            self.scales.t().contiguous(),
            self.zeros.t().contiguous(),
        )

    @property
    def scale(self):
        if self.scales.shape[1] != 1:
            raise NotImplementedError('pack3 requires per-row quantization')
        return self.scales[:, 0].reshape(-1, 1)

    @property
    def zero(self):
        if self.zeros.shape[1] != 1:
            raise NotImplementedError('pack3 requires per-row quantization')
        return self.zeros[:, 0].reshape(-1, 1)


@torch.no_grad()
def quantize_weight(w, wbits, groupsize=-1, sym=False, mse=False):
    """Quantize a weight to integers plus per-output/per-group affine parameters."""
    org_shape = w.shape
    w2 = w.reshape(-1, org_shape[-1])
    out_features, in_features = w2.shape
    gs = in_features if groupsize in (None, -1, 0) else int(groupsize)
    if gs <= 0 or in_features % gs != 0:
        raise ValueError(f'groupsize={groupsize} cannot divide in_features={in_features}')

    n_group = in_features // gs
    maxq = 2 ** int(wbits) - 1
    q = torch.empty(w2.shape, dtype=torch.uint8, device=w2.device)
    scales = torch.empty((out_features, n_group), dtype=w2.dtype, device=w2.device)
    zeros = torch.empty_like(scales)

    for g in range(n_group):
        sl = slice(g * gs, (g + 1) * gs)
        chunk = w2[:, sl]
        quantizer = Quantizer()
        quantizer.configure(wbits, perchannel=True, sym=sym, mse=mse)
        quantizer.find_params(chunk, weight=True)
        scale = quantizer.scale.reshape(-1)
        zero = quantizer.zero.reshape(-1)
        q[:, sl] = (torch.round(chunk / scale[:, None]) + zero[:, None]).clamp_(
            0, maxq
        ).to(torch.uint8)
        scales[:, g] = scale
        zeros[:, g] = zero

    return AWQQuantState(
        q=q.reshape(org_shape), scales=scales, zeros=zeros, maxq=maxq,
        group_size=gs, original_shape=org_shape, original_dtype=w.dtype,
        symmetric=sym, bits=int(wbits),
    )


@torch.no_grad()
def group_quantize(w, wbits, groupsize=-1, sym=False, mse=False):
    """Group affine quantize -> dequantized weight, preserving the old API."""
    return quantize_weight(w, wbits, groupsize, sym, mse).to_legacy()


# --------------------------------------------------------------------------------------
@torch.no_grad()
def get_act_scale(x):
    """per-input-channel 的激活幅度（沿 token 维取 mean(|x|)）。"""
    return x.detach().abs().reshape(-1, x.shape[-1]).float().mean(dim=0).cpu()


# 等价缩放的折叠（把 s 分配进前一个算子）
# --------------------------------------------------------------------------------------
@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    """LayerNorm/RMSNorm -> FC：ln 的 weight/bias 除以 s，FC 的输入通道乘 s。"""
    if not isinstance(fcs, (list, tuple)):
        fcs = [fcs]
    scales = scales.to(device=ln.weight.device, dtype=ln.weight.dtype)

    ln.weight.div_(scales)
    if getattr(ln, 'bias', None) is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0, 'NaN after LN scaling'
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0, 'NaN after FC scaling'


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    """FC -> FC：fc1 的输出行除以 s，fc2 的输入通道乘 s。

    `fc1.weight[-n:]` 的写法来自上游实现，用于 gate/up 融合（up_proj 的输出来自最后一段行）。
    """
    assert isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear)
    scales = scales.to(device=fc1.weight.device, dtype=fc1.weight.dtype)
    n = scales.numel()
    assert fc1.weight.shape[0] >= n

    fc1.weight[-n:].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias[-n:].div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0, 'NaN after fc1 scaling'
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0, 'NaN after fc2 scaling'


@torch.no_grad()
def scale_act_fc(act, fc, scales):
    """激活 -> FC：激活后的输出由 ScaledActivation 除以 s，fc 的输入通道乘 s。"""
    assert isinstance(fc, nn.Linear)
    fc.weight.mul_(scales.view(1, -1).to(device=fc.weight.device, dtype=fc.weight.dtype))
    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0, 'NaN after act->fc scaling'


# --------------------------------------------------------------------------------------
# 各架构的缩放边界
#
# 每个 ScaleGroup 描述一条"边界"，LLM 的 block 里通常有 3~4 条：
#   norm -> {q,k,v} / {gate,up}      (attn_in / mlp_in)
#   v_proj -> o_proj                  (attn_out)
#   up_proj(or act) -> down_proj      (mlp_out)
# --------------------------------------------------------------------------------------
ScaleGroup = namedtuple('ScaleGroup', 'kind prev_op layers inspect')


def _n(block, module):
    return get_op_name(block, module)


def find_scale_groups(block):
    """返回当前 block 的缩放边界列表（自动适配 LLaMA / OPT / BLOOM 及通用结构）。"""
    for matcher in (_groups_llama, _groups_opt, _groups_bloom):
        groups = matcher(block)
        if groups:
            return groups
    groups = _groups_generic(block)
    if not groups:
        raise NotImplementedError(
            f'AWQ: 无法自动识别 {type(block).__name__} 的缩放边界。'
            ' 可以改用 --quant_method gptq / rtn，或扩展 awq.find_scale_groups。'
        )
    print(f'[awq] 未识别的 block 类型 {type(block).__name__}，已按命名启发式推断缩放边界。')
    return groups


def _groups_llama(block):
    attn = getattr(block, 'self_attn', None)
    mlp = getattr(block, 'mlp', None)
    norm_in = getattr(block, 'input_layernorm', None)
    norm_post = getattr(block, 'post_attention_layernorm', None)
    if attn is None or mlp is None or norm_in is None or not hasattr(attn, 'q_proj'):
        return None

    groups = []
    qkv = [n for n in ('q_proj', 'k_proj', 'v_proj') if hasattr(attn, n)]
    groups.append(ScaleGroup(
        'attn_in', _n(block, norm_in),
        tuple(_n(block, getattr(attn, n)) for n in qkv), _n(block, attn),
    ))
    o_proj, v_proj = getattr(attn, 'o_proj', None), getattr(attn, 'v_proj', None)
    # GQA 下 v_proj 与 o_proj 形状不同，折叠不成立（上游 PR #67）
    if o_proj is not None and v_proj is not None and v_proj.weight.shape == o_proj.weight.shape:
        groups.append(ScaleGroup('attn_out', _n(block, v_proj), (_n(block, o_proj),), None))

    mlp_in = [n for n in ('gate_proj', 'up_proj') if hasattr(mlp, n)]
    if norm_post is not None and mlp_in:
        groups.append(ScaleGroup(
            'mlp_in', _n(block, norm_post),
            tuple(_n(block, getattr(mlp, n)) for n in mlp_in), _n(block, mlp),
        ))
    down = getattr(mlp, 'down_proj', None)
    prev = getattr(mlp, 'up_proj', None) or getattr(mlp, 'gate_proj', None)
    if down is not None and prev is not None:
        groups.append(ScaleGroup('mlp_out', _n(block, prev), (_n(block, down),), None))
    return groups


def _groups_opt(block):
    norm_in = getattr(block, 'self_attn_layer_norm', None)
    norm_out = getattr(block, 'final_layer_norm', None)
    attn = getattr(block, 'self_attn', None)
    fc1, fc2 = getattr(block, 'fc1', None), getattr(block, 'fc2', None)
    if norm_in is None or attn is None or fc1 is None or fc2 is None:
        return None

    groups = []
    qkv = [n for n in ('q_proj', 'k_proj', 'v_proj') if hasattr(attn, n)]
    groups.append(ScaleGroup(
        'attn_in', _n(block, norm_in),
        tuple(_n(block, getattr(attn, n)) for n in qkv), _n(block, attn),
    ))
    out_proj = getattr(attn, 'out_proj', None)
    if out_proj is not None and hasattr(attn, 'v_proj'):
        groups.append(ScaleGroup('attn_out', _n(block, attn.v_proj), (_n(block, out_proj),), None))
    if norm_out is not None:
        groups.append(ScaleGroup('mlp_in', _n(block, norm_out), (_n(block, fc1),), None))
    # OPT 的激活是 ReLU（正齐次），可以直接折叠进 fc1 的输出行
    groups.append(ScaleGroup('mlp_out', _n(block, fc1), (_n(block, fc2),), None))
    return groups


def _groups_bloom(block):
    norm_in = getattr(block, 'input_layernorm', None)
    norm_post = getattr(block, 'post_attention_layernorm', None)
    attn = getattr(block, 'self_attention', None)
    mlp = getattr(block, 'mlp', None)
    if norm_in is None or attn is None or mlp is None:
        return None
    qkv = getattr(attn, 'query_key_value', None)
    if qkv is None:
        return None

    groups = [ScaleGroup('attn_in', _n(block, norm_in), (_n(block, qkv),), _n(block, attn))]
    fc1 = getattr(mlp, 'dense_h_to_4h', None)
    fc2 = getattr(mlp, 'dense_4h_to_h', None)
    act = getattr(mlp, 'gelu_impl', None) or getattr(mlp, 'gelu', None) or getattr(mlp, 'act', None)
    if norm_post is not None and fc1 is not None:
        groups.append(ScaleGroup('mlp_in', _n(block, norm_post), (_n(block, fc1),), _n(block, mlp)))
    # BLOOM 的激活是 GELU（非齐次），必须折叠到激活之后，因此 prev_op 是激活层
    if fc2 is not None and act is not None:
        groups.append(ScaleGroup('mlp_out', _n(block, act), (_n(block, fc2),), None))
    return groups


_QKV_NAMES = ('q_proj', 'k_proj', 'v_proj', 'qkv_proj', 'query_key_value',
              'wqkv', 'c_attn', 'in_proj', 'qkv', 'wq', 'wk', 'wv')
_ATTN_OUT_NAMES = ('o_proj', 'out_proj', 'dense', 'c_proj', 'wo', 'proj')
_MLP_IN_NAMES = ('fc1', 'up_proj', 'gate_proj', 'c_fc', 'dense_h_to_4h',
                 'gate_up_proj', 'w1', 'w3', 'fc_in')
_MLP_OUT_NAMES = ('fc2', 'down_proj', 'c_proj', 'dense_4h_to_h', 'w2', 'fc_out')


def _groups_generic(block):
    """命名启发式兜底：适配 transformers 里常见的 x_proj / fc1 / fc2 命名。"""
    linears = [(n, m) for n, m in block.named_modules() if isinstance(m, nn.Linear)]
    norms = [(n, m) for n, m in block.named_modules() if _is_norm(m)]
    acts = [(n, m) for n, m in block.named_modules() if _is_activation(m)]
    if not linears:
        return []

    qkv, attn_out, mlp_in, mlp_out = [], [], [], []
    for name, _ in linears:
        leaf = name.split('.')[-1].lower()
        in_attn = 'attn' in name.lower() or 'attention' in name.lower()
        if leaf in _QKV_NAMES:
            qkv.append(name)
        elif leaf in _MLP_IN_NAMES:
            mlp_in.append(name)
        elif leaf in _MLP_OUT_NAMES:
            (attn_out if in_attn else mlp_out).append(name)
        elif leaf in _ATTN_OUT_NAMES:
            (attn_out if in_attn else mlp_out).append(name)

    groups = []

    def _parent(name):
        parent = name.rsplit('.', 1)[0] if '.' in name else ''
        return parent

    if qkv and norms:
        groups.append(ScaleGroup('attn_in', norms[0][0], tuple(qkv), _parent(qkv[0]) or None))
    if mlp_in and len(norms) > 1:
        groups.append(ScaleGroup('mlp_in', norms[-1][0], tuple(mlp_in), _parent(mlp_in[0]) or None))
    if attn_out and qkv:
        groups.append(ScaleGroup('attn_out', qkv[-1], (attn_out[0],), None))
    if mlp_out:
        prev = acts[-1][0] if acts else (mlp_in[0] if mlp_in else None)
        if prev:
            groups.append(ScaleGroup('mlp_out', prev, (mlp_out[-1],), None))
    return groups


# --------------------------------------------------------------------------------------
# 阶段一：缩放系数的搜索
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _search_module_scale(block, linears, x, inspect_mod, wbits, groupsize, sym,
                         n_grid=20, mse=False, kwargs=None, verbose=True):
    """在一个 block 上网格搜索 alpha（s = x_max^alpha），最小化 `inspect_mod` 的输出 MSE。"""
    params = next(block.parameters())
    dev, dtype = params.device, params.dtype
    x = x.to(device=dev, dtype=dtype)

    org_out = _call_with_kwargs(inspect_mod, x, kwargs)

    x_max = get_act_scale(x)
    snapshot = [(fc, {k: v.detach().clone() for k, v in fc.state_dict().items()}) for fc in linears]

    best_error = float('inf')
    best_scales = None
    best_alpha = -1.0
    history = []

    for i in range(n_grid):
        alpha = i / n_grid                      # 0 即"完全不缩放"的 RTN 基线
        scales = x_max.pow(alpha).clamp(min=1e-4)
        # 归一化（几何平均为 1），避免改变量化 scale 的量级
        scales = scales / (scales.max() * scales.min()).sqrt()
        scales = scales.view(-1).to(device=dev, dtype=dtype)

        for fc in linears:
            fc.weight.mul_(scales.view(1, -1))
            q = quantize_weight(fc.weight.data, wbits, groupsize, sym, mse).dequantize()
            fc.weight.data = (q / scales.view(1, -1)).to(dtype)

        out = _call_with_kwargs(inspect_mod, x, kwargs)
        loss = (org_out - out).float().pow(2).mean().item()
        history.append(loss)
        if loss < best_error:
            best_error, best_scales, best_alpha = loss, scales.detach().clone(), alpha

        for fc, sd in snapshot:                 # 回滚，避免误差累积
            fc.load_state_dict(sd)

    if best_scales is None:
        raise RuntimeError(f'AWQ scale search failed, history={history}')
    assert torch.isnan(best_scales).sum() == 0, 'NaN scales'
    if verbose:
        print(f'    awq scale: alpha={best_alpha:.2f} mse={best_error:.4e}')
    return best_scales


@torch.no_grad()
def search_scales(block, input_feat, wbits, groupsize=-1, sym=False, n_grid=20,
                  mse=False, kwargs=None, groups=None, verbose=True):
    """搜索所有缩放边界上的 s。

    返回 scales_list: [(prev_op_name, (layer_names...), scales), ...]
    注意：本函数 **不修改** block，只做搜索；应用见 `apply_scales`。
    """
    if groups is None:
        groups = find_scale_groups(block)

    scales_list = []
    for group in groups:
        if group.layers[0] not in input_feat:
            if verbose:
                print(f'    awq scale: skip {group.kind}（缺少 {group.layers[0]} 的标定输入）')
            continue
        linears = [get_op_by_name(block, n) for n in group.layers]
        x = input_feat[group.layers[0]]
        inspect_mod = get_op_by_name(block, group.inspect) if group.inspect else linears[0]
        try:
            scales = _search_module_scale(
                block, linears, x, inspect_mod, wbits, groupsize, sym,
                n_grid=n_grid, mse=mse, kwargs=kwargs, verbose=verbose,
            )
        except TypeError as err:
            # 少数架构的子模块无法单独调用（缺少必需参数），退化为单层粒度的搜索
            if inspect_mod is linears[0]:
                raise
            if verbose:
                print(f'    awq scale: {group.kind} 无法调用 {type(inspect_mod).__name__}'
                      f'（{err}），退化为单层搜索')
            scales = _search_module_scale(
                block, linears, x, linears[0], wbits, groupsize, sym,
                n_grid=n_grid, mse=mse, kwargs=kwargs, verbose=verbose,
            )
        scales_list.append((group.prev_op, group.layers, scales))
    return scales_list


@torch.no_grad()
def apply_scales(block, scales_list, input_feat=None):
    """把搜索到的 s 折叠进前一个算子（LN / Linear / 激活），并同步修正标定输入。"""
    for prev_name, layer_names, scales in scales_list:
        prev = get_op_by_name(block, prev_name)
        layers = [get_op_by_name(block, n) for n in layer_names]

        if isinstance(prev, nn.Linear):
            assert len(layers) == 1, 'fc->fc 折叠只支持单个下游层'
            scale_fc_fc(prev, layers[0], scales)
        elif _is_norm(prev):
            scale_ln_fcs(prev, layers, scales)
        elif _is_activation(prev):
            set_op_by_name(block, prev_name, ScaledActivation(prev, scales))
            scale_act_fc(prev, layers[0], scales)
        else:
            raise NotImplementedError(f'AWQ: 不支持的前驱算子 {type(prev).__name__}')

        if input_feat is not None:
            for name in layer_names:
                if name in input_feat:
                    inp = input_feat[name]
                    input_feat[name] = (inp / scales.view(1, -1).to(
                        device=inp.device, dtype=inp.dtype))


# --------------------------------------------------------------------------------------
# 阶段二：权重裁剪范围的搜索
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _quant_dequant_groups(w2d, wbits, sym):
    """对 [rows, groupsize] 做分组量化并反量化（量化语义与 group_quantize 完全一致）。"""
    quantizer = Quantizer()
    quantizer.configure(wbits, perchannel=True, sym=sym, mse=False)
    return quantize_weight(w2d, wbits, w2d.shape[-1], sym).dequantize()


@torch.no_grad()
def _clip_layer(w, input_feat, wbits, groupsize, sym,
                n_grid=20, max_shrink=0.5, n_sample_token=256, oc_batch_size=64):
    """对单个 Linear 搜索每组权重的 clamp 上限（激活加权输出误差最小）。

    oc_batch_size / n_sample_token 只影响显存峰值：`feat * w` 的广播中间张量
    大小约为 oc_batch_size * n_sample_token * in_features 个元素。
    """
    assert w.dim() == 2
    co = w.shape[0]
    gs = w.shape[1] if groupsize in (None, -1, 0) else int(groupsize)
    assert w.shape[1] % gs == 0

    feat = input_feat.reshape(-1, input_feat.shape[-1])
    feat = feat.reshape(1, feat.shape[0], -1, gs)
    step = max(1, feat.shape[1] // max(1, n_sample_token))
    feat = feat[:, ::step].to(device=w.device, dtype=torch.float32)

    w4 = w.reshape(co, 1, -1, gs).to(torch.float32)
    best_max_vals = []
    for s in range(0, co, oc_batch_size):
        wb = w4[s:s + oc_batch_size]
        org_max = wb.abs().amax(dim=-1, keepdim=True)                   # [b,1,ng,1]
        best_max = org_max.clone()
        min_err = torch.full_like(org_max, float('inf'))
        org_out = (feat * wb).sum(dim=-1)                               # [b,T,ng]

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max * (1 - i_s / n_grid)
            cur_w = torch.clamp(wb, -max_val, max_val)
            q = _quant_dequant_groups(cur_w.reshape(-1, gs), wbits, sym).reshape(cur_w.shape)
            cur_out = (feat * q).sum(dim=-1)
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_err.shape)
            better = err < min_err
            min_err[better] = err[better]
            best_max[better] = max_val[better]

        best_max_vals.append(best_max)

    return torch.cat(best_max_vals, dim=0).squeeze(1)                   # [co, ng, 1]


@torch.no_grad()
def search_clips(block, input_feat, wbits, groupsize=-1, sym=False, n_grid=20,
                 max_shrink=0.5, n_sample_token=256, verbose=True,
                 skip=('q_', 'k_', 'query', 'key', 'Wqkv')):
    """对 block 内所有（除 q/k 之外）的 Linear 搜索裁剪上限。

    跳过 q/k 的原因：它们后面紧跟 QK^T，权重被 clamp 后输出误差无法用单层的
    输出误差近似（与上游实现一致）。
    """
    clip_list = []
    for name, m in block.named_modules():
        if not isinstance(m, nn.Linear) or name not in input_feat:
            continue
        if any(k in name for k in skip):
            if verbose:
                print(f'    awq clip: skip {name}')
            continue
        max_val = _clip_layer(m.weight.data, input_feat[name], wbits, groupsize, sym,
                              n_grid=n_grid, max_shrink=max_shrink,
                              n_sample_token=n_sample_token)
        clip_list.append((name, max_val))
    return clip_list


@torch.no_grad()
def apply_clips(block, clip_list):
    """把搜索到的 clamp 上限应用到权重上。"""
    for name, max_val in clip_list:
        layer = get_op_by_name(block, name)
        max_val = max_val.to(device=layer.weight.device, dtype=layer.weight.dtype)
        org_shape = layer.weight.shape
        w = layer.weight.data.reshape(max_val.shape[0], max_val.shape[1], -1)  # [co, n_group, gs]
        w = torch.clamp(w, -max_val, max_val)
        layer.weight.data = w.reshape(org_shape)


# --------------------------------------------------------------------------------------
# 组合：搜索 + 就地应用
# --------------------------------------------------------------------------------------
@torch.no_grad()
def apply_awq(block, input_feat, wbits, groupsize=-1, sym=False, n_grid=20,
              clip=True, clip_n_grid=20, clip_max_shrink=0.5, kwargs=None, verbose=True):
    """AWQ 等价变换全流程（就地修改 block，并同步修正 input_feat）。"""
    scales_list = search_scales(
        block, input_feat, wbits, groupsize=groupsize, sym=sym,
        n_grid=n_grid, kwargs=kwargs, verbose=verbose,
    )
    apply_scales(block, scales_list, input_feat=input_feat)

    clip_list = []
    if clip:
        clip_list = search_clips(block, input_feat, wbits, groupsize=groupsize, sym=sym,
                                 n_grid=clip_n_grid, max_shrink=clip_max_shrink,
                                 verbose=verbose)
        apply_clips(block, clip_list)
    return {'scale': scales_list, 'clip': clip_list}
