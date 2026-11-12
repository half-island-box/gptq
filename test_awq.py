"""AWQ / RTN / GPTQ 集成自测（可在 CPU 上运行，不需要 GPU 与真实模型）。

    python test_awq.py

覆盖内容：
  1. `group_quantize` 与 `quant.py` 的 `Quantizer` 语义一致，且量化是幂等的；
  2. AWQ 的等价缩放**不改变** block 的输出（数学等价性，这是整个方法的前提）；
  3. AWQ 缩放 + 裁剪后，int4 量化的 block 输出误差明显小于纯 RTN；
  4. `quantize_block` 对 rtn / gptq / awq / awq_gptq 四种方法都能跑通，
     量化后的权重落在量化格点上、无 NaN，并打印 block 输出误差对比表；
  5. 3-bit 的 `Quant3Linear.pack` 与 `rtn` 产出的 scale/zero 兼容。
"""

import types

import torch
import torch.nn as nn

import awq
import quant_methods as qm
from quant import Quantizer, quantize


HIDDEN = 64
GROUP = 32
SEQ = 48
NSAMPLES = 4


# --------------------------------------------------------------------------------------
# 一个结构上等价于 LLaMA block 的迷你 block（CPU 上可跑，用于验证算法而不是精度）
# --------------------------------------------------------------------------------------
class TinyAttention(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, attention_mask=None):
        # 只保留形状/非线性依赖，不实现真正的 attention；
        # 关键点：输出必须只对 v 线性（真实模型里 o_proj 的输入就是 attention
        # 加权的 v），否则 v_proj -> o_proj 的缩放折叠不成立。
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        gate = torch.sigmoid(q + k)
        return self.o_proj(v * gate)


class TinyMLP(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = TinyAttention(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.mlp = TinyMLP(hidden)

    def forward(self, x, attention_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask=attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


def make_block(seed=0, outlier_channels=4):
    """构造一个带"激活离群通道"的 block —— 这正是 AWQ 要处理的场景。"""
    torch.manual_seed(seed)
    block = TinyBlock().eval()
    with torch.no_grad():
        for p in block.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return block, outlier_channels


def make_inputs(outlier_channels=4, seed=1):
    torch.manual_seed(seed)
    x = 0.5 * torch.randn(NSAMPLES, SEQ, HIDDEN)
    # 少数通道的激活幅度远大于其他通道（LLM 中常见的 outlier）
    for c in range(outlier_channels):
        x[..., c * 7] *= 60.0
    return x


def clone(block):
    import copy
    return copy.deepcopy(block)


def block_output(block, x):
    with torch.no_grad():
        return torch.cat([block(x[j:j + 1]) for j in range(x.shape[0])], dim=0)


def make_args(**overrides):
    base = dict(
        wbits=4, groupsize=GROUP, sym=False, nsamples=NSAMPLES,
        percdamp=0.01, act_order=False, static_groups=False, trits=False,
        awq_calib_samples=NSAMPLES, awq_n_grid=20, awq_no_clip=False,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------------------
# 1. 量化语义
# --------------------------------------------------------------------------------------
def test_group_quantize_matches_quantizer():
    torch.manual_seed(0)
    W = torch.randn(16, HIDDEN)

    q, scales, zeros = awq.group_quantize(W, 4, GROUP, sym=False)
    assert q.shape == W.shape
    assert scales.shape == (HIDDEN // GROUP, 16)

    ref = torch.zeros_like(W)
    for g in range(HIDDEN // GROUP):
        sl = slice(g * GROUP, (g + 1) * GROUP)
        quantizer = Quantizer()
        quantizer.configure(4, perchannel=True, sym=False, mse=False)
        quantizer.find_params(W[:, sl], weight=True)
        ref[:, sl] = quantize(W[:, sl], quantizer.scale, quantizer.zero, quantizer.maxq)
        assert torch.equal(scales[g], quantizer.scale.reshape(-1))
        assert torch.equal(zeros[g], quantizer.zero.reshape(-1))
    assert torch.equal(q, ref), 'group_quantize 与 Quantizer 语义不一致'

    # 幂等性：对已经量化过的权重再量化一次结果不变
    q2, _, _ = awq.group_quantize(q, 4, GROUP, sym=False)
    assert torch.allclose(q, q2, atol=1e-6), '量化不幂等'

    # 逐行（groupsize=-1）与 --nearest 的现有实现一致
    q_row, _, _ = awq.group_quantize(W, 4, -1, sym=False)
    quantizer = Quantizer()
    quantizer.configure(4, perchannel=True, sym=False, mse=False)
    quantizer.find_params(W, weight=True)
    ref_row = quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq)
    assert torch.equal(q_row, ref_row), 'per-row RTN 与原有 --nearest 不一致'
    print('  [ok] group_quantize 与 Quantizer 语义一致、量化幂等')


# --------------------------------------------------------------------------------------
# 2. 等价缩放不改变 block 输出
# --------------------------------------------------------------------------------------
def test_scaling_is_equivalent():
    block, outliers = make_block()
    x = make_inputs(outliers)
    args = make_args()

    groups = awq.find_scale_groups(block)
    kinds = [g.kind for g in groups]
    assert kinds == ['attn_in', 'attn_out', 'mlp_in', 'mlp_out'], kinds

    before = block_output(block, x)
    # 只做缩放、不做裁剪/量化：必须严格等价
    scales_list = awq.search_scales(
        block, {n: v.detach().clone() for n, v in
                qm.collect_input_features(qm.find_layers(block), x, lambda t: block(t), args).items()},
        wbits=4, groupsize=GROUP, sym=args.sym, n_grid=2, verbose=False,
    )
    awq.apply_scales(block, scales_list)
    after = block_output(block, x)
    err = (before - after).abs().max().item() / before.abs().max().item()
    assert err < 1e-4, f'等价缩放改变了 block 输出，相对误差 {err}'
    print(f'  [ok] 缩放前后 block 输出一致（相对偏差 {err:.2e}）')


# --------------------------------------------------------------------------------------
# 3. AWQ 优于纯 RTN
# --------------------------------------------------------------------------------------
def test_awq_beats_rtn():
    block, outliers = make_block()
    x = make_inputs(outliers)
    ref = block_output(block, x)
    args = make_args()

    def rel_mse(model):
        with torch.no_grad():
            out = block_output(model, x)
        return ((out - ref) ** 2).mean().item() / (ref ** 2).mean().item()

    errs = {}
    for method in ('rtn', 'awq'):
        model = clone(block)
        outs = torch.zeros_like(x)
        qm.quantize_block(model, x, outs, lambda t, m=model: m(t), method, args, layer_idx=0)
        errs[method] = rel_mse(model)
    assert errs['awq'] < errs['rtn'], f'AWQ 没有改善: {errs}'
    print(f'  [ok] int4 误差 awq={errs["awq"]:.4e} < rtn={errs["rtn"]:.4e}')


# --------------------------------------------------------------------------------------
# 4. 四种方法端到端（含 GPTQ 的 Hessian 路径）
# --------------------------------------------------------------------------------------
def test_all_methods_run():
    block, outliers = make_block()
    x = make_inputs(outliers)
    ref = block_output(block, x)
    args = make_args(groupsize=-1)   # GPTQ 路径用逐行量化，便于验证

    def rel_mse(out):
        return ((out - ref) ** 2).mean().item() / (ref ** 2).mean().item()

    results = {}
    for method in qm.METHODS:
        model = clone(block)
        outs = torch.zeros_like(x)
        quantizers = qm.quantize_block(
            model, x, outs, lambda t, m=model: m(t), method, args, layer_idx=0
        )
        assert set(quantizers) == {'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                                   'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj',
                                   'mlp.down_proj'}, sorted(quantizers)
        for name, module in qm.find_layers(model).items():
            assert torch.isnan(module.weight).sum() == 0, f'{method}:{name} NaN'
            _check_on_grid(module.weight, quantizers[name], f'{method}:{name}')
        results[method] = rel_mse(outs)

    print('  block 输出相对 MSE（越小越好）:')
    for method, err in results.items():
        print(f'    {method:<9} {err:.4e}')
    # 四种方法都应显著优于"不量化"以外的随机解，至少不能爆炸
    for method, err in results.items():
        assert err < 0.5, f'{method} 误差过大: {err}'
    print('  [ok] rtn / gptq / awq / awq_gptq 均跑通')


def _check_on_grid(W, quantizer, tag=''):
    """量化后的权重必须落在 dequantize(q) 的格点上（再量化一次不变）。"""
    Wq = quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq)
    assert torch.allclose(W, Wq, atol=1e-4), f'{tag}: 权重不在量化格点上'


# --------------------------------------------------------------------------------------
# 5. --save（3-bit 打包）兼容性
# --------------------------------------------------------------------------------------
def test_pack3_compatible():
    block, _ = make_block()
    model = clone(block)
    args = make_args(wbits=3, groupsize=-1)

    linears = qm.find_layers(model)
    quantizers = {}
    for name, module in linears.items():
        quantizers[name] = qm._rtn_quantize(module, args)

    from quant import Quant3Linear
    for name, module in linears.items():
        packed = Quant3Linear(module.in_features, module.out_features)
        packed.pack(module, quantizers[name].scale, quantizers[name].zero)
        assert packed.qweight.shape[0] == module.in_features // 32 * 3
    print('  [ok] rtn/awq 的 scale/zero 可直接用于 3-bit 打包（--save）')


# --------------------------------------------------------------------------------------
# 6. 命令行归一化
# --------------------------------------------------------------------------------------
def test_resolve_method():
    def r(**kw):
        base = dict(wbits=4, quant_method=None, nearest=False, trits=False)
        base.update(kw)
        return qm.resolve_method(types.SimpleNamespace(**base))

    assert r() == 'gptq'
    assert r(nearest=True) == 'rtn'
    assert r(quant_method='awq') == 'awq'
    assert r(quant_method='awq_gptq') == 'awq_gptq'
    assert r(wbits=16) is None
    try:
        r(quant_method='awq', trits=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError('--trits 应该只在 gptq 下可用')
    print('  [ok] --nearest / --quant_method 归一化正确')


if __name__ == '__main__':
    torch.manual_seed(0)
    tests = [
        test_group_quantize_matches_quantizer,
        test_scaling_is_equivalent,
        test_awq_beats_rtn,
        test_all_methods_run,
        test_pack3_compatible,
        test_resolve_method,
    ]
    for fn in tests:
        print(f'== {fn.__name__}')
        fn()
    print('\nALL TESTS PASSED')
