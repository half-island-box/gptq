"""Focused CPU tests for the isolated SmoothQuant W8A8 path."""

import torch
from torch import nn

from smoothquant import W8A8Linear, _smooth_group, quantize_block_w8a8


class Args:
    wbits = 16
    smooth_alpha = 0.5
    smooth_calib_samples = 2


class LlamaLikeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(5)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(5, 5)
        self.self_attn.k_proj = nn.Linear(5, 5)
        self.self_attn.v_proj = nn.Linear(5, 5)
        self.post_attention_layernorm = nn.LayerNorm(5)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(5, 8)
        self.mlp.up_proj = nn.Linear(5, 8)
        self.mlp.down_proj = nn.Linear(8, 5)

    def forward(self, x):
        hidden = self.self_attn.v_proj(self.input_layernorm(x))
        projected = (
            self.mlp.gate_proj(self.post_attention_layernorm(x))
            + self.mlp.up_proj(self.post_attention_layernorm(x))
        )
        return hidden + self.mlp.down_proj(projected)


def test_smoothing_is_mathematically_equivalent():
    torch.manual_seed(7)
    norm = nn.LayerNorm(6)
    first = nn.Linear(6, 4)
    second = nn.Linear(6, 4)
    x = torch.randn(3, 6)
    expected = first(norm(x)) + second(norm(x))
    act_scales = x.abs().max(dim=0)[0].float() + 0.1

    assert _smooth_group(
        norm, {'first': first, 'second': second},
        {'first': act_scales, 'second': act_scales}, alpha=0.5,
    )
    actual = first(norm(x)) + second(norm(x))
    assert torch.allclose(expected, actual, atol=2e-5, rtol=2e-4)


def test_block_is_replaced_by_w8a8_linears():
    torch.manual_seed(0)
    block = LlamaLikeBlock().eval()
    x = torch.randn(2, 3, 5)
    expected = block(x)

    quantize_block_w8a8(
        block, x, torch.empty_like(x), lambda value: block(value), Args(), layer_idx=0
    )
    actual = block(x)
    assert torch.isfinite(actual).all()
    assert actual.shape == expected.shape
    assert sum(type(m).__name__ == 'W8A8Linear' for m in block.modules()) == 6
    assert (actual - expected).norm() / expected.norm() < 0.02


def test_w8a8_linear_quantizes_dynamic_activations():
    linear = nn.Linear(4, 3)
    module = W8A8Linear.from_float(linear).eval()
    x = torch.randn(5, 4)
    output = module(x)
    assert output.shape == (5, 3)
    assert module.weight_quant_name == 'per_channel'
    assert module.act_quant_name == 'per_token'


if __name__ == '__main__':
    test_smoothing_is_mathematically_equivalent()
    test_block_is_replaced_by_w8a8_linears()
    test_w8a8_linear_quantizes_dynamic_activations()
    print('SmoothQuant tests passed.')
