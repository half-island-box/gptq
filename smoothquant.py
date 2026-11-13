"""SmoothQuant W8A8 fake-quantization support.

This path is intentionally independent from the repository's W4/W3 GPTQ code.
The implementation follows MIT-Han-Lab SmoothQuant: per-channel activation
scales are used to move activation outliers into the preceding LayerNorm /
RMSNorm and the paired Linear weights, after which every block Linear is
replaced by an INT8 simulated ``W8A8Linear``.

The current implementation is a numerical simulation (dequantized INT8
matmul); it is suitable for perplexity validation, not yet for INT8 speedup.
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.pytorch_utils import Conv1D as TransformersConv1D
except ImportError:  # pragma: no cover - very old Transformers
    TransformersConv1D = None


def _is_conv1d(module):
    return TransformersConv1D is not None and isinstance(module, TransformersConv1D)

__all__ = [
    'W8A8Linear', 'add_smoothquant_args', 'validate_smoothquant_args',
    'quantize_block_w8a8',
]


def _quantize_activation_per_token_absmax(x, n_bits=8):
    """Symmetric dynamic per-token activation fake quantization."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    scales = x2.abs().max(dim=1, keepdim=True)[0]
    qmax = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5) / qmax
    q = torch.clamp(torch.round(x2 / scales), -qmax, qmax) * scales
    return q.reshape(shape).to(x.dtype)


def _quantize_weight_per_channel_absmax(weight, n_bits=8):
    """Symmetric per-output-channel INT8 weight fake quantization.

    ``weight`` must have Linear orientation: ``[out_features, in_features]``.
    """
    scales = weight.abs().max(dim=1, keepdim=True)[0]
    qmax = 2 ** (n_bits - 1) - 1
    scales = scales.float().clamp(min=1e-5) / qmax
    q = torch.clamp(torch.round(weight.float() / scales), -qmax, qmax) * scales
    return q.to(weight.dtype)


class W8A8Linear(nn.Module):
    """Dequantized INT8 linear layer used for W8A8 accuracy experiments."""

    def __init__(self, in_features, out_features, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_quant_name = 'per_channel'
        self.act_quant_name = 'per_token'
        if bias is None:
            self.register_buffer('bias', None)
        else:
            self.register_buffer('bias', bias.detach().clone())

    @staticmethod
    def from_float(module):
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            weight = module.weight.data
        elif _is_conv1d(module):
            in_features = module.weight.shape[0]
            out_features = module.weight.shape[1]
            weight = module.weight.data.t()
        else:
            raise TypeError(f'Unsupported SmoothQuant layer: {type(module)!r}')

        new = W8A8Linear(in_features, out_features, getattr(module, 'bias', None))
        new.weight = nn.Parameter(
            _quantize_weight_per_channel_absmax(weight),
            requires_grad=False,
        )
        return new.to(weight.device)

    def forward(self, x):
        x = _quantize_activation_per_token_absmax(x)
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, weight_quant={self.weight_quant_name}, '
            f'act_quant={self.act_quant_name}'
        )


def add_smoothquant_args(parser):
    group = parser.add_argument_group('SmoothQuant W8A8')
    group.add_argument(
        '--w8a8', action='store_true',
        help='Enable isolated SmoothQuant W8A8 fake quantization (requires --wbits 16).'
    )
    group.add_argument(
        '--smooth-alpha', type=float, default=0.5,
        help='Balance between activation and weight smoothing difficulty.'
    )
    group.add_argument(
        '--smooth-calib-samples', type=int, default=16,
        help='Number of block calibration samples used to search activation scales.'
    )
    return group


def validate_smoothquant_args(args):
    if not args.w8a8:
        return
    if args.wbits != 16:
        raise ValueError('--w8a8 must be used with --wbits 16 to stay separate from W4/W3 GPTQ.')
    if getattr(args, 'quant_method', None) is not None:
        raise ValueError('--w8a8 cannot be combined with --quant_method.')
    if getattr(args, 'nearest', False):
        raise ValueError('--w8a8 cannot be combined with --nearest.')
    if getattr(args, 'trits', False):
        raise ValueError('--w8a8 cannot be combined with --trits.')
    if getattr(args, 'save', ''):
        raise ValueError('--save is not supported by the W8A8 path yet.')
    if getattr(args, 'load', ''):
        raise ValueError('--load is not supported by the W8A8 path yet.')
    if not 0.0 <= args.smooth_alpha <= 1.0:
        raise ValueError('--smooth-alpha must be in [0, 1].')
    if args.smooth_calib_samples < 1:
        raise ValueError('--smooth-calib-samples must be positive.')


def _linear_layers(module):
    result = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) or _is_conv1d(child):
            result[name] = child
    return result


@torch.no_grad()
def _collect_act_scales(linears, inps, fwd, n_samples):
    scales = {name: None for name in linears}

    def hook(module, inputs, output, name):
        x = inputs[0].detach()
        hidden = x.shape[-1]
        current = x.reshape(-1, hidden).abs().max(dim=0)[0].float()
        if scales[name] is None:
            scales[name] = current
        else:
            scales[name] = torch.maximum(scales[name], current)

    handles = [
        module.register_forward_hook(functools.partial(hook, name=name))
        for name, module in linears.items()
    ]
    try:
        for i in range(min(n_samples, inps.shape[0])):
            fwd(inps[i].unsqueeze(0))
    finally:
        for handle in handles:
            handle.remove()
    return scales


def _smooth_group(norm, linears, act_scales, alpha):
    """Apply the SmoothQuant equivalence to one norm and its input linears."""
    modules = [linears[name] for name in linears]
    reference = next((name for name in linears if act_scales.get(name) is not None), None)
    if reference is None:
        return False

    activation = act_scales[reference]
    device, dtype = modules[0].weight.device, modules[0].weight.dtype
    activation = activation.to(device=device, dtype=dtype)

    maxima = []
    for module in modules:
        weight = module.weight.data.t() if _is_conv1d(module) else module.weight.data
        maxima.append(weight.abs().max(dim=0)[0])
    weight_scale = torch.stack(maxima).max(dim=0)[0].clamp(min=1e-5)
    scale = (
        activation.float().pow(alpha) / weight_scale.float().pow(1 - alpha)
    ).clamp(min=1e-5).to(device=device, dtype=dtype)

    with torch.no_grad():
        if norm.weight is not None:
            norm.weight.div_(scale)
        if getattr(norm, 'bias', None) is not None:
            norm.bias.div_(scale)
        for module in modules:
            if _is_conv1d(module):
                module.weight.mul_(scale.view(-1, 1))
            else:
                module.weight.mul_(scale.view(1, -1))
    return True


def _smooth_architecture_groups(layer, act_scales, alpha):
    """Smooth the pairs for which the scale can be folded into a preceding norm."""
    groups = []

    # OPT: LayerNorm -> q/k/v and LayerNorm -> fc1.
    self_attn = getattr(layer, 'self_attn', None)
    if self_attn is not None:
        qkv = {
            f'self_attn.{name}_proj': getattr(self_attn, f'{name}_proj')
            for name in ('q', 'k', 'v')
            if hasattr(self_attn, f'{name}_proj')
        }
        if qkv:
            norm = getattr(layer, 'input_layernorm', None)
            if norm is None:
                norm = getattr(layer, 'self_attn_layer_norm', None)
            if norm is not None:
                groups.append((norm, qkv))

    # BLOOM: LayerNorm -> fused QKV Conv1D.
    self_attention = getattr(layer, 'self_attention', None)
    if self_attention is not None and hasattr(self_attention, 'query_key_value'):
        qkv = {'self_attention.query_key_value': self_attention.query_key_value}
        norm = getattr(layer, 'input_layernorm', None)
        if norm is not None:
            groups.append((norm, qkv))

    # OPT first MLP projection.
    if hasattr(layer, 'fc1'):
        groups.append((layer.final_layer_norm, {'fc1': layer.fc1}))

    mlp = getattr(layer, 'mlp', None)
    if mlp is not None and hasattr(mlp, 'dense_h_to_4h'):
        groups.append((
            layer.post_attention_layernorm,
            {'mlp.dense_h_to_4h': mlp.dense_h_to_4h},
        ))

    # LLaMA first MLP projections.
    if (
        mlp is not None
        and hasattr(mlp, 'gate_proj')
        and hasattr(mlp, 'up_proj')
    ):
        groups.append((
            layer.post_attention_layernorm,
            {'mlp.gate_proj': mlp.gate_proj, 'mlp.up_proj': mlp.up_proj},
        ))

    applied = 0
    for norm, names in groups:
        applied += int(_smooth_group(norm, names, act_scales, alpha))
    return applied


def _replace_child(module, name, new_child):
    setattr(module, name, new_child)


@torch.no_grad()
def quantize_block_w8a8(layer, inps, outs, fwd, args, layer_idx=None):
    """Calibrate, smooth, and replace one decoder block's Linear layers."""
    if args.wbits != 16:
        raise ValueError('SmoothQuant W8A8 expects --wbits 16.')
    if args.smooth_calib_samples > inps.shape[0]:
        print(
            f'[smooth] layer {layer_idx}: reducing calibration samples from '
            f'{args.smooth_calib_samples} to {inps.shape[0]}'
        )
    n_samples = min(args.smooth_calib_samples, inps.shape[0])
    linears = _linear_layers(layer)

    print(f'[smooth] layer {layer_idx}: collecting activation scales')
    act_scales = _collect_act_scales(linears, inps, fwd, n_samples)
    n_groups = _smooth_architecture_groups(layer, act_scales, args.smooth_alpha)
    print(f'[smooth] layer {layer_idx}: smoothed {n_groups} norm groups; replacing linears')

    replacements = []
    for name, module in linears.items():
        replacements.append((name, W8A8Linear.from_float(module)))

    # Replace only after traversal, so linears remains a valid snapshot.
    for name, new_module in replacements:
        parent = layer
        parts = name.split('.')
        for part in parts[:-1]:
            parent = getattr(parent, part)
        _replace_child(parent, parts[-1], new_module)

    del act_scales
    for j in range(inps.shape[0]):
        outs[j] = fwd(inps[j].unsqueeze(0))
    return {}
