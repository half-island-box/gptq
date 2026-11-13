import numpy as np
import torch
import torch.nn as nn


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse # 是否用均方误差来搜索最优scale
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits: # 三值量化
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


try:
    import quant_cuda
    _HAS_QUANT_CUDA = True
except ImportError:
    quant_cuda = None
    _HAS_QUANT_CUDA = False
    print('CUDA extension not installed; Quant4Linear CPU fallback will be used.')

# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class Quant3Linear(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) 

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype
            if self.faster:
                x = x.half()
                quant_cuda.vecquant3matmul_faster(x, self.qweight, y, self.scales, self.zeros)
            else:
                x = x.float()
                quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError('Only supports a single token currently.')

def make_quant3(module, names, name='', faster=False):
    if isinstance(module, Quant3Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant3Linear(tmp.in_features, tmp.out_features, faster=faster)
            )
    for name1, child in module.named_children():
        make_quant3(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

# Packed 4-bit format: each uint8 holds two little-endian nibbles along in_features.
# qweight: [out_features, in_features / 2], scales/zeros: [out_features, n_groups].
class Quant4Linear(nn.Module):
    def __init__(self, infeatures, outfeatures, groupsize=-1):
        super().__init__()
        if infeatures % 2:
            raise ValueError('Quant4Linear requires an even in_features')
        if groupsize in (None, -1, 0):
            groupsize = infeatures
        if groupsize <= 0 or infeatures % groupsize:
            raise ValueError('Quant4Linear groupsize must divide in_features')

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.groupsize = int(groupsize)
        self.register_buffer('quant_format', torch.tensor(4, dtype=torch.uint8))
        self.register_buffer(
            'qweight', torch.zeros((outfeatures, infeatures // 2), dtype=torch.uint8)
        )
        self.register_buffer(
            'scales',
            torch.zeros((outfeatures, infeatures // self.groupsize)),
        )
        self.register_buffer(
            'zeros', torch.zeros((outfeatures, infeatures // self.groupsize))
        )
        self.register_buffer('bias', torch.zeros(outfeatures))

    @torch.no_grad()
    def pack(self, linear, state):
        if state.bits != 4:
            raise ValueError(f'Quant4Linear expects 4-bit state, got {state.bits}')
        if tuple(linear.weight.shape) != tuple(state.original_shape):
            raise ValueError('linear/state shape mismatch')
        if linear.in_features != self.infeatures or linear.out_features != self.outfeatures:
            raise ValueError('linear/module feature dimensions mismatch')
        if state.group_size != self.groupsize:
            raise ValueError('state/module groupsize mismatch')

        self.scales = state.scales.detach().clone()
        self.zeros = state.zeros.detach().clone()
        if linear.bias is not None:
            self.bias = linear.bias.detach().clone()

        q = state.q.detach().to(torch.uint8).cpu()
        q = q.reshape(self.outfeatures, self.infeatures)
        self.qweight = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()

    def _dequantized_weight(self, dtype=None):
        q = torch.empty(
            (self.outfeatures, self.infeatures),
            dtype=torch.uint8,
            device=self.qweight.device,
        )
        q[:, 0::2] = self.qweight & 0xF
        q[:, 1::2] = self.qweight >> 4
        scales = self.scales[:, :, None]
        zeros = self.zeros[:, :, None]
        w = scales * (q.reshape(
            self.outfeatures, -1, self.groupsize
        ).to(scales.dtype) - zeros)
        dtype = self.scales.dtype if dtype is None else dtype
        return w.reshape(self.outfeatures, self.infeatures).to(dtype)

    def forward(self, x):
        if x.shape[-1] != self.infeatures:
            raise ValueError(f'expected ... x {self.infeatures}, got {x.shape[-1]}')
        outshape = list(x.shape)
        outshape[-1] = self.outfeatures
        dtype = x.dtype
        x2 = x.reshape(-1, self.infeatures)

        if x2.is_cuda:
            if not _HAS_QUANT_CUDA or not hasattr(quant_cuda, 'vecquant4matmul'):
                raise RuntimeError(
                    'AWQ 4-bit CUDA inference requires rebuilding quant_cuda with vecquant4matmul'
                )
            xf = x2.float().contiguous()
            y = self.bias.float().expand(x2.shape[0], self.outfeatures).clone()
            scales = self.scales.float().contiguous()
            zeros = self.zeros.float().contiguous()
            quant_cuda.vecquant4matmul(
                xf, self.qweight.contiguous(), y, scales, zeros, self.groupsize
            )
            return y.to(dtype).reshape(outshape)

        # Correctness/reference path for CPU-only machines. It unpacks only while
        # computing and does not replace the model's packed qweight with a float copy.
        y = torch.nn.functional.linear(x2, self._dequantized_weight(dtype), self.bias.to(dtype))
        return y.reshape(outshape)


def make_quant4(module, names, name='', groupsize=-1):
    if isinstance(module, Quant4Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr,
                Quant4Linear(tmp.in_features, tmp.out_features, groupsize=groupsize),
            )
    for name1, child in module.named_children():
        make_quant4(
            child, names, name + '.' + name1 if name != '' else name1, groupsize=groupsize
        )
