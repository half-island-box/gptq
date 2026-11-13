# GPTQ

This repository contains the code for the ICLR 2023 paper [GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323). 
The current release includes the following features:

* An efficient implementation of the GPTQ algorithm: `gptq.py`
* Selectable INT4 quantization methods — RTN / GPTQ / AWQ / AWQ+GPTQ (`--quant_method`): `awq.py`, `quant_methods.py`
* Compressing all models from the OPT and BLOOM families to 2/3/4 bits, including weight grouping: `opt.py`, `bloom.py`, `zeroShot/`
* Evaluating the perplexity of quantized models on several language generation tasks: `opt.py`, `bloom.py`
* Evaluating the performance of quantized models on several ZeroShot tasks: `zeroShot/`
* A 3-bit quantized matrix full-precision vector product CUDA kernel: `quant_cuda_kernel.cu`, `quant_cuda.cpp`, `setup_cuda.py`
* Benchmarking code for individual matrix-vector products and for language generation with quantized models: `test_kernel.py`, `opt.py`

## New Features

Update July 2023:

* Added `--static-groups` options which determines all group-grids in advance rather than dynamically during quantization, which has the effect that `--act-order` does not require any inference changes (that may cause slowdown) when used together with this option.

Together with the camera ready version of the paper we have added several updates to this repository:

* Slightly adjusted preprocessing of C4 and PTB for more realistic evaluations (used in our updated results); can be activated via the flag `--new-eval`.
* Optimized 3bit kernels, which are considerably faster especially on the A100, e.g. 1.9x -> 3.25x generation speedup for OPT-175B; can be activated via `--faster-kernel`.
* A minimal LlaMa integration (for more complete features see the [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) repository), which demonstrates two new tricks:`--act-order` (quantizing columns in order of decreasing activation size) and `--true-sequential` (performing sequential quantization even within a single Transformer block). Those fix GPTQ's strangely bad performance on the 7B model (from 7.15 to 6.09 Wiki2 PPL) and lead to slight improvements on most models/settings in general.

Here is a summary of LLaMa results:

| Wiki2 PPL | FP16 | 4bit-RTN | 4bit-GPTQ | 3bit-RTN | 3bit-GPTQ | 3g128-GPTQ |
|:---------:|:----:|:--------:|:---------:|:--------:|:---------:|:----------:|
| LLaMa-7B  | 5.68 | 6.29     | **6.09**  | 25.54    | **8.07**  | 6.61       |
| LLaMa-13B | 5.09 | 5.53     | **5.36**  | 11.40    | **6.63**  | 5.62       |
| LLaMa-30B | 4.10 | 4.54     | **4.45**  | 14.89    | **5.69**  | 4.80       |
| LLaMa-65B | 3.53 | 3.92     | **3.84**  | 10.59    | **5.04**  | 4.17       |

Here is a sample command:

```
python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --new-eval
```

Run FP16, RTN, and GPTQ sequentially for Llama-7B. The model is read from a local
checkpoint path when one is given, from the local ModelScope cache when available,
and otherwise downloaded from ModelScope. The default `llama-7b` alias first uses
the verified mirror `shakechen/Llama-2-7b-hf`, with fallbacks for other caches:

```
CUDA_VISIBLE_DEVICES=0 python run_llama_experiments.py c4 --wbits 4 --true-sequential --act-order --new-eval
```

The `--act-order` heuristic also dramatically improves accuracy on the OPT-66B outlier model: 9.55 to 9.34 and 14.16 to 9.95 PPL on Wiki2 for 4bit and 3bit, respectively.

## Selectable INT4 Methods: RTN / GPTQ / AWQ

`--quant_method` chooses how the INT4 weights are produced. All four methods share the
same quantization semantics (`quant.py:Quantizer`), the same calibration data and the same
evaluation path, so their perplexities are directly comparable.

| `--quant_method` | Weight pre-processing | Needs calibration | Extra cost per layer |
|:--|:--|:--|:--|
| `rtn` | none (round-to-nearest, group-wise) | no | none |
| `gptq` (default) | none; Hessian-based column compensation | yes (`--nsamples`, 128) | 1 forward pass + Cholesky |
| `awq` | AWQ equivalent scaling + weight clipping, then group-wise RTN | yes (`--awq-calib-samples`, 16) | 20 forward passes per scale grid + clip search |
| `awq_gptq` | AWQ equivalent scaling + clipping, then GPTQ | yes (128 + 16) | as above, plus GPTQ |

Notes:

* `--nearest` is kept as an alias of `--quant_method rtn`, and `--wbits 4` without
  `--quant_method` still means GPTQ — existing commands behave exactly as before.
* AWQ is a port of [llm-awq](https://github.com/mit-han-lab/llm-awq) (`awq.py`). It
  searches a per-input-channel scale `s` (grid search over `s = mean(|x|)^alpha`,
  minimizing the block output MSE), folds `s` into the preceding LayerNorm / Linear /
  activation, searches an activation-aware weight clipping range, and only then quantizes.
  Only the weight-preprocessing part is ported: the INT4 format, packing and kernels are
  unchanged.
* `--awq-calib-samples` (default 16) sets how many calibration samples feed the AWQ
  search, `--awq-n-grid` (default 20) the alpha grid size, `--awq-no-clip` skips the
  clipping search.
* `--true-sequential`, `--act-order` and `--static-groups` only affect the GPTQ step:
  they are ignored by `rtn`/`awq` and honoured by `gptq`/`awq_gptq`.
* `--save` now produces a true AWQ INT4 packed checkpoint for
  `--quant_method awq --wbits 4`. `qweight` stores two 4-bit values per byte and
  `scales`/`zeros` retain one entry per output channel per input group. CUDA
  inference unpacks/dequantizes inside `vecquant4matmul`; a CPU reference path is
  available for tests. The legacy 3-bit format still requires
  `--wbits 3 --groupsize -1`.
* The scripts under `zeroShot/` keep their own copies of the OPT/BLOOM drivers and only
  support the RTN (`--nearest`) and GPTQ paths.
* `opt.py`, `bloom.py`, and `llama.py` also support an isolated SmoothQuant W8A8
  accuracy path with `--w8a8`. It keeps `--wbits 16`, smooths activation outliers
  into paired norms/weights, and simulates per-channel INT8 weights plus
  per-token dynamic INT8 activations. This path is for perplexity validation and
  does not yet use a packed INT8 kernel.

```bash
# one method at a time (OPT)
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --quant_method rtn
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --quant_method gptq --groupsize 128
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --quant_method awq  --groupsize 128
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --quant_method awq_gptq --groupsize 128

# LLaMA / BLOOM
CUDA_VISIBLE_DEVICES=0 python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --quant_method awq --groupsize 128 --true-sequential --act-order --new-eval
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4 --quant_method awq --groupsize 128

# FP16 / RTN / GPTQ / AWQ in one shot (shared results.csv)
CUDA_VISIBLE_DEVICES=0 python run_llama_experiments.py c4 --wbits 4 --true-sequential --act-order --new-eval

# SmoothQuant W8A8 numerical validation (fake-INT8 matmul)
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --w8a8 --smooth-alpha 0.5
CUDA_VISIBLE_DEVICES=0 python llama.py LLAMA_HF_FOLDER c4 --w8a8 --smooth-alpha 0.5
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --w8a8 --smooth-alpha 0.5

# CPU-only sanity checks (no checkpoint needed)
python test_smoothquant.py
python test_awq.py
```

## Dependencies

* `torch`: 2.3.0, CUDA 12.x wheel
* `transformers`: 4.40.2
* `datasets`: 2.19.2
* `modelscope`: 1.18.0, used for model and dataset downloads
* CUDA 12.1 toolkit or newer for compiling the 3-bit PyTorch CUDA extension
* Install the Python dependencies with `pip install -r requirements.txt`

All experiments were run on a single 80GB NVIDIA A100. However, most experiments will work on a GPU with a lot less memory as well.

## Language Generation

### OPT

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --nearest
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 [--groupsize 1024]
````

To run other OPT models replace `opt-125m` with one of: `opt-350m`, `opt-1.3b`, `opt-2.7b`, `opt-6.7b`, `opt-13b`, `opt-66b`.
For the 175B-parameter mode, you have to request access from Meta and then convert it to a local HuggingFace checkpoint using their scripts in `metaseq`.
Once you have such a checkpoint, simply pass its path instead of `facebook/opt-125m`. 

AWQ (and AWQ+GPTQ) uses the same command with `--quant_method awq` / `--quant_method awq_gptq`;
see [Selectable INT4 Methods](#selectable-int4-methods-rtn--gptq--awq).

### BLOOM

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4 --nearest
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4 [--groupsize 1024]
````

To run other BLOOM models replace `bloom-560m` with one of: `bloom-1b1`, `bloom-1b7`, `bloom-3b`, `bloom-7b1`, `bloom`.

## ZeroShot

See `zeroShot/` folder.

## 3-bit CUDA Kernels 

```
# Install kernels (a CUDA 12.x nvcc must be available)
python setup_cuda.py install

# Benchmark and verify the kernels. If no GPU is visible during the build,
# set TORCH_CUDA_ARCH_LIST to the target compute capability first.
CUDA_VISIBLE_DEVICES=0 python test_kernel.py

# Benchmark language generation with 3-bit OPT-175B:
# OPT175B denotes the name of the folder with the HuggingFace OPT-175b checkpoint (see above)

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python opt.py OPT175B c4 --wbits 3 --save opt175-3bit.pt
# Benchmark generating a 128 token sequence with the saved model
CUDA_VISIBLE_DEVICES=0 python opt.py OPT175B c4 --load opt175b-3bit.pt --benchmark 128
# Benchmark FP16 baseline, note that the model will be split across all listed GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python opt.py OPT175B c4 --benchmark 128
```

Please note that our 3-bit kernels are currently only optimized for OPT-175B running on 1xA100 or 2xA6000 and may thus yield suboptimal performance on smaller models or on other GPUs.

## Cite

If you found this work useful, please consider citing:

```
@article{frantar-gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers}, 
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  year={2022},
  journal={arXiv preprint arXiv:2210.17323}
}
```
