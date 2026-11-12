# Repository Guidelines

## Project Structure & Module Organization

This is a compact research codebase for GPTQ model compression. `gptq.py` implements the core quantizer; `quant.py` contains quantization primitives; `modelutils.py`, `datautils.py`, and experiment entry points (`opt.py`, `bloom.py`, `llama.py`) support model and dataset utilities. CUDA kernels and their build script are `quant_cuda.cpp`, `quant_cuda_kernel.cu`, and `setup_cuda.py`. `test_kernel.py` benchmarks kernels. Zero-shot evaluation code and documentation live in `zeroShot/`.

## Build, Test, and Development Commands

Install PyTorch, Transformers, Datasets, and a CUDA toolkit matching your environment before running experiments.

- `python setup_cuda.py install` — build and install the `quant_cuda` extension.
- `CUDA_VISIBLE_DEVICES=0 python test_kernel.py` — benchmark the 3-bit CUDA kernel.
- `CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4` — evaluate an FP16 OPT model.
- `CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4` — evaluate 4-bit GPTQ.
- `CUDA_VISIBLE_DEVICES=0 python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --new-eval` — run the LLaMa workflow.

See `zeroShot/README.md` for zero-shot commands and required datasets.

## Coding Style & Naming Conventions

- Use Python 3 and 4-space indentation; match the existing style in nearby files.
- Prefer `snake_case` for functions, variables, and CLI flags; use `PascalCase` for classes such as `GPTQ`.
- Keep model-specific orchestration in its respective top-level script and shared logic in utility modules.
- Preserve command-line option names because they are documented in the README and may be used in published experiments.
- CUDA/C++ code follows the surrounding formatting conventions.

## Testing Guidelines

There is no formal pytest suite or CI configuration. Validate changes by running a small model end-to-end (for example, `facebook/opt-125m` on `c4`) and by running `test_kernel.py` after CUDA changes. Compare outputs against FP16, the `--nearest` baseline, and expected perplexity trends. Add focused scripts or update existing benchmarks when behavior changes.

## Commit & Pull Request Guidelines

Recent history uses short imperative or present-tense subjects (for example, “Added saving option for BLOOM” and “Update README.md”); keep the same concise style. Wrap lines near 72–80 characters where practical.

For pull requests, describe the changed behavior, list the exact model/dataset/GPU configuration tested, and link relevant issues. Include benchmark or perplexity output for performance-sensitive changes and call out compatibility risks clearly.

## Security & Configuration Tips

Do not commit generated checkpoints, cached datasets, model folders, or secrets. Many commands require substantial GPU memory; document the minimum tested configuration when introducing new options.

