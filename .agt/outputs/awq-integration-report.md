# 把 AWQ 迁移进 GPTQ 仓库：INT4 方法可切换（RTN / GPTQ / AWQ / AWQ+GPTQ）

> 目标仓库：`/Users/bj/Coding/github/gptq`（`mit-han-lab/GPTQ` 本地副本）
> 参考实现：`/Users/bj/Coding/github/llm-awq`（AWQ 算法来源）
> 状态：已实现并通过 CPU 侧的自动化验证（详见 §7）；真实模型 PPL 需要在 GPU 机器上跑（§8）

---

## 1. 目标与结果

| 编号 | 目标 | 状态 |
|---|---|---|
| G1 | 一个参数 `--quant_method {rtn,gptq,awq,awq_gptq}` 切换 INT4 方法 | ✅ |
| G2 | 三种方法共用同一套标定数据、同一套量化语义、同一套评估流程 → PPL 严格可比 | ✅ |
| G3 | 原有命令行为不变（`--wbits 4` 仍是 GPTQ，`--nearest` 仍是 RTN） | ✅ |
| G4 | 用 GPTQ README 的 Python 命令风格启动 AWQ（`python opt.py xxx c4 --wbits 4 --quant_method awq`） | ✅ |
| G5 | 不改动 CUDA kernel / 3-bit 打包 / 部署格式 | ✅ |
| G6 | 新增方法只需实现一个类/一个分支 | ✅（`quant_methods.py`） |

启动方式（与 README 原有风格一致）：

```bash
CUDA_VISIBLE_DEVICES=0 python opt.py   facebook/opt-125m c4 --wbits 4 --quant_method awq  --groupsize 128
CUDA_VISIBLE_DEVICES=0 python llama.py LLAMA_HF_FOLDER  c4 --wbits 4 --quant_method awq  --groupsize 128 --true-sequential --act-order --new-eval
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4 --quant_method awq --groupsize 128
```

---

## 2. 核心洞察：三种方法只在“前端”不同

三种方法最终产物完全一致（`Quantizer` 的 scale/zero + 反量化权重，后者是 `--save`
打包或 `test_kernel.py` 的输入）：

```
                  ┌── 钩子 A：权重预处理（可选）──┐   ┌── 钩子 B：fp → INT4 ──┐
rtn              :  identity                    →   分组 RTN
gptq             :  identity                    →   Hessian 列补偿（原有 gptq.py）
awq              :  per-channel scale + clip    →   分组 RTN
awq_gptq         :  per-channel scale + clip    →   Hessian 列补偿
```

所以实现不是“三套流水线”，而是**两个正交钩子 + 一个统一驱动**：

* 钩子 A 只在 `awq` / `awq_gptq` 下生效（`awq.py`）；
* 钩子 B 二选一：`quant.py:Quantizer`（RTN/AWQ）或 `gptq.py:GPTQ.fasterquant`（GPTQ/AWQ+GPTQ）；
* 两者都在 `quant_methods.quantize_block()` 里按固定顺序调用，顺序约束见 §5.3。

**全仓库只有一份 round 规则**：`awq.py:group_quantize()` 内部就是循环调用
`quant.py` 的 `Quantizer.find_params/quantize`，因此 RTN 与 AWQ 的量化语义逐位一致
（`test_awq.py` 用 `torch.equal` 断言了这一点），AWQ 的改进完全来自权重预处理。

---

## 3. 文件清单

| 文件 | 动作 | 内容 |
|---|---|---|
| `awq.py` | **新增**（~620 行） | AWQ 核心：缩放系数搜索/折叠、裁剪范围搜索/应用、架构自适应边界识别、`group_quantize` |
| `quant_methods.py` | **新增**（~250 行） | 方法注册与统一驱动：`add_method_args` / `resolve_method` / `quantize_block` / Hessian 收集 / RTN 落地点 |
| `test_awq.py` | **新增**（~300 行） | CPU 自测：量化语义、等价性、AWQ 优于 RTN、四方法跑通、3-bit 打包兼容、CLI 归一化 |
| `opt.py` | 改 | 删除内联 GPTQ 循环与 eval 里的 RTN 分支，改调 `quantize_block`；新增方法参数 |
| `llama.py` | 改 | 同上（保留 `--true-sequential` 的子集切分，作为 `subsets=` 传入） |
| `bloom.py` | 改 | 同上 |
| `gptq.py` | 改 1 处 | `torch.cuda.synchronize()` 加 `is_available()` 保护（CPU 自测需要），算法零改动 |
| `run_llama_experiments.py` | 改 | 增加 AWQ 一档（FP16 / RTN / GPTQ / AWQ） |
| `README.md` | 改 | 新增 “Selectable INT4 Methods” 章节 + 命令示例 |

未改动：`quant_cuda_kernel.cu`、`quant_cuda.cpp`、`setup_cuda.py`、`quant.py`、
`datautils.py`、`zeroShot/`（其内部自带 opt/bloom 驱动副本，仍是 RTN/GPTQ 两条路径）。

---

## 4. `awq.py`：从 llm-awq 迁移了什么

逐段对应（便于 review 时确认与原实现一致）：

| llm-awq 位置 | 本仓库位置 | 说明 |
|---|---|---|
| `awq/quantize/auto_scale.py:get_act_scale` | `awq.py:get_act_scale` | `mean(|x|)` per input channel |
| `auto_scale.py:scale_ln_fcs / scale_fc_fc / scale_gelu_fc` | `scale_ln_fcs / scale_fc_fc / scale_act_fc` | 三类折叠，NaN 断言保留 |
| `qmodule.py:ScaledActivation` | `ScaledActivation` | act→fc 折叠用的包装层 |
| `auto_scale.py:auto_scale_block`（按模型 isinstance 分支） | `find_scale_groups` + `_groups_llama/_groups_opt/_groups_bloom` | 用属性探测而不是 isinstance，跨 transformers 版本更稳；另加命名启发式兜底 |
| `auto_scale.py:_search_module_scale`（alpha 网格 + block 输出 MSE） | 同名函数 | 网格 `alpha ∈ {0, 1/20, …, 19/20}`，`s = mean(|x|)^alpha`，归一化到几何均值 1；每组都回滚权重 |
| `auto_scale.py:apply_scale` | `apply_scales` | 同步把 `input_feat` 除以 s（裁剪搜索要用变换后的输入） |
| `auto_clip.py:auto_clip_layer / auto_clip_block / apply_clip` | `_clip_layer / search_clips / apply_clips` | `max_shrink=0.5`、`n_grid=20`、跳过 q/k |
| `quantizer.py:pseudo_quantize_tensor` | `group_quantize`（走 `Quantizer`） | **不复制** 上游的 round 规则，改用本仓库语义 → 跨方法可比 |

### 4.1 为什么缩放有效（一句话）

分组仿射量化的误差 ∝ 组内步长 `Δ=(max-min)/15`；把某个输入通道放大 `s` 倍后该通道
权重的相对误差变成 `Δ/(s·|w|)`，而它在输出上的贡献被 `|x|` 加权。于是应当放大的
是“激活大 / 权重相对小”的通道——通过把 `s` 折进前一个 LayerNorm / Linear / 激活，
这一步是**数学等价**的重参数化，不引入额外计算。

### 4.2 边界识别（每个 block 3~4 条）

| 架构 | attn_in | attn_out | mlp_in | mlp_out |
|---|---|---|---|---|
| LLaMA/Qwen2 类 | `input_layernorm → {q,k,v}`（inspect=attn） | `v_proj → o_proj`（形状相等时） | `post_attention_layernorm → {gate,up}`（inspect=mlp） | `up_proj → down_proj` |
| OPT | `self_attn_layer_norm → {q,k,v}` | `v_proj → out_proj` | `final_layer_norm → fc1` | `fc1 → fc2`（ReLU 正齐次，可折进 fc1 输出行） |
| BLOOM | `input_layernorm → query_key_value` | 跳过（上游注释：MQ 结构下不安全） | `post_attention_layernorm → dense_h_to_4h` | `gelu_impl → dense_4h_to_h`（GELU 非齐次，必须折到激活**之后**） |

GQA（`v_proj.out != o_proj.in`）时自动跳过 `attn_out`（对应 llm-awq PR #67）。
无法识别时明确报错并建议改用 `--quant_method gptq/rtn`，不会静默产生错误结果。

---

## 5. `quant_methods.py`：统一驱动

### 5.1 接口

```python
quantize_block(layer, inps, outs, fwd, method, args,
               layer_idx=None, layer_kwargs=None, subsets=None) -> dict[str, Quantizer]
```

* `fwd(x)` 由各入口脚本提供（OPT: `attention_mask`；LLaMA: `+position_ids`；BLOOM: `+alibi`），
  这样模型相关代码留在各自的 script 里；
* `layer_kwargs` 传给被搜索的子模块（`self_attn` / `mlp`），并做了签名适配：
  自动丢弃 `past_key_values/use_cache/cache_position`、补齐无默认值的必填参数
  （BLOOM 的 `BloomAttention(residual=None)`）、子模块调用失败时退化为单层粒度搜索；
* 返回值 `{层内局部名: Quantizer}`，由脚本加全局前缀（保持 `model.layers.3.mlp.up_proj` 这类键名不变）。

### 5.2 时间线（单层）

```
 [rtn]        ─────────────────────────────────────────────► 量化 → 重算输出   (1 次前向)
 [gptq]       标定+Hessian(1 次前向) ──► 列补偿量化 ──► 重算输出              (2 次前向，同原实现)
 [awq]        标定(16 样本) ─► 缩放搜索 ─► 裁剪搜索 ─► 分组 RTN ─► 重算输出
 [awq_gptq]   标定(16) ─► 缩放+裁剪 ─► Hessian(重新收集！) ─► 列补偿 ─► 重算输出
```

### 5.3 一个必须遵守的顺序约束

AWQ 的 `apply_scales` 会把 LN/线性层权重和 `input_feat` 一起改掉，因此
**GPTQ 的 Hessian 必须在 AWQ 变换之后重新收集**，否则 H 对应的是变换前的激活分布，
与变换后的权重不匹配。`awq_gptq` 分支里显式重建了 `GPTQ` 对象（代价：多一趟标定前向），
而不是去“缩放”旧的 H——正确性优先。

### 5.4 与 `--true-sequential` 的关系

`--true-sequential` 的子集切分只对 GPTQ 系有意义：`_subsets()` 在
`gptq/awq_gptq` 下使用脚本传入的 `subsets`，在 `rtn/awq` 下自动合并为一个整体
（AWQ 的缩放边界跨子块，必须整块搜索）。`awq_gptq` + `--true-sequential` 时，
每个子集的 H 都在前面子集量化完成后收集（符合 true-sequential 的原意）。

---

## 6. CLI

新增参数（三个入口脚本共用 `add_method_args`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--quant_method` | `None` → `gptq` | `rtn` / `gptq` / `awq` / `awq_gptq` |
| `--nearest` | 关 | 保留，等价于 `--quant_method rtn` |
| `--awq-calib-samples` | 16 | AWQ 搜索用多少个标定样本（每层 `n × seqlen` 个 token） |
| `--awq-n-grid` | 20 | alpha 网格粒度 |
| `--awq-no-clip` | 关 | 跳过裁剪搜索（只做缩放，速度约快 3×） |

归一化逻辑（`resolve_method`）：`--wbits 16` → 不量化；`--nearest` → `rtn`；
未指定方法 → `gptq`（保持兼容）。非法值由 argparse 拒绝并列出可选值；
`--trits` 只在 `gptq` 下允许（RTN/AWQ 的 `group_quantize` 不支持三值）。
`--save`（3-bit 打包）与 `--groupsize > 0` 组合会给出明确报错而不是静默出错。

结果 CSV 的 `quant_method` 列现在输出 `FP16 / RTN / GPTQ / AWQ / AWQ_GPTQ`。

---

## 7. 已完成的验证（CPU，无需 GPU / 无需下载模型）

`python test_awq.py`（本次实跑环境：torch 2.15 CPU + transformers 5.22，迷你 block）：

```
  [ok] group_quantize 与 Quantizer 语义一致、量化幂等
  [ok] 缩放前后 block 输出一致（相对偏差 3.56e-08）      ← 等价缩放的前提
  [ok] int4 误差 awq=1.0226e-05 < rtn=2.1574e-05         ← AWQ 确实优于 RTN
  [ok] rtn / gptq / awq / awq_gptq 均跑通
        rtn 2.91e-05 | gptq 6.73e-06 | awq 1.22e-05 | awq_gptq 6.69e-06
  [ok] rtn/awq 的 scale/zero 可直接用于 3-bit 打包（--save）
  [ok] --nearest / --quant_method 归一化正确
```

另外用**真实的 transformers 迷你模型**（2 层 LLaMA / OPT / BLOOM，随机初始化）跑了
完整的分层量化循环（Catcher → 逐层 `quantize_block` → 下游 LM loss），三种架构
× 四种方法全部跑通，验证了：

* `find_scale_groups` 与真实 `LlamaDecoderLayer / OPTDecoderLayer / BloomBlock`
  的属性名一致；
* BLOOM 的 `BloomAttention(residual, alibi)` 签名、`gelu_impl` 激活折叠；
* OPT 的 `fc1/fc2` + ReLU 折叠；
* transformers ≥5 传入的 `Cache` 对象不会污染子模块调用（随机权重下 loss 变化 <0.02，
  因为随机初始化模型的 loss 平坦，数量级不代表真实精度）。

> 注意：随机初始化的迷你模型**不能**用来判断 AWQ/GPTQ 的真实收益，那需要真实
> checkpoint + 真实标定数据，见 §8。

### 7.2 入口脚本的 CLI 级 dry-run（已验证）

把 `get_opt/get_llama/get_bloom` 和 `get_loaders` 换成迷你模型与假数据后，
直接执行 `opt.py` / `llama.py` / `bloom.py` 的 `__main__`（完整 argparse → 标定 →
量化 → 评估 → CSV 流程），3 个脚本 × 4 种方法 **全部跑通**：

```
opt.py   --wbits 4 --groupsize 32 --quant_method {rtn,gptq,awq,awq_gptq}   ✔
llama.py --wbits 4 --groupsize 32 --true-sequential --quant_method ...     ✔
bloom.py --wbits 4 --groupsize 32 --quant_method ...                       ✔
```

同时验证了边界行为：

| 场景 | 结果 |
|---|---|
| `--wbits 16` | 跳过量化，直接走 FP16 评估 |
| `--wbits 3 --save model.pt` | 3-bit 打包成功（`opt_pack3`/`llama_pack3`/`bloom_pack3` 都能拿到 quantizers） |
| `--quant_method awq --groupsize 32 --save` | **量化开始前**就 `parser.error` 提示改用 `--groupsize -1` |
| `--quant_method awq --trits` | `parser.error: --trits 只在 gptq 下支持` |
| `--quant_method xxx` | argparse 拒绝并列出可选值 |
| 结果 CSV 的 `quant_method` 列 | 正确写入 `FP16 / RTN / GPTQ / AWQ / AWQ_GPTQ` |

---

## 8. 在 GPU 机器上应补的实验

固定条件：`--wbits 4 --groupsize 128`、同样 `--seed`、同样标定数据
（`--nsamples 128`，AWQ 默认 `--awq-calib-samples 16`）、评估 wikitext2 PPL。

```bash
for m in "" "--nearest" "--quant_method gptq" "--quant_method awq" "--quant_method awq_gptq"; do
  CUDA_VISIBLE_DEVICES=0 python llama.py LLAMA_HF_FOLDER wikitext2 --wbits 4 --groupsize 128 \
      --true-sequential --act-order --new-eval $m
done
```

期望：`AWQ ≈ GPTQ ≪ RTN`，且 `awq_gptq ≤ min(AWQ, GPTQ)`（验证互补性）。
重点回归项：

1. **GPTQ bit-exact**：`--quant_method gptq` 与改动前的同一命令得到的 PPL 应完全一致
   （驱动逻辑逐行等价，见 §5.2/§5.4）；
2. **RTN 一致性**：`--nearest` 与 `--quant_method rtn` 等价（现在 RTN 也能用 `--groupsize`）；
3. **AWQ 折叠无损**：`--awq-no-clip --q_backend` 之外，可临时把 `sym`/`groupsize` 固定，
   验证“只折叠不量化”时 PPL 与 FP16 一致（`test_awq.py` 已在 block 级别证明）。

---

## 9. 已知限制 / 后续可做

1. `zeroShot/` 目录有自己的一份 opt/bloom 驱动副本，未接线 `--quant_method`；
   如需 zero-shot 评测 AWQ，应把 `models/opt.py`、`models/bloom.py` 的量化循环也换成
   `quantize_methods.quantize_block`（约 30 行）。
2. `--save`（3-bit 打包）只在 `--groupsize -1` 下支持（这是原有打包格式的限制）；
   4-bit 目前仍是 fake-quant 评测 + 3-bit kernel 打包，AWQ 没有改变这一点。
3. `desc_act / act-order` 与 AWQ 组合时由 GPTQ 内部置换处理，未做“折叠置换”优化；
   `awq_gptq + act-order` 可用，但没有为 4-bit kernel 做 act-order 的运行时 gather
   （本仓库本来也没有 4-bit kernel）。
4. 内存：AWQ 每层需要保存 `awq_calib_samples × seqlen × hidden`（fp16，CPU）的激活，
   LLaMA-7B（seqlen 2048、16 样本）约 0.27 GB/层 × 7 个线性层 ≈ 1.9 GB 主机内存，
   可通过 `--awq-calib-samples` 调低。
5. AWQ 的 alpha 网格搜索是每层 20 次 block 前向（上游同样），比 GPTQ 慢；
   用 `--awq-no-clip` 可以省掉裁剪搜索。
