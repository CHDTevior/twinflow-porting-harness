# TwinFlow 迁移与蒸馏验收 Harness

日期：2026-05-08

目标：把 `https://github.com/inclusionAI/TwinFlow` 迁移到任意已有预训练生成项目，并形成可重复、可审查、可验收的训练与评估流程。

## 0. 强制规则

1. 先生成并验证 `INPUT_CONTRACT.json`；缺数据路径、输出路径、checkpoint、conda env、eval 样本、Slurm 信息时，agent 必须继续问用户，不能开始改代码。
2. 先审查原项目和官方 TwinFlow，再写代码。
3. 所有路径使用绝对路径记录。
4. 所有长训前必须通过 smoke。
5. 蒸馏后推理默认 `cfg=0`，除非明确证明需要开启。
6. 每次可视化输出第一列必须是 `cond` 条件图片。
7. 每个质量问题都要配对应截图或 crop，不能只写文字。
8. 训练日志分析遇到 resume 重复 step 时，按 step 去重取最后一次。
9. eval 输出目录默认必须新建，非空目录只能在显式 override 并写入 manifest 时复用。
10. formal 或阶段成果必须经过 clean-context review；review 不能代替数据，但数据结果也不能跳过 review。

## 0.1 输入合同门禁

每个新项目先运行：

```bash
python -m twinflow_porting_harness init ...
python -m twinflow_porting_harness validate <INPUT_CONTRACT.json>
```

合同必须至少明确：

- `project_root`
- `dataset_root`
- `train_metadata`
- `eval_metadata`
- `output_root`
- `run_name`
- `base_checkpoint`
- `conda_env`
- `official_twinflow_ref`
- `original_inference_entry`
- `condition_type`
- `cluster`
- `gpu_count`
- `eval_sample_spec`
- `eval_artifact_spec`
- `slurm_log_dir` and `slurm_partition`/`sbatch_template` when `cluster=slurm`

未满足时启动流程停在提问阶段。不要猜测路径，不要用当前目录下看起来像的 CSV，不要复用旧 output。

## 1. 输入信息模板

新项目开始时先填：

```text
project_root =
official_twinflow_root =
official_twinflow_commit =
base_checkpoint =
dataset_root =
train_metadata =
eval_metadata =
output_root =
run_name =
condition_type = image/text/multiview/control/other
original_inference = sampler + NFE + cfg
distilled_eval = few 1/4/8 or project-specific few/any/mul protocol
cluster = Slurm/DDP/FSDP/local
conda_env =
slurm_log_dir =
eval_sample_spec =
eval_artifact_spec =
```

## 2. 迁移前代码审查

必须审查：

- 模型 forward 是否能加入 TwinFlow two-time interface。
- 新增 TwinFlow 参数是否有从旧模型可解释 warm-start 的策略。
- checkpoint warm-start 是否只允许审查过的新 key/missing key。
- 原 scheduler、`sigma_min`、`t_rescale` 是否和 TwinFlow 兼容。
- dataset/collate 是否会按长度、稀疏体素数、图像尺寸排序。
- branch mask 是否会和排序后的样本统计相关。
- microbatch 是否改变 branch 分布。
- DDP/FSDP 下 EMA 是否一致。
- cond / uncond / neg_cond 是否和预训练一致。
- logging 是否能写 NumPy/Torch scalar。
- save/resume 是否覆盖 student、optimizer、EMA、TwinFlow state。
- eval 是否固定 sample、seed、noise、cond。
- eval 是否会在数据读取失败时 silent retry 换样本。
- eval manifest 是否记录 sample id、seed、mode、路径和样本 hash。
- 旧 checkpoint 进入 TwinFlow wrapper 时，新增 key 是否被正确初始化而不是随机残留。

## 3. 核心迁移实现

TwinFlow full 训练语义：

```text
t ~ Beta(time_dist_ctrl[0], time_dist_ctrl[1]) * time_dist_ctrl[2]
tt = t - U(0,1) * consistc_ratio * t
t_min = t - U(0,1) * t * min(0.05, consistc_ratio)

e2e: t=1, tt=0
mul: tt=t_min
any: tt=tt
adv: make fake x from model, then use negative-time branch
```

推荐 full branch：

```text
probs = {"e2e": 1, "mul": 1, "any": 1, "adv": 1}
```

实现要点：

- local batch 最好至少 `4`。
- official partition 先在完整 local batch 上构造，再 microbatch。
- 如果原 collate 会排序，branch assignment 要随机化，避免 branch 和样本大小相关。
- `dist_match` 只对 e2e 分支生效；target 不应反传。
- RCGM/enhanced target 若使用 EMA teacher，要保存和恢复 EMA。
- LoRA 是可选路径；本次 TRELLIS 迁移使用 full-parameter，不使用 LoRA。

## 4. 配置基线

保守 baseline：

```text
batch_size_per_gpu=4
lr=1e-5
optimizer=AdamW
fp16-master
8 GPU DDP
probs={"e2e":0,"mul":1,"any":1,"adv":0}
estimate_order=2
enhanced_ratio=2.0
dist_match=0
image_free=false
t_rescale=false
cfg_eval=0
```

TwinFlow distribution-distillation：

```text
batch_size_per_gpu=4
batch_split=2 or 4
lr=1e-5
optimizer=AdamW
fp16-master
8 GPU DDP
probs={"e2e":1,"mul":1,"any":1,"adv":1}
dist_match=0.5
estimate_order=2
enhanced_ratio=2.0 initially
image_free=false initially
t_rescale=false
cfg_eval=0
```

OOM 时第一优先级只改：

```text
batch_split: 2 -> 4
```

不要同时改 branch、lr、batch、dist_match，否则实验不可归因。

## 5. Smoke 顺序

长训前按顺序跑：

1. 输入合同 validate。
2. JSON/YAML 配置解析。
3. shell 脚本 `bash -n`。
4. Python `py_compile`。
5. synthetic 1-step。
6. real-data fetch preflight。
7. 单卡真实 1-step。
8. 多卡 DDP 1-step。
9. save/resume 1-step，强制 `i_log=1`。
10. eval smoke，输出 `condition + GT/target if available + original + distilled probe`。
11. decode/export smoke，输出 `eval_artifact_spec` 指定产物和 manifest。

失败就停，不要启动长训。

## 6. 训练监控

每次检查：

```bash
squeue -j <jobid> -o '%.18i %.9P %.35j %.8u %.2t %.10M %.6D %R'
tail -n 40 <output_dir>/log.txt
tail -n 120 slurm_logs/slurm_output_<jobid>.log
srun --jobid=<jobid> --overlap --ntasks=1 nvidia-smi
```

日志必须检查：

- step 是否增长。
- loss/mse/grad_norm 是否 finite。
- branch count 是否符合配置。
- e2e 时是否出现 `e2e_loss`。
- dist-match 打开时是否实际进入 e2e 分支。
- GPU 是否忙。
- checkpoint 是否按计划写出。

不要被这些非致命噪声误导：

- MFS/ObjectStorage 偶发 `GetObject`。
- kaolin/ipyevents warning。
- 初始 20 step 比较慢。

## 7. 标准评估

输出列：

```text
cond
GT/target if available
original/pretrained model, standard NFE
distilled few 1 NFE
distilled few 4 NFE
distilled few 8 NFE
optional distilled any 2/4
optional distilled mul 30
```

如果要对比 old non-distilled 的快速采样 probe，列名必须写清楚是不是官方 baseline 还是通过 TwinFlow-compatible wrapper 跑出的旧 checkpoint probe。例如：

```text
cond | GT | old_mul25 | old_few1 | old_any2 | old_any4 | distilled_few1 | distilled_any2 | distilled_any4
```

不要把 `old_few1/any2/any4` 自动写成“官方 old baseline”；它们通常只是旧权重在新 sampler 接口下的 probe。

每个样本保存 `eval_artifact_spec` 指定的项目原生产物，例如：

```text
condition_artifact_*
target_artifact_*
baseline_artifact_*
distilled_artifact_*
comparison_artifact_*
manifest.json
```

manifest 必须包含：

- sample index / ID。
- seed。
- sampler mode and NFE。
- checkpoint path。
- cond path。
- artifact paths from `eval_artifact_spec`。
- decode/render/export status。
- recommended: sample SHA256 or another deterministic identity hash。

strict eval 必须禁止 silent retry 换样本，或显式记录被替换的样本并标记该结果不能作为严格证据。

质量报告必须包含：

- cond 是否和输出语义一致。
- GT 是否正常。
- 原模型是否正常。
- distilled few 1/4/8 是否坍塌。
- few 1/4/8 是否非单调。
- 身份、表情、眼睛、牙齿、口腔、文字、铭牌、局部细节问题。
- 每个问题一个图。

## 8. 验收标准

训练验收：

- 正常结束。
- checkpoint 完整。
- EMA checkpoint 可加载。
- resume 可跑。
- branch count 正确。
- loss finite。

可视化验收：

- 对比产物包含 condition input 或等价 conditioning 证据。
- 视角/展示方式符合 `eval_artifact_spec`。
- render/export 没有明显的项目管线伪影。
- 项目原生可查看/可加载产物能被对应工具打开。
- manifest 包含所有样本和模式路径。

科研验收：

- 用同一批 cond/noise/sample 比较。
- 原模型、GT、蒸馏 few 都在同一渲染管线下。
- 推理 cfg=0。
- 报告里问题与图片一一对应。
- 若 old 和 distilled 为节省显存顺序加载，必须说明 condition encoder 是否复用同一缓存 tensor；如果没有复用，报告里把它列为 caveat。

## 9. 过去错误示例

1. 错把内部训练表示直接当可导出表示，导致 GT/target 错。
2. fallback export/render 路径引入伪影，导致误判模型质量。
3. 用不符合协议的视角/展示方式验收，导致质量判断偏移。
4. 蒸馏后推理误开 CFG，重复条件引导。
5. batch 内排序和固定 branch mask 相关，导致 branch 数据分布偏。
6. microbatch 先切再分 branch，导致和官方 batch 语义不等价。
7. `t_rescale` 未审查，可能把 `t` 推出合理范围。
8. `json.dumps` 不支持 NumPy scalar，长训中途被日志写崩。
9. 前 2k 不 OOM 不代表后续不会 OOM，稀疏形状和分支会造成峰值变化。
10. contact sheet 没有 `cond`，无法判断条件一致性。
11. 评估时数据读取失败被 retry 随机换样本，导致 full eval 看似有 6 个样本但实际有重复。
12. GT/target 保存前没有走正确反变换或导出路径，导致它不是模型要对齐的真实目标。
13. 旧 checkpoint 缺新增 TwinFlow 参数时直接 `strict=False`，会把新增权重随机初始化且不报错。
14. 复用非空 eval 输出目录，旧文件混入新 manifest。
15. 只看一张对比图就做质量结论，没有检查项目原生产物和 manifest 是否都对应同一 sample。

## 10. 新项目启动 Prompt 模板

```text
请在 <project_root> 上迁移 https://github.com/inclusionAI/TwinFlow。
开始前先运行 twinflow_porting_harness 输入合同门禁；缺 dataset/output/checkpoint/conda/slurm/eval sample/eval artifact spec 任一项就继续向用户提问，不要开始改代码。
先审查原项目与官方 TwinFlow 的模型 forward、t/tt、sampler、dataset/collate、microbatch、DDP/EMA、checkpoint、eval/render 全调用链。
实现后必须跑 synthetic smoke、real-data smoke、多卡 smoke、resume smoke。
训练前给出 config diff 和验收标准。
训练中监控 loss/branch/GPU/OOM/NaN。
训练后标准评估必须按 eval_artifact_spec 输出 condition + GT/target if available + 原模型 + 蒸馏 few/any/mul 协议列，并导出项目原生产物和 manifest。
推理 cfg 默认 0。
每个质量问题必须配对应图片。
把失败经验、关键决策、sample hash、review 结论和绝对路径写入 handoff。
```
