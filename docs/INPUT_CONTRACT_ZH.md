# TwinFlow 启动输入合同与交互门禁

日期：2026-05-16

目的：让 agent 在迁移 TwinFlow 前先拿到完整、可验证、绝对路径化的输入合同。缺少这些参数时，不允许开始改代码、训练或评估。

## 1. 为什么需要这个门禁

真实迁移中最容易造成返工的不是模型公式本身，而是边界信息不完整：

- 数据 CSV 或 dataset root 没明确，agent 默认拿错 split。
- output directory 没明确，旧结果和新结果混在一起。
- checkpoint 路径没明确，old/pretrained/distilled 权重对不上。
- Slurm log 路径没明确，监控和错误归因断链。
- eval sample 没固定，retry 后静默换样本，contact sheet 看起来完整但实际重复。
- conda env 没私有化，agent 和用户互相污染环境。

因此 harness 的第一个动作是生成并验证 `INPUT_CONTRACT.json`。

## 2. CLI

查看必填字段：

```bash
python -m twinflow_porting_harness required
```

生成启动包：

```bash
python -m twinflow_porting_harness init \
  --project-root /abs/project \
  --dataset-root /abs/project/data \
  --train-metadata /abs/project/data/train.csv \
  --eval-metadata /abs/project/data/test.csv \
  --output-root /abs/project/outputs \
  --run-name twinflow_distill_v1 \
  --base-checkpoint /abs/ckpts/base.pt \
  --conda-env /abs/conda/envs/private_env \
  --official-twinflow-ref https://github.com/inclusionAI/TwinFlow \
  --original-inference-entry "python example.py ..." \
  --condition-type image \
  --cluster slurm \
  --gpu-count 8 \
  --eval-sample-spec "fixed indices 0 1 2 4 5, seed 42" \
  --slurm-log-dir /abs/project/slurm_logs \
  --slurm-partition gpu
```

验证：

```bash
python -m twinflow_porting_harness validate /abs/project/handoff/twinflow_<run_name>/INPUT_CONTRACT.json
```

## 3. 非交互模式的行为

如果缺字段，CLI 退出码为 `2`，并打印：

```text
ERROR: TwinFlow porting input contract is incomplete.
Do not start code changes, training, or evaluation until these are resolved.

Ask the user for these required values:
- dataset_root: 数据集根目录的绝对路径
- output_root: 实验输出根目录的绝对路径
...
```

这段输出就是给 agent 的硬指令：必须继续问用户，不能自己猜路径。

路径必须以 `/` 开头并按原样写入合同；不要传 `~/...` 或相对路径。本地 TwinFlow ref 必须指向带 `.git` 元数据的 clone/worktree 目录。URL 必须包含完整 scheme、host、repo path，例如 `https://github.com/inclusionAI/TwinFlow`。

## 4. 交互模式

带 `--ask` 时，CLI 会对缺失字段反复提问，直到合同完整：

```bash
python -m twinflow_porting_harness init --ask
```

适合人手动初始化新项目。对于自动 agent，推荐非交互模式：缺什么就让 agent 把缺失字段原样问给用户。

## 5. 字段语义

| 字段 | 要求 | 作用 |
| --- | --- | --- |
| `project_root` | existing absolute directory | 目标项目，agent 只应在这里写代码 |
| `dataset_root` | existing absolute directory | 数据集存放根路径 |
| `train_metadata` | existing absolute path | 训练 split CSV/manifest |
| `eval_metadata` | existing absolute path | eval split CSV/manifest |
| `output_root` | existing absolute directory, or create with `--create-output-root` | checkpoints、samples、eval 输出根目录 |
| `run_name` | slug | 本次迁移 run 名称 |
| `base_checkpoint` | existing absolute file | 原始预训练权重 |
| `conda_env` | existing absolute directory | 私有 conda 环境 |
| `official_twinflow_ref` | full URL or existing absolute git clone/worktree directory | 官方 TwinFlow 对照来源 |
| `original_inference_entry` | text | 原项目 pretrained 推理入口或命令 |
| `condition_type` | image/text/multiview/control/other | 条件类型 |
| `cluster` | local/slurm/ddp/fsdp/other | 运行环境 |
| `gpu_count` | positive integer | 计划 GPU 数量 |
| `eval_sample_spec` | text | 固定 eval 样本、seed、采样规则 |
| `slurm_log_dir` | existing absolute directory, or create with `--create-output-root` | Slurm 日志目录，`cluster=slurm` 时必填 |
| `slurm_partition` | text | Slurm partition；和 `sbatch_template` 二选一 |
| `sbatch_template` | existing absolute file | sbatch 模板；和 `slurm_partition` 二选一 |

## 6. 输出文件

默认输出到：

```text
<project_root>/handoff/twinflow_<run_name>/
```

包含：

- `INPUT_CONTRACT.json`：机器可读合同。
- `START_PROMPT.md`：给 agent 的启动 prompt。
- `ACCEPTANCE_CHECKLIST.md`：从代码审查到正式 eval 的门禁。
- `MANIFEST.md`：启动包索引。

## 7. Agent 必须遵守的行为

- 没有完整合同，不开始改代码。
- 合同字段无效，不开始训练。
- output dir 非空时，不混用旧结果；必须新建目录或显式记录 override。
- eval 禁止 silent sample substitution；如果底层 dataset 会 retry 换样本，必须记录 sample hash 或在 strict eval 中禁用替换。
- 所有阶段性结论写入 handoff/MEMORY，并附绝对路径。
