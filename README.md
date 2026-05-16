# TwinFlow Porting Harness

一套把 `https://github.com/inclusionAI/TwinFlow` 迁移到已有预训练生成项目的工程化 checklist、smoke 流程和验收模板。

这不是官方 TwinFlow 代码的 fork；它是迁移时用的 harness。目标是降低二次迁移时最常见的错误：

- `t/tt` 接入不完整。
- dense Tensor 代码误搬到 SparseTensor latent。
- branch partition 和 microbatch 不等价。
- EMA、checkpoint、resume 不完整。
- 蒸馏后推理误开 CFG。
- eval 没有展示 `cond`。
- GT/target 或导出管线错误导致误判模型质量。
- 数据路径、输出路径、checkpoint、Slurm log 等关键输入没有被显式指定，agent 直接开始改代码。
- 数据读取失败被 dataset retry 静默替换，导致评估样本看似完整但实际重复。

## 核心文档

- [TwinFlow 迁移易错 Bug 清单](docs/BUG_CHECKLIST_ZH.md)
- [TwinFlow 迁移与蒸馏验收 Harness](docs/MIGRATION_HARNESS_ZH.md)
- [启动输入合同与交互门禁](docs/INPUT_CONTRACT_ZH.md)
- [新项目启动 Prompt](templates/START_PROMPT_ZH.md)
- [实验交接模板](templates/HANDOFF_TEMPLATE_ZH.md)
- [验收报告模板](templates/EVAL_REPORT_TEMPLATE_ZH.md)

## 先做输入合同，不然不要开始

这个 harness 现在带一个纯标准库 CLI，用来强制收集关键路径和运行参数。缺少数据路径、输出路径、checkpoint、conda env、评估样本、Slurm log 等必填项时，CLI 会拒绝生成启动 prompt；在交互模式下会一直追问，非交互模式下会打印 agent 必须向用户追问的字段。

查看必填项：

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
  --eval-artifact-spec "condition input + GT/target if available + baseline/distilled comparison + manifest" \
  --slurm-log-dir /abs/project/slurm_logs \
  --slurm-partition gpu
```

`output_root` 和 `slurm_log_dir` 默认必须已经存在；如果希望 CLI 创建它们，加入 `--create-output-root`。`--dry-run` 永远不会创建目录。

所有本地路径必须是以 `/` 开头的绝对路径，不接受 `~/...` 或相对路径；本地 TwinFlow ref 必须指向带 `.git` 元数据的 clone/worktree 目录；远程 TwinFlow ref 必须使用包含 scheme、host、repo path 的完整 URL，例如 `https://github.com/inclusionAI/TwinFlow`。

输出：

```text
<project_root>/handoff/twinflow_<run_name>/
  INPUT_CONTRACT.json
  START_PROMPT.md
  ACCEPTANCE_CHECKLIST.md
  MANIFEST.md
```

验证：

```bash
python -m twinflow_porting_harness validate /abs/project/handoff/twinflow_<run_name>/INPUT_CONTRACT.json
```

本仓库自测：

```bash
python -m unittest discover -s tests -v
```

## 最小流程

```text
输入合同门禁
  -> 审查原项目
  -> 对照官方 TwinFlow
  -> 实现 t/tt + sampler + trainer
  -> synthetic smoke
  -> real-data smoke
  -> DDP smoke
  -> resume smoke
  -> eval/decode smoke
  -> 长训
  -> 按 eval_artifact_spec 做协议验收
```

## 强制验收规则

1. 没有完整 `INPUT_CONTRACT.json`，不得开始改代码、训练或评估。
2. 蒸馏后推理默认 `cfg=0`。
3. 标准对比产物必须展示条件输入或等价的 conditioning 证据。
4. GT/target 如果存在，必须和模型输出走同一 decode/render/export 路径。
5. 每个样本必须输出 `eval_artifact_spec` 指定的项目原生产物和 manifest。
6. manifest 必须记录 sample id/seed/mode/path，推荐记录样本 hash，避免 silent retry 混入。
7. 每个质量问题必须配对应图片或 crop。
8. 长训前必须完成 smoke，不允许直接上 Slurm 长训。
9. 阶段成果和正式结果必须经过 clean-context review；数据证据可以推翻 review，但不能跳过 review。

## 建议目录结构

```text
project/
  docs/
    BUG_CHECKLIST_ZH.md
    MIGRATION_HARNESS_ZH.md
  templates/
    START_PROMPT_ZH.md
    HANDOFF_TEMPLATE_ZH.md
    EVAL_REPORT_TEMPLATE_ZH.md
  twinflow_porting_harness/
    cli.py
```

## 适配经验来源

这套 harness 来自一次 TRELLIS stage-two TwinFlow 迁移，但代码门禁不绑定 TRELLIS；其中可迁移的经验包括：

- 项目内部 tensor/container 表示和条件输入表示不一致。
- DDP + fp16 master + EMA teacher。
- `mul/any` baseline。
- `e2e/mul/any/adv + dist_match` distribution distillation。
- 项目原生产物、对比图/contact sheet、manifest 验收。
- step-2000 早期 checkpoint 协议评估暴露的 silent retry、GT/target 处理、旧 checkpoint 新增参数 warm-start、fresh output dir、sample hash 记录等问题。

不要把这里的超参当成所有项目的默认最优；它们是一个可靠起点。迁移时先保持变量少，再逐项做 ablation。
