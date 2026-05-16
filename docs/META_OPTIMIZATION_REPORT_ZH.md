# Meta-Optimization Report: TwinFlow Porting Harness

日期：2026-05-16

目标：把一次 TRELLIS stage-two TwinFlow 蒸馏迁移中的经验、失败、审查结论和验收门槛固化回 `twinflow-porting-harness`，让下一次迁移更通用、更鲁棒。

## 数据来源

项目 `.aris/meta/events.jsonl` 只有 `2` 次 `skill_invoke`，低于 `$meta-optimize` 的正式统计门槛 `5`。因此本次不是全局统计优化，而是一次明确标注来源的 case-study optimization。

使用的证据：

- `MEMORY.md` 中的训练、Slurm、checkpoint、eval 记录。
- `.aris/traces/twinflow_code_review/*.md` clean-context call-chain review。
- `.aris/traces/experiment_audit/*.md` experiment audit。
- step-2000 协议评估中暴露的 strict eval / silent retry / GT denorm / output dir / sample hash 问题。
- 用户多轮干预点：权重 key、坏数据、训练前是否 clean review、是否有生成图片、按协议看结果。

## 主要经验

| 信号 | 教训 | Harness 改动 |
| --- | --- | --- |
| 用户需要显式指定数据 CSV、输出目录、conda env、Slurm 方式 | 这些不能靠 agent 猜 | 新增 `python -m twinflow_porting_harness init/validate/required` 输入合同门禁 |
| 训练前用户追问是否做了 clean-context review | TwinFlow 与原项目接口边界最容易 silent drift | 文档和 checklist 强制 call-chain review |
| old checkpoint 缺新增 TwinFlow 参数 | `strict=False` 会隐藏随机新权重 | checklist 要求只允许 audited new/missing keys，并写明 warm-start 策略 |
| GT/target 初始 decode 风险 | 项目内部训练表示不一定等同可导出表示 | eval 文档要求 GT/target 与模型输出走同一 decode/render/export 路径 |
| full eval 中 sample 3 被数据 retry 替换 | dataset fallback 会造成“看似完整”的假样本对齐 | strict eval 要求禁用 silent substitution 或记录 sample hash |
| 非空输出目录可能混入旧文件 | artifact 不能只靠文件名肉眼判断 | eval 默认新目录，manifest 记录 paths/hash |
| old `few/any` 不是官方 baseline | 通过 TwinFlow wrapper 的旧权重 probe 容易被误称 baseline | 文档要求列名和报告区分 old standard vs old probe |
| 登录机推理可行但训练要 Slurm | 调度策略需要显式记录 | 输入合同加入 cluster、gpu_count、Slurm log/partition/template |

## 验收标准更新

新增或强化的门禁：

- 输入合同必须 validate。
- 数据路径、输出路径、checkpoint、conda env、eval sample、Slurm log 不能缺省。
- agent 缺参数时必须问用户，不能开始改代码。
- eval 输出目录默认 fresh。
- manifest 记录 sample id/hash、seed、mode、checkpoint、`eval_artifact_spec` 指定产物路径。
- strict eval 禁止 silent sample substitution。
- GT/target decode/export 必须明确且和模型输出同路径。
- 阶段成果必须 clean-context review。

## 不做的事

- 不把 TRELLIS 的具体超参变成所有项目默认。
- 不假设所有项目都有 sparse latent、VAE、GLB 导出。
- 不强制所有 Slurm 集群必须有 account；只要求 partition 或 sbatch template 至少一个可用启动信息。
- 不把 visual smoke 写成模型质量或论文结论。
