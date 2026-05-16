# TwinFlow 迁移交接模板

日期：

项目：

`<absolute_project_root>`

官方 TwinFlow：

`<official_repo_or_local_clone>`

输入合同：

`<INPUT_CONTRACT.json>`

## 1. 当前状态

- [ ] 未开始
- [ ] 审查中
- [ ] 已实现
- [ ] smoke 通过
- [ ] 长训中
- [ ] 长训完成
- [ ] eval 完成

一句话结论：

## 2. 关键路径

配置：

`<config_path>`

训练脚本：

`<train_script>`

输出目录：

`<output_dir>`

Slurm log：

`<slurm_log>`

Slurm queue/partition：

`<partition_or_template>`

checkpoint：

`<ckpt_dir>`

标准 eval：

`<eval_dir>`

## 3. 关键超参

| 项 | 值 |
| --- | --- |
| base checkpoint | |
| trainer | |
| max steps | |
| GPU | |
| per-GPU batch | |
| global batch | |
| microbatch / batch_split | |
| optimizer | |
| lr | |
| fp16 | |
| branch mix | |
| enhanced_ratio | |
| estimate_order | |
| dist_match | |
| EMA | |
| t_rescale | |
| inference cfg | `0` |
| dataset root | |
| train metadata | |
| eval metadata | |
| eval sample spec | |
| eval artifact spec | |

## 4. 已完成

- [ ] model forward 支持 TwinFlow two-time interface
- [ ] 新增 TwinFlow 参数 warm-start 策略已审查
- [ ] TwinFlow trainer
- [ ] UnifiedSampler
- [ ] EMA save/resume
- [ ] eval scripts
- [ ] decode/export scripts
- [ ] eval_artifact_spec 指定的对比产物已生成
- [ ] strict eval sample hash / identity 记录
- [ ] clean-context review 完成

## 5. smoke 结果

- [ ] config parse
- [ ] bash -n
- [ ] py_compile
- [ ] synthetic 1-step
- [ ] real-data fetch
- [ ] DDP 1-step
- [ ] resume 1-step
- [ ] eval/decode
- [ ] strict eval 不发生 silent sample substitution

## 6. 当前训练状态

job id：

node：

latest step：

latest loss：

latest mse：

latest grad norm：

branch count：

OOM/NaN/NCCL：

## 7. 验收结果

comparison artifact：

`<comparison_artifact>`

布局/列顺序：

```text
condition input | GT/target if available | original/pretrained | old probes if any | distilled few/any/mul
```

project-native exports：

`<artifact_paths>`

主要问题：

- [ ] 身份漂移
- [ ] 局部细节过锐/过平滑
- [ ] 眼睛/牙齿/口腔不稳定
- [ ] 文字/铭牌失败
- [ ] NFE 非单调
- [ ] cond 不一致
- [ ] GT/target decode/export 异常
- [ ] data retry/sample substitution

manifest：

`<manifest.json>`

sample hashes：

```text
<sample_id> <hash>
```

## 8. 关键决策

1.
2.
3.

## 9. 失败与修复

| 问题 | 现象 | 根因 | 修复 | 是否验证 |
| --- | --- | --- | --- | --- |
| | | | | |

## 10. Review / Audit

| 阶段 | reviewer | trace path | verdict | action |
| --- | --- | --- | --- | --- |
| code call-chain | | | | |
| smoke gate | | | | |
| checkpoint gate | | | | |
| protocol eval | | | | |

## 11. 下一个 session 要做什么

1.
2.
3.
