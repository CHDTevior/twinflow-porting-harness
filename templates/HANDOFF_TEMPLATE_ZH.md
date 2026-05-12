# TwinFlow 迁移交接模板

日期：

项目：

`<absolute_project_root>`

官方 TwinFlow：

`<official_repo_or_local_clone>`

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

## 4. 已完成

- [ ] model forward 支持 `tt`
- [ ] `tt_embedder` warm-start
- [ ] TwinFlow trainer
- [ ] UnifiedSampler
- [ ] EMA save/resume
- [ ] eval scripts
- [ ] decode/export scripts
- [ ] cond 第一列可视化

## 5. smoke 结果

- [ ] config parse
- [ ] bash -n
- [ ] py_compile
- [ ] synthetic 1-step
- [ ] real-data fetch
- [ ] DDP 1-step
- [ ] resume 1-step
- [ ] eval/decode

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

contact sheet：

`<comparison_all.png>`

列顺序：

```text
cond | GT latent | original | distilled few 1 | distilled few 4 | distilled few 8
```

3D exports：

`<ply_dir>`

`<glb_dir>`

主要问题：

- [ ] 身份漂移
- [ ] 局部细节过锐/过平滑
- [ ] 眼睛/牙齿/口腔不稳定
- [ ] 文字/铭牌失败
- [ ] NFE 非单调
- [ ] cond 不一致

## 8. 关键决策

1.
2.
3.

## 9. 失败与修复

| 问题 | 现象 | 根因 | 修复 | 是否验证 |
| --- | --- | --- | --- | --- |
| | | | | |

## 10. 下一个 session 要做什么

1.
2.
3.
