# 新项目 TwinFlow 迁移启动 Prompt

```text
请在 <project_root> 上迁移 https://github.com/inclusionAI/TwinFlow。

开始前先验证输入合同：
- 必须有 INPUT_CONTRACT.json。
- project_root、dataset_root、train_metadata、eval_metadata、output_root、base_checkpoint、conda_env、official_twinflow_ref、condition_type、cluster、gpu_count、eval_sample_spec 都必须明确。
- cluster=slurm 时，slurm_log_dir 以及 slurm_partition 或 sbatch_template 必须明确。
- 缺任一项时继续问我，不要开始改代码、训练或评估。

先审查原项目与官方 TwinFlow 的全调用链：
- model forward 是否支持 t/tt
- sampler 的 few/any/mul 语义
- dataset/collate 是否排序
- dense/sparse tensor 差异
- microbatch/DDP/FSDP
- EMA/checkpoint/resume
- condition/uncondition 构造
- eval/render/export 流程
- checkpoint/resume/save retention
- silent data retry/sample substitution

实现前先给出：
- 代码改动计划
- branch mix
- training config
- smoke 计划
- 验收标准

实现后必须跑：
- config parse
- bash -n
- py_compile
- synthetic 1-step
- real-data fetch smoke
- DDP 1-step
- resume 1-step
- eval/decode smoke

训练中监控：
- loss/mse/grad_norm finite
- branch count
- e2e_loss/dist_match 是否进入
- GPU utilization/memory
- OOM/NaN/NCCL/Traceback
- Slurm 队列和 log 路径
- checkpoint 数量和 retention

训练后标准评估必须输出：
cond + denormalized GT latent + 原模型标准 NFE + distilled few/any/mul 协议列。

每个样本必须导出：
NPZ、PLY、GLB、front-view PNG、contact sheet、manifest。

manifest 必须记录 sample id/hash、seed、mode、NFE、checkpoint、cond、NPZ/PLY/GLB/PNG 路径。
strict eval 不允许 silent sample substitution；如果底层数据集会 fallback 换样本，必须禁用或在报告中标记不严格。

推理 cfg 默认 0。
每个质量问题都要配对应图片或 crop。
最后写 handoff，记录关键决策、失败、修复、review 结论、命令和绝对路径。
```
