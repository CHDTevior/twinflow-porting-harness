# 新项目 TwinFlow 迁移启动 Prompt

```text
请在 <project_root> 上迁移 https://github.com/inclusionAI/TwinFlow。

先审查原项目与官方 TwinFlow 的全调用链：
- model forward 是否支持 t/tt
- sampler 的 few/any/mul 语义
- dataset/collate 是否排序
- dense/sparse tensor 差异
- microbatch/DDP/FSDP
- EMA/checkpoint/resume
- condition/uncondition 构造
- eval/render/export 流程

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

训练后标准评估必须输出：
cond + GT latent + 原模型标准 NFE + distilled few 1/4/8。

每个样本必须导出：
NPZ、PLY、GLB、front-view PNG、contact sheet、manifest。

推理 cfg 默认 0。
每个质量问题都要配对应图片或 crop。
最后写 handoff，记录关键决策、失败、修复、命令和绝对路径。
```
