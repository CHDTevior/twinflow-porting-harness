# TwinFlow Porting Harness

一套把 `https://github.com/inclusionAI/TwinFlow` 迁移到已有预训练生成项目的工程化 checklist、smoke 流程和验收模板。

这不是官方 TwinFlow 代码的 fork；它是迁移时用的 harness。目标是降低二次迁移时最常见的错误：

- `t/tt` 接入不完整。
- dense Tensor 代码误搬到 SparseTensor latent。
- branch partition 和 microbatch 不等价。
- EMA、checkpoint、resume 不完整。
- 蒸馏后推理误开 CFG。
- eval 没有展示 `cond`。
- GT latent 或渲染管线错误导致误判模型质量。

## 核心文档

- [TwinFlow 迁移易错 Bug 清单](docs/BUG_CHECKLIST_ZH.md)
- [TwinFlow 迁移与蒸馏验收 Harness](docs/MIGRATION_HARNESS_ZH.md)
- [新项目启动 Prompt](templates/START_PROMPT_ZH.md)
- [实验交接模板](templates/HANDOFF_TEMPLATE_ZH.md)
- [验收报告模板](templates/EVAL_REPORT_TEMPLATE_ZH.md)

## 最小流程

```text
审查原项目
  -> 对照官方 TwinFlow
  -> 实现 t/tt + sampler + trainer
  -> synthetic smoke
  -> real-data smoke
  -> DDP smoke
  -> resume smoke
  -> eval/decode smoke
  -> 长训
  -> cond + GT + 原模型 + distilled few 1/4/8 验收
```

## 强制验收规则

1. 蒸馏后推理默认 `cfg=0`。
2. 标准 contact sheet 第一列必须是 `cond` 条件图。
3. 每个样本至少输出 NPZ、PLY、GLB、front-view PNG、manifest。
4. 每个质量问题必须配对应图片或 crop。
5. 长训前必须完成 smoke，不允许直接上 Slurm 长训。

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
```

## 适配经验来源

这套 harness 来自一次 TRELLIS stage-two 512 PRO full-parameter TwinFlow 迁移，包括：

- sparse latent / dense condition 混用。
- DDP + fp16 master + EMA teacher。
- `mul/any` baseline。
- `e2e/mul/any/adv + dist_match` distribution distillation。
- 3D 输出 NPZ/PLY/GLB/PNG/contact sheet 验收。

不要把这里的超参当成所有项目的默认最优；它们是一个可靠起点。迁移时先保持变量少，再逐项做 ablation。
