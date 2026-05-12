# TwinFlow 蒸馏评估报告模板

日期：

run：

checkpoint：

## 1. 评估设置

| 项 | 值 |
| --- | --- |
| sample indices | |
| seed | |
| original model NFE | |
| distilled few NFE | `1,4,8` |
| any/mul | |
| cfg | `0` |
| render view | front-view |

## 2. 产物路径

NPZ：

`<npz_dir>`

PLY：

`<ply_dir>`

GLB：

`<glb_dir>`

PNG：

`<png_dir>`

contact sheet：

`<contact_sheet>`

manifest：

`<manifest>`

## 3. 标准拼图

必须包含：

```text
cond | GT latent | original/pretrained | distilled few 1 | distilled few 4 | distilled few 8
```

插图：

`<image>`

## 4. 逐问题分析

### 4.1 cond 一致性

图：

结论：

### 4.2 主体结构是否坍塌

图：

结论：

### 4.3 身份/语义漂移

图：

结论：

### 4.4 眼睛/牙齿/口腔

图：

结论：

### 4.5 文字/铭牌/细小结构

图：

结论：

### 4.6 细节过锐或过平滑

图：

结论：

### 4.7 NFE 非单调

图：

结论：

## 5. 结论

- [ ] 管线正确
- [ ] few-step 可用
- [ ] 需要继续训练
- [ ] 需要改超参
- [ ] 需要排查数据/渲染

下一步：

1.
2.
3.
