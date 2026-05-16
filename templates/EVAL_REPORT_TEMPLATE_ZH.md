# TwinFlow 蒸馏评估报告模板

日期：

run：

checkpoint：

## 1. 评估设置

| 项 | 值 |
| --- | --- |
| sample indices | |
| sample hashes | |
| seed | |
| original model NFE | |
| old probe NFE/modes | |
| distilled few/any/mul NFE | |
| cfg | `0` |
| render view | front-view |
| strict data retry | disabled / hash-checked / not strict |

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

decode/render status：

`<decode_summary>`

## 3. 标准拼图

必须包含：

```text
cond | denormalized GT latent | original/pretrained | old probes if any | distilled few/any/mul
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

### 4.8 数据/样本严格性

图或 manifest 证据：

结论：

### 4.9 GT denorm/decode 正确性

图或 manifest 证据：

结论：

## 5. 结论

- [ ] 管线正确
- [ ] few-step 可用
- [ ] 需要继续训练
- [ ] 需要改超参
- [ ] 需要排查数据/渲染
- [ ] 当前只是 visual smoke，不支持质量/论文 claim

下一步：

1.
2.
3.
