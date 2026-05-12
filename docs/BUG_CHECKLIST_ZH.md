# TwinFlow 迁移易错 Bug 清单

日期：2026-05-13

用途：以后在新机器或新仓库重新迁移 `https://github.com/inclusionAI/TwinFlow` 时逐项检查。这里列的是我们这次 TRELLIS 适配里实际踩过或高风险的坑。

## 0. 最容易漏掉的 12 个硬规则

- [ ] 蒸馏是在预训练 checkpoint 上 full-parameter 继续训练，不是 LoRA，除非明确切 LoRA 路线。
- [ ] 模型 forward 必须同时支持 `t` 和 `tt`。
- [ ] 新增 `tt_embedder` 后，warm-start 只允许缺 `tt_embedder.*`，并从 `t_embedder` 初始化。
- [ ] TwinFlow 推理默认 `cfg=0`，不要二次 CFG。
- [ ] local batch official full 至少为 `4`，否则 `e2e/mul/any/adv` 无法各占一个样本。
- [ ] branch mask 必须在 full local batch 上构造，再切 microbatch。
- [ ] 如果 dataset/collate 会排序，branch assignment 要随机化。
- [ ] sparse latent 不能直接套官方 dense tensor 写法。
- [ ] eval/contact sheet 第一列必须是 `cond`。
- [ ] GT latent 保存前必须反归一化到 VAE latent 空间。
- [ ] 渲染验收必须 front-view，不能有黑色三角描边伪影。
- [ ] 长训前必须通过 synthetic、real-data、DDP、resume、eval/decode smoke。

## 1. 模型结构类

### 1.1 忘记加 `tt`

症状：

- 训练能跑但 `few/any/mul` 没有语义差别。
- sampler 传了 `tt`，模型实际忽略。

检查：

- [ ] model forward signature 有 `tt`。
- [ ] `tt` 经过 timestep embedding。
- [ ] `t` 和 `tt` 都传入 block/modulation。
- [ ] 新旧 checkpoint 加载时 `tt_embedder` 初始化正确。

### 1.2 strict load 把旧 checkpoint 拒掉

症状：

- 预训练 checkpoint 加载失败，missing `tt_embedder.*`。

修复：

- [ ] trainer warm-start 允许缺 `tt_embedder.*`。
- [ ] 其他 missing/unexpected key 仍然报错。
- [ ] `tt_embedder` 从 `t_embedder` copy，不随机初始化。

### 1.3 EMA teacher 不是每个 rank 一致

症状：

- DDP 各 rank loss 漂移。
- resume 后结果突变。

检查：

- [ ] EMA teacher 从同一个 student 初始化。
- [ ] EMA update 每个 rank 在相同 step 执行。
- [ ] EMA checkpoint 独立保存。
- [ ] resume 同时恢复 student、optimizer、EMA、TwinFlow state。

## 2. dense / sparse tensor 类

### 2.1 把 official dense 写法直接搬到 sparse latent

错误示例：

```python
noise = torch.randn_like(x)
```

正确写法：

```python
noise = x.replace(torch.randn_like(x.feats))
```

检查：

- [ ] `x_0/x_t/target/velocity` 是 SparseTensor。
- [ ] `cond/t/tt/mask` 是 dense tensor。
- [ ] 所有 sparse 运算保持 `coords/layout`。
- [ ] 不直接对 SparseTensor 用 dense in-place 赋值。

### 2.2 loss 对全部 feats 直接 mean

错误：

```python
loss = mse(pred.feats, target.feats)
```

问题：voxel 多的样本权重更大。

正确：

- [ ] 按 `layout[i]` 每个样本分别算 loss。
- [ ] 再对 batch 维平均。

### 2.3 dense `t` 广播到 sparse feats 维度错

症状：

- shape mismatch。
- silent broadcast 到错误维度。

检查：

- [ ] `t/tt` 是 `[B]`。
- [ ] 进入 sparse 运算时用 `unsqueeze(-1)` 或 sparse batch broadcast。
- [ ] 每个 sparse sample 的 feats slice 使用对应 batch 的 scalar。

### 2.4 adv 分支 sparse 替换错

错误风险：

```python
x[m] = x_fake
```

正确：

- [ ] `x[m]` 会重建 sparse coords/layout。
- [ ] 使用 `sparse_unbind -> 替换对应 sample -> sparse_cat`。
- [ ] 替换后 batch order、coords batch id、layout 都正确。

## 3. t / tt / scheduler 类

### 3.1 `t_rescale` 未审查

症状：

- `t` 可能出 `[0,1]`。
- sparse length 小时甚至出现负 rescale。

规则：

- [ ] TwinFlow 默认 `t_rescale=false`。
- [ ] 若开启，必须重新推导公式并 assert/clamp。

### 3.2 `sigma_min` 不等于 0

风险：

- official TwinFlow/UCGM 公式和原项目 flow schedule 不一致。

检查：

- [ ] TwinFlow trainer 对不兼容 `sigma_min` 直接报错。
- [ ] config 中 `sigma_min=0.0`。

### 3.3 adv 负时间处理错

检查：

- [ ] `adv` 分支先用 fake sample 替换输入。
- [ ] `tt=-t`。
- [ ] diffuse 构造 `x_t` 时使用 `abs(t)`。
- [ ] model forward/loss 使用带符号的 `t`。

### 3.4 `few/any/mul` sampler 语义混了

规则：

```text
few: tt = 0
any: tt = t_next
mul: tt = t_cur or local flow style
```

检查：

- [ ] eval 图列名准确。
- [ ] few 1/4/8 不是 any/mul。
- [ ] snapshot 里 few/any/mul 分开标。

## 4. branch partition / microbatch 类

### 4.1 先 microbatch 再分 branch

问题：

- local batch=4, batch_split=2 时，每个 microbatch 无法保留 official 1:1:1:1。

正确：

- [ ] full local batch 上先生成 masks。
- [ ] masks 随 microbatch 一起切。
- [ ] loss 按 microbatch 数缩放。

### 4.2 collate 排序导致 branch 数据偏

问题：

- sparse load balancing 可能按 voxel count 排序。
- 固定前 1 个样本给 e2e，会让 branch 和 sparse size 相关。

正确：

- [ ] 保留 official branch counts。
- [ ] sample-to-branch assignment 用随机 permutation。
- [ ] 日志记录 branch count。

### 4.3 batch size 小于 4

规则：

- [ ] full official TwinFlow 必须 batch size >= 4。
- [ ] batch=1 只能用于 smoke 或 multinomial，不代表官方等价。

## 5. target / loss 类

### 5.1 enhanced target 理解错

实际增强：

```text
target += enhanced_ratio * (EMA_cond_velocity - EMA_uncond_velocity)
```

不是增强图片，不是增强 cond embedding，不是推理 CFG。

检查：

- [ ] enhanced 只改 velocity target。
- [ ] `adv` 分支不增强。
- [ ] 当前 `[0,1]` 且 `<1` 会让 e2e 的 `t=1` 通常不增强。

### 5.2 RCGM target 和 enhanced 顺序错

当前顺序：

```text
base target -> enhanced target -> RCGM target -> loss
```

检查：

- [ ] 如果改顺序，必须重新说明实验含义。
- [ ] RCGM 用的是已 enhanced 的 target。

### 5.3 dist-match 反传路径错

规则：

- [ ] dist-match 的 fake/real target 部分不应反传。
- [ ] 只在 e2e 分支加 dist-match loss/grad。
- [ ] `dist_match_cof > 0` 时日志必须出现 `e2e_loss`。

### 5.4 Barron reweighting 没 detach weight

风险：

- loss 权重本身参与梯度，和原算法不一致。

检查：

- [ ] reweighting weight detach。
- [ ] time weighting 与 adv 分支规则一致。

## 6. CFG / condition 类

### 6.1 推理时二次 CFG

错误：

- 训练 target 已经蒸进条件引导，推理又 `cfg>0`。

规则：

- [ ] distilled eval `cfg_strength=0.0`。
- [ ] 报告里明确写 `cfg=0`。

### 6.2 `neg_cond` 构造不一致

检查：

- [ ] image-conditioned 路径中 `neg_cond=zeros_like(cond)` 是否符合原模型习惯。
- [ ] text-conditioned 路径不要误用 image zero cond。
- [ ] cond encoder 和预训练完全一致。

### 6.3 eval 没保存 cond

规则：

- [ ] contact sheet 第一列是 `cond`。
- [ ] 每个样本保存 `sample_cond_image_*.png` 或等价条件可视化。

## 7. 数据 / latent 类

### 7.1 normalized latent 当 VAE latent 解码

症状：

- GT 图异常。
- 模型看起来全错，但其实是导出错。

检查：

- [ ] 训练用 normalized latent。
- [ ] eval/decode 保存前反归一化。
- [ ] `mean/std` 和 base checkpoint 匹配。

### 7.2 latent 文件格式没确认

风险：

- 有些文件是 `feats`，有些可能是 `mean/logvar`。

检查：

- [ ] 统计真实数据 latent key。
- [ ] 若采样 `mean/logvar`，记录随机性和 seed。

### 7.3 数据失败被 silent retry 掩盖

风险：

- MFS 失败后随机换样本，长训能跑但数据分布变。

检查：

- [ ] real-data preflight。
- [ ] 记录失败 UUID/path。
- [ ] 监控 data elapsed。

## 8. 评估 / 渲染 / 导出类

### 8.1 斜侧面图当验收图

规则：

- [ ] 标准图必须 front-view。
- [ ] yaw/elev 固定。

### 8.2 黑点伪影

原因：

- CPU/PIL fallback 给三角面画黑色 outline。

规则：

- [ ] 验收 PNG 不画三角边。
- [ ] 可疑时看 PLY/GLB，不只看 PNG。

### 8.3 只保存 NPZ，不导出 3D 可查看格式

规则：

- [ ] 保存 PLY。
- [ ] 保存 GLB。
- [ ] 保存 PNG preview。
- [ ] manifest 记录每个文件路径。

### 8.4 eval 没固定同一 noise/sample

检查：

- [ ] GT、原模型、蒸馏模型使用同一 sample。
- [ ] 原模型和蒸馏模型使用同一 init noise。
- [ ] seed 写入 manifest。

## 9. checkpoint / resume / logging 类

### 9.1 EMA checkpoint 没保存

检查：

- [ ] `denoiser_step*.pt`
- [ ] `denoiser_twinflow_ema_step*.pt`
- [ ] `misc_step*.pt`
- [ ] `twinflow_state_step*.pt`

### 9.2 resume 后日志重复 step

规则：

- [ ] 分析时按 step 去重，取最后一次。
- [ ] 不用 raw line count 当训练 step 数。

### 9.3 JSON 写日志崩溃

风险：

- NumPy scalar / Torch tensor 不能直接 `json.dumps`。

检查：

- [ ] logger 支持 NumPy scalar/array。
- [ ] logger 支持 Torch scalar/tensor。
- [ ] TensorBoard/W&B 只写 finite scalar。

## 10. Slurm / 环境类

### 10.1 `sbatch --wrap` 用了 `/bin/sh`

规则：

- [ ] 需要 `source` 或 bash array 时，用 `bash -lc`。

### 10.2 NCCL 网卡/IB 配置错误

检查：

- [ ] `NCCL_SOCKET_IFNAME` 存在。
- [ ] `GLOO_SOCKET_IFNAME` 同步。
- [ ] master addr/port 正确。
- [ ] local/slurm 两套启动路径都测。

### 10.3 依赖路径不固定

检查：

- [ ] conda env 绝对路径。
- [ ] DINO/CLIP/其他 encoder checkpoint 绝对路径。
- [ ] TORCH_HOME 等 cache 路径固定。

## 11. OOM / 性能类

### 11.1 前 2k 没 OOM 就以为安全

原因：

- sparse voxel 数随 batch 变。
- e2e/adv/dist-match/RCGM/EMA 都会改变峰值。

处理：

- [ ] 先只增大 `batch_split`。
- [ ] 不同时改 batch/lr/branch。
- [ ] 记录 OOM 发生 step 和输入 size。

### 11.2 teacher copy 时 OOM

检查：

- [ ] EMA teacher 只在需要时创建。
- [ ] 创建位置在 DDP/FSDP 语义下可控。
- [ ] smoke 覆盖 teacher 创建、保存、resume。

## 12. Git / 项目交接类

### 12.1 把 pycache、outputs、slurm log 一起提交

规则：

- [ ] `.gitignore` 排除 `__pycache__`, `outputs`, `slurm_logs`, checkpoints。
- [ ] GitHub 项目只放 harness、模板、说明，别放大模型和数据。

### 12.2 没记录关键决策

每次必须记录：

- [ ] 为什么这个 branch mix。
- [ ] 为什么这个 enhanced_ratio。
- [ ] 为什么 cfg=0。
- [ ] 是否 full-param 或 LoRA。
- [ ] 是否开启 dist-match/image-free/t_rescale。
- [ ] 失败和修复。

## 13. 最小验收命令模板

```bash
python -m json.tool <config.json>
bash -n <train.sh> <eval.sh> <decode.sh>
python -m py_compile <changed_python_files>
```

训练日志检查：

```bash
tail -n 40 <output_dir>/log.txt
rg -n "RuntimeError|Traceback|OOM|NaN|nan|Saving checkpoint|Training finished" <slurm_log>
```

标准评估必须输出：

```text
cond | GT latent | original/pretrained | distilled few 1 | distilled few 4 | distilled few 8
```

如果没有 `cond` 第一列，验收不通过。
