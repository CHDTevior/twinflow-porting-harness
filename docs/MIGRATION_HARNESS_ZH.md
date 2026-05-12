# TwinFlow 迁移与蒸馏验收 Harness

日期：2026-05-08

目标：把 `https://github.com/inclusionAI/TwinFlow` 迁移到任意已有预训练生成项目，并形成可重复、可审查、可验收的训练与评估流程。

## 0. 强制规则

1. 先审查原项目和官方 TwinFlow，再写代码。
2. 所有路径使用绝对路径记录。
3. 所有长训前必须通过 smoke。
4. 蒸馏后推理默认 `cfg=0`，除非明确证明需要开启。
5. 每次可视化输出第一列必须是 `cond` 条件图片。
6. 每个质量问题都要配对应截图或 crop，不能只写文字。
7. 训练日志分析遇到 resume 重复 step 时，按 step 去重取最后一次。

## 1. 输入信息模板

新项目开始时先填：

```text
project_root =
official_twinflow_root =
official_twinflow_commit =
base_checkpoint =
train_data =
test_data =
condition_type = image/text/multiview/control/other
original_inference = sampler + NFE + cfg
distilled_eval = few 1/4/8 + optional any/mul
cluster = Slurm/DDP/FSDP/local
conda_env =
```

## 2. 迁移前代码审查

必须审查：

- 模型 forward 是否能加入 `tt`。
- 新 `tt_embedder` 是否能从原 `t_embedder` 初始化。
- checkpoint warm-start 是否允许只缺 `tt_embedder.*`。
- 原 scheduler、`sigma_min`、`t_rescale` 是否和 TwinFlow 兼容。
- dataset/collate 是否会按长度、稀疏体素数、图像尺寸排序。
- branch mask 是否会和排序后的样本统计相关。
- microbatch 是否改变 branch 分布。
- DDP/FSDP 下 EMA 是否一致。
- cond / uncond / neg_cond 是否和预训练一致。
- logging 是否能写 NumPy/Torch scalar。
- save/resume 是否覆盖 student、optimizer、EMA、TwinFlow state。
- eval 是否固定 sample、seed、noise、cond。

## 3. 核心迁移实现

TwinFlow full 训练语义：

```text
t ~ Beta(time_dist_ctrl[0], time_dist_ctrl[1]) * time_dist_ctrl[2]
tt = t - U(0,1) * consistc_ratio * t
t_min = t - U(0,1) * t * min(0.05, consistc_ratio)

e2e: t=1, tt=0
mul: tt=t_min
any: tt=tt
adv: make fake x from model, then use negative-time branch
```

推荐 full branch：

```text
probs = {"e2e": 1, "mul": 1, "any": 1, "adv": 1}
```

实现要点：

- local batch 最好至少 `4`。
- official partition 先在完整 local batch 上构造，再 microbatch。
- 如果原 collate 会排序，branch assignment 要随机化，避免 branch 和样本大小相关。
- `dist_match` 只对 e2e 分支生效；target 不应反传。
- RCGM/enhanced target 若使用 EMA teacher，要保存和恢复 EMA。
- LoRA 是可选路径；本次 TRELLIS 迁移使用 full-parameter，不使用 LoRA。

## 4. 配置基线

保守 baseline：

```text
batch_size_per_gpu=4
lr=1e-5
optimizer=AdamW
fp16-master
8 GPU DDP
probs={"e2e":0,"mul":1,"any":1,"adv":0}
estimate_order=2
enhanced_ratio=2.0
dist_match=0
image_free=false
t_rescale=false
cfg_eval=0
```

TwinFlow distribution-distillation：

```text
batch_size_per_gpu=4
batch_split=2 or 4
lr=1e-5
optimizer=AdamW
fp16-master
8 GPU DDP
probs={"e2e":1,"mul":1,"any":1,"adv":1}
dist_match=0.5
estimate_order=2
enhanced_ratio=2.0 initially
image_free=false initially
t_rescale=false
cfg_eval=0
```

OOM 时第一优先级只改：

```text
batch_split: 2 -> 4
```

不要同时改 branch、lr、batch、dist_match，否则实验不可归因。

## 5. Smoke 顺序

长训前按顺序跑：

1. JSON/YAML 配置解析。
2. shell 脚本 `bash -n`。
3. Python `py_compile`。
4. synthetic 1-step。
5. real-data fetch preflight。
6. 单卡真实 1-step。
7. 多卡 DDP 1-step。
8. save/resume 1-step，强制 `i_log=1`。
9. eval smoke，输出 `cond + GT + original + few`。
10. decode smoke，输出 PLY/GLB/front-view PNG/contact/manifest。

失败就停，不要启动长训。

## 6. 训练监控

每次检查：

```bash
squeue -j <jobid> -o '%.18i %.9P %.35j %.8u %.2t %.10M %.6D %R'
tail -n 40 <output_dir>/log.txt
tail -n 120 slurm_logs/slurm_output_<jobid>.log
srun --jobid=<jobid> --overlap --ntasks=1 nvidia-smi
```

日志必须检查：

- step 是否增长。
- loss/mse/grad_norm 是否 finite。
- branch count 是否符合配置。
- e2e 时是否出现 `e2e_loss`。
- dist-match 打开时是否实际进入 e2e 分支。
- GPU 是否忙。
- checkpoint 是否按计划写出。

不要被这些非致命噪声误导：

- MFS/ObjectStorage 偶发 `GetObject`。
- kaolin/ipyevents warning。
- 初始 20 step 比较慢。

## 7. 标准评估

输出列：

```text
cond
GT latent
original/pretrained model, standard NFE
distilled few 1 NFE
distilled few 4 NFE
distilled few 8 NFE
optional distilled any 2/4
optional distilled mul 30
```

每个样本保存：

```text
sample_cond_image_*.png
sample_gt_latent_*.npz
sample_undistilled_*.npz
sample_distilled_few_*.npz
*.ply
*.glb
*.png front-view render
comparison_all.png
manifest.json
```

质量报告必须包含：

- cond 是否和输出语义一致。
- GT 是否正常。
- 原模型是否正常。
- distilled few 1/4/8 是否坍塌。
- few 1/4/8 是否非单调。
- 身份、表情、眼睛、牙齿、口腔、文字、铭牌、局部细节问题。
- 每个问题一个图。

## 8. 验收标准

训练验收：

- 正常结束。
- checkpoint 完整。
- EMA checkpoint 可加载。
- resume 可跑。
- branch count 正确。
- loss finite。

可视化验收：

- 第一列是 `cond`。
- front-view。
- 无黑色三角描边伪影。
- GLB 可在 3D 软件打开。
- manifest 包含所有样本和模式路径。

科研验收：

- 用同一批 cond/noise/sample 比较。
- 原模型、GT、蒸馏 few 都在同一渲染管线下。
- 推理 cfg=0。
- 报告里问题与图片一一对应。

## 9. 过去错误示例

1. 错把 normalized training latent 当 VAE latent 解码，导致 GT 错。
2. CPU/PIL fallback 给三角面画黑边，导致 GT 满是黑点。
3. 用斜侧面视角验收，导致用户看到“躺着的侧面”。
4. 蒸馏后推理误开 CFG，重复条件引导。
5. batch 内排序和固定 branch mask 相关，导致 branch 数据分布偏。
6. microbatch 先切再分 branch，导致和官方 batch 语义不等价。
7. `t_rescale` 未审查，可能把 `t` 推出合理范围。
8. `json.dumps` 不支持 NumPy scalar，长训中途被日志写崩。
9. 前 2k 不 OOM 不代表后续不会 OOM，稀疏形状和分支会造成峰值变化。
10. contact sheet 没有 `cond`，无法判断条件一致性。

## 10. 新项目启动 Prompt 模板

```text
请在 <project_root> 上迁移 https://github.com/inclusionAI/TwinFlow。
先审查原项目与官方 TwinFlow 的模型 forward、t/tt、sampler、dataset/collate、microbatch、DDP/EMA、checkpoint、eval/render 全调用链。
实现后必须跑 synthetic smoke、real-data smoke、多卡 smoke、resume smoke。
训练前给出 config diff 和验收标准。
训练中监控 loss/branch/GPU/OOM/NaN。
训练后标准评估必须输出 cond + GT + 原模型 + 蒸馏 few 1/4/8，并导出 NPZ/PLY/GLB/front-view PNG/contact/manifest。
推理 cfg 默认 0。
每个质量问题必须配对应图片。
把失败经验和关键决策写入 handoff。
```
