# PIDG + Original 3DGS — ResearchBoost Acceleration Experiments

本仓库整理两组单张 **NVIDIA A800-SXM4-80GB** 实验：PIDG 光流模块优化，以及原始 3DGS 光栅化反向优化。包含固定上游版本、补丁、公开数据字节校验、模块/完整训练驱动及结果。六个原有查看器在实验期间保留；所有计时均为**共享卡探索性结果**。

使用了 [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent) 的流程、技能卡和未修改的通用 `numerics` 组件。未启动上游 `clawo`；**不宣称完整 ResearchBoost 四门验收通过**。

可直接复制的摘要：[SPEEDUP.md](SPEEDUP.md)。

## 1. PIDG：保留光流优化，拒绝光栅化候选

保留优化为 gather 反向中省略**精确为零**的梯度向量；前向、非零/非有限梯度、原精度及高阶求导路径保留。默认候选使用原始 CUDA 光栅化算法。

### 五场景模块组对比

三列均测同一模块组：**三次原生渲染 + 原速度网络 + 光流/重建损失 + 反向 + 五张 PNG 实际写出**。列名表示替换哪个组件；不是单个光流算子的耗时，也不是完整训练步。

| PIDG 场景 | 仅光流：保留版 | 仅光栅化：被拒候选 | 两者合用：被拒候选 |
|---|---:|---:|---:|
| dry_ice | 4.11× | 1.02×，未超波动 | 4.41×，时钟证据不足 |
| rubber_duck | 4.94× | 1.00×，未超波动 | 5.13× |
| balls_reaction | 6.08× | 1.00×，未超波动 | 6.17× |
| mechanics_cloth | 17.86×，时钟证据不足 | 1.00×，未超波动 | 17.47×，时钟证据不足 |
| motion_kuro | 1.03×，未超波动 | 1.10× | 1.12× |

- 主模块计时：每臂 1 次预热，3 窗 × 1 次，原版→候选→原版，三个独立进程。数值被拒列仅作性能诊断，不能当作可用收益。
- `rubber_duck` 原配置关闭光流；该行光流实验是额外组件覆盖，不能解释为其默认训练加速。
- `mechanics_cloth` 的 17.86× 缺少足够活动时钟证据，不作为已分辨收益；`motion_kuro` 的 1.03× 未超过实测波动。

### 保留版 dry_ice 较长复测

| 范围 | 原版 | 光流优化 + 原光栅算法 | 耗时比 |
|---|---:|---:|---:|
| 同一模块组，每臂 1 预热、3 窗 × 3 次 | 9369.788 ms | 2244.311 ms | **4.17×** |

实测分辨阈值约 3.27%，时钟可比、无运行错误；仍为探索性模块观测。

### 数值与范围

- **R03 光流版**对应 `checks[1]`：2435 项模块检查通过，**1360** 项逐位相同。**R06 最终保留版**对应 `checks[4]`：2435 项通过，**1368** 项逐位相同，0 项未决；较长复测也属于 R06。
- R06 的原始标签仍是 `both`：表示从候选目录选择光流和光栅器两个模块。此时光栅 CUDA 源码已逐字节恢复为原版，候选本地二进制已重建，因此实际只有光流优化。被拒的“合用”特指含优化光栅器的 **R04**，与 R06 区分。
- R01 的 embedding gather 实现有 **82** 项连续量超界；R03 换成精确零梯度项省略实现后通过。R04 光栅化及其合用候选各有 **84** 项超界，均被拒；随后恢复原光栅器得到 R06。五次检查使用同一冻结门槛，完整映射与源码哈希见 [运行历史](docs/pidg-run-history.md) 和 [run-provenance.json](results/pidg/run-provenance.json)。
- 正式状态保持 **`formal_ResearchBoost_acceptance: false`**，**`formal_benefit_verdict: UNAVAILABLE_SHARED_WORKLOAD`**。模块数值 PASS 与 3.27% 实测分辨阈值不构成完整 ResearchBoost 验收通过。
- 数据为五场景的真实原始 100000 点初始化及前两张训练帧 `0002/0003`，原生宽度1600策略；模块不含形变网络、TV/Cauchy 物理、增点或整段训练。
- dry_ice 的 601 步原生短训完成并通过保存产物有限性检查；该短训**不提供全训练速度或收敛/画质结论**。原训练每100步才尝试光流，且受 warmup/下一帧条件限制，模块倍数不能外推。

原始表格及检查结果：[measurements.csv](results/pidg/measurements.csv)、[summary.json](results/pidg/summary.json)。

## 2. 原始 3DGS：Mip-NeRF360 三场景完整训练

使用 2023 原始实现；修改仅在光栅化反向：完整32线程且同高斯ID时做 FP32 局部梯度规约，其余情况保留原子累加。前向与训练配方保留。静态 3DGS 无光流分支。

每场景原版/候选均完成种子0/1、30000步原生训练，7000/30000步全部留出视角评估。官方尺度：bonsai `images_2`，bicycle/garden `images_4`；留出视角分别37/25/24张。

### 完整原生流程耗时

下表是可配对的**种子1**。包括新子进程初始化、训练、原生评估/保存；排除额外 CPU 产物检查，读取冻结驱动已有时间戳。原版种子0缺少统一时间边界，仅参与画质对比。bicycle/garden种子0的初版产物检查失败退出原样保留；独立严格检查恢复了其画质评估资格，未把失败改成退出0。

| 场景 | 原版 | 实验版 | 耗时比 |
|---|---:|---:|---:|
| bonsai | 1050.319 s | 816.463 s | **1.29×** |
| bicycle | 1587.130 s | 1436.742 s | **1.10×** |
| garden | 1632.278 s | 1520.652 s | **1.07×** |
| 三场景总流程 | 4269.727 s | 3773.857 s | **1.13×** |

有效计时要求原版/候选均完成且退出0、均有隔离 GPU 的新 CPU 产物检查及其开始时间戳，并绑定冻结的源码、数据、配置、种子与训练事件。这个规则在优化前固定。原版种子0的 bonsai 使用旧检查器、缺少同口径边界；bicycle/garden 的原始训练后检查 SIGBUS（−7），独立严格检查只恢复画质资格，因此三个种子0基线均不参与耗时配对。

每场景只有一组有效种子1配对，原版先采集、优化版随后运行，没有完整流程的原版→候选→原版重复计时；**完整流程的运行间噪声和置信区间未估计**。阶段探针的噪声不能替代这个缺失量。

最终高斯数分别为 bonsai **1275632→1271323（−0.34%）**、bicycle **6071245→5971655（−1.64%）**、garden **5927094→5951459（+0.41%）**。完整流程耗时比包含轨迹与增点后工作量变化，不能单独解释为等工作量的内核收益。数据见 [native-timing-results.csv](results/gs3d/native-timing-results.csv) 与 [计时资格/噪声说明](results/gs3d/native-timing-notes.json)。

### 固定高斯数量的阶段训练步

| 场景 | checkpoint100 | checkpoint7000 | checkpoint29900 |
|---|---:|---:|---:|
| bonsai | 2.12× | 1.33× | 1.28× |
| bicycle | 2.17× | 1.12× | 1.05× |
| garden | 1.78× | 1.09× | 1.04× |

27个计时分支有效，原版—候选—原版，实测时钟一致。每臂20预热、3窗×175实际步骤，窗前在计时外精确恢复检查点。固定阶段探针排除原生增密、重置、评估、保存等事件，**不能替代完整训练计时**。

### 最终画质：全部留出视角，两个种子的均值

| 场景 | PSNR 原版→实验版（dB） | ΔPSNR | ΔSSIM | ΔL1 |
|---|---:|---:|---:|---:|
| bonsai | 32.1943→32.2318 | +0.0375 | +0.0000151 | −0.0000601 |
| bicycle | 25.2363→25.2222 | −0.0140 | −0.0007797 | −0.0000666 |
| garden | 27.3985→27.3963 | −0.0023 | −0.0001121 | +0.0000259 |

PSNR/SSIM 越高越好，L1越低越好。PSNR沿用该历史版本 `train.py` 的3×H×W调用口径，逐通道计算后平均，再平均全部留出视角。7000/30000步的逐种子均值、最差值见 [quality-results.csv](results/gs3d/quality-results.csv)；全部逐帧比较见 [summary.json](results/gs3d/summary.json)。未测 LPIPS。

### 未解决的验证项

**冻结数值结果仍为 `INCORRECT / exit 2`：1287项中11项连续量超界；精确状态、运行错误、dtype、math-mode失败均为0。** 11项均保留为未解决，见 [numerics.json](results/gs3d/numerics.json)。

原版自对照也分别有2项/3项连续失败，详见 [original-copy-controls.json](results/gs3d/original-copy-controls.json)；这些与候选11项失败分开呈现，不将候选失败归因于原版。未继续放宽门槛或重复抽样直至通过。

画质共有 **52个端点超出优化前冻结的双种子描述带**（3均值、2最差值、47逐帧，28项方向为退化，其余改善）。两个种子不足以证明统计画质等价。独立 N1 源码审查及18套实际内核路径核对完成，不能替代数值认证。

本次按明确授权继续完整训练与画质探索。**3DGS 补丁仅为实验版；不宣称 ResearchBoost 验收通过或数值/画质等价。** 第一版稀疏分组规约在 bonsai7000 仅0.76×，已放弃；其补丁单独保留。

## 3. 复现代码

```bash
git clone https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost.git
cd Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost
# 使用独立的兼容 PyTorch/CUDA 环境，先阅读 docs/environment.md。
python -m pip install -r requirements-repro.txt
python -c "import torch, torchvision, cv2; print(torch.__version__, torch.version.cuda)"

python scripts/prepare.py pidg       # 默认：仅保留光流优化
python scripts/prepare.py gs3d       # 明确标记为数值未解决的实验补丁
python scripts/gpu.py --gpu YOUR_ALLOCATED_GPU -- python scripts/build.py pidg
python scripts/gpu.py --gpu YOUR_ALLOCATED_GPU -- python scripts/build.py gs3d
```

数据下载、配置、完整测量命令见 **[docs/reproduce.md](docs/reproduce.md)**；版本与公开可移植化差异见 [docs/environment.md](docs/environment.md)。脚本不停止查看器，不分配新GPU，不改共享包。已有部署使用统一GPU锁时，通过 `scripts/gpu.py --lock PATH` 指向现有锁；避免与旧启动器同时运行。

代码按固定提交获取，构建产物树内保存；公开数据逐文件校验 SHA256。机器路径/GPU UUID 在本地生成配置后写入忽略的 `workspace/`。新机器必须重建参考记录并重新冻结；历史门槛、二进制哈希及认证不能继承。

| 上游 | 固定提交 |
|---|---|
| PIDG | `b77fd2107a50c875d7219bb4f82388d5e5ea1319` |
| flow raster | `2146a1c0dc3558644af420f20c21a0ad07ffa67a` |
| Original3DGS | `b2ada78a779ba0455dfdc2b718bdf1726b05a1b6` |
| 3DGS raster | `8064f52ca233942bdec2d1a1451c026deedd320b` |
| KNN | `44f764299fa305faf6ec5ebd99939e0508331503` |
| GLM | `5c46b9c07008ae65cb81ab79cd677ecc1934b903` |
| ResearchBoost-Agent | `ab7b569789317bf872c041a1d596454d58dcfdeb` |

原始数据/模型/大体积GPU trace、编译二进制与部署状态不入Git；本仓库保存可重新生成它们的代码和公开结果。来源及原许可见 [NOTICES.md](NOTICES.md)。
