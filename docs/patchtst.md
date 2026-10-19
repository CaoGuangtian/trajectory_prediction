# PatchTST Raw 使用说明

## 实验定义

`patchtst_raw_v1` 是独立的深度学习对照模型。它使用标准化后的原始轨迹，不额外平滑高度：

- 输入：1 Hz 的 `t-59` 至 `t`，共 60 点；
- 输出：`t+1` 至 `t+60`，共 60 点；
- 特征与标签：以预测起点为原点的局部 ENU 三维位移；
- 标签始终来自项目的标准化轨迹，不使用平滑后的未来数据；
- train 用于参数与归一化器，validation 用于早停和最优 checkpoint，test 只由独立评估命令读取；
- 主实验不按 `trusted` 过滤，窗口不跨 `segment_id`；
- 插值或异常速度目标通过损失掩码排除，但不会删除整个预测起点。

模型骨架与官方 PatchTST 参考实现对齐，保留 patching、channel-independent encoder、residual attention、BatchNorm、可学习位置编码、RevIN 和 flatten prediction head；项目侧继续使用直接多步预测和 masked Huber loss。没有复用参考仓库的数据切分、全轨迹平滑或逐轮测试逻辑。

## Python 环境

预处理环境没有强行固定 PyTorch，因为 CPU 版与不同 CUDA 版本的安装包不同。先激活并确认项目环境：

```powershell
conda activate traj-pre
python -c "import sys; print(sys.executable)"
python -c "import numpy, pyarrow; print(numpy.__version__, pyarrow.__version__)"
```

`numpy` 和 `pyarrow` 已由项目的 `environment.yml` 管理。确认解释器指向 `traj-pre` 后，再从 PyTorch 官方安装选择器中选择与机器匹配的 CPU 或 CUDA 命令：<https://pytorch.org/get-started/locally/>。不要从未激活环境的基础 Python 执行安装。

安装后可以检查运行设备：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 设备选择

配置默认使用 `device: auto`：检测到可用 CUDA 时使用 `cuda:0`，否则自动回退到 CPU。混合精度只会在 CUDA 上启用。

也可以从命令行显式覆盖：

```powershell
--device cpu
--device cuda --cuda-device 0
--device auto
```

如果显式指定 CUDA 但 CUDA 不可用，程序会直接报错，不会静默改用 CPU。

## 训练

以下命令只使用 subset1 的 train/validation，不读取 test：

```powershell
$env:PYTHONPATH = "src"
python -s -m trajectory_prediction.cli.train_patchtst --subset 1 --device auto
```

断点续训：

```powershell
$env:PYTHONPATH = "src"
python -s -m trajectory_prediction.cli.train_patchtst --subset 1 --device auto --resume
```

训练产物位于 `outputs/models/patchtst/subset_1/patchtst_raw_v1/training/`，包括最优/最近 checkpoint、train-only 归一化参数、训练历史和训练 manifest。

`training_history.csv` 会逐轮记录 train/validation loss，以及训练、验证和整轮耗时，便于比较本地 CPU 与服务器 GPU 的吞吐量。

## 冻结后的 test 评估

```powershell
$env:PYTHONPATH = "src"
python -s -m trajectory_prediction.cli.evaluate_patchtst --subset 1 --device auto
```

评估会生成与现有前端兼容的：

- `manifest.yaml`
- `metrics_by_horizon.csv`
- `metrics_by_segment.parquet`

重启前端后即可发现该实验。只有在方案与 checkpoint 已冻结后才运行 test 评估，不能根据 test 结果反复调参。

## 参考实现

模型骨架适配自本地 `D:\code\RealTimeTrajectory\reference\patchTST` 中的官方 PatchTST 实现，具体许可和修改范围见 `THIRD_PARTY_NOTICES.md`。其通用 CSV 数据集、随机拆分、平滑代码和实验循环不适用于本项目，未被接入。

