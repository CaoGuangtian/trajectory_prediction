# IMM–PatchTST 残差融合使用说明

## 模型定义

`imm_patchtst_residual_v1` 使用冻结的 Standard IMM 生成物理预测，并训练一个与 `patchtst_raw_v1` 同结构的 PatchTST 预测修正量：

\[
\hat Y_{fusion}=\hat Y_{IMM}+f_\theta(X)
\]

- 两个分支使用相同的 60 点历史和预测原点；
- Standard IMM 使用原始米制局部 ENU 历史与因果观测质量掩码；
- PatchTST 使用训练集尺度归一化的历史输入；
- IMM 预测除以相同尺度后与 PatchTST 残差相加；
- 未来 60 点标签不平滑；
- IMM 没有可训练参数，第一版从头训练残差 PatchTST；
- 模型选择只使用 validation，test 只由独立评估命令读取。

配置入口：

```text
configs/models/imm_patchtst_residual_v1.yaml
```

该配置组合：

```text
configs/models/patchtst_raw_v1.yaml
configs/models/standard_imm_v1.yaml
```

## 历史观测掩码

标准化 Parquet 已包含 `is_interpolated` 与 `is_speed_suspicious`。数据窗口会据此生成历史观测掩码。掩码为 false 的时间步仍执行 IMM 状态预测，但按 IMM 配置跳过测量更新。无需重新预处理或修改现有数据文件。

PatchTST 第一版仍使用完整的固定 1 Hz 历史输入；历史掩码只控制 IMM 测量更新，不作为额外神经网络通道。

## 第一步：生成 train 和 validation IMM 缓存

IMM 不会在每个 epoch 重复运行。训练前先生成与 PatchTST 窗口严格对齐的缓存：

```powershell
$env:PYTHONPATH = "src"
python -s -m trajectory_prediction.cli.build_imm_patchtst_cache --subset 1
```

默认只生成 train 和 validation。缓存包含未来 1～60 秒连续预测、预测原点模型概率和 fallback 标记，并校验：

- 融合、PatchTST 与 IMM 配置哈希；
- split 与窗口数量；
- 窗口顺序和语料指纹；
- 预测数组形状。

缓存目录：

```text
outputs/cache/imm_patchtst/subset_1/imm_patchtst_residual_v1/
```

## 第二步：训练融合模型

```powershell
$env:PYTHONPATH = "src"
python -s -m trajectory_prediction.cli.train_imm_patchtst --subset 1 --device auto
```

训练产物目录：

```text
outputs/models/imm_patchtst/subset_1/imm_patchtst_residual_v1/training/
```

支持：

```text
--resume
--overwrite
--device cpu
--device cuda --cuda-device 0
```

训练命令不会创建或读取 test corpus，也不会自动生成 test 缓存。

## 第三步：冻结方案后生成 test 缓存

```powershell
$env:PYTHONPATH = "src"
python -s -m trajectory_prediction.cli.build_imm_patchtst_cache --subset 1 --split test
```

这一步应在模型结构、训练设置与验证集选择方案冻结后执行。

## 第四步：测试集评估

```powershell
$env:PYTHONPATH = "src"
python -s -m trajectory_prediction.cli.evaluate_imm_patchtst --subset 1 --device auto
```

默认读取训练目录中的 `checkpoints/best.pt` 与 `normalization.json`，输出：

```text
manifest.yaml
metrics_by_horizon.csv
metrics_by_segment.parquet
```

输出格式与现有实验前端兼容。

## 第一版实验边界

当前实现刻意不包含：

- Gated IMM 替换；
- IMM 模型概率作为 PatchTST 输入；
- 协方差或创新序列输入；
- 动态融合权重；
- Raw PatchTST checkpoint 热启动；
- Smooth PatchTST 残差分支。

这些应在最小残差融合有效后逐项加入并做消融，避免同时改变多个变量。
