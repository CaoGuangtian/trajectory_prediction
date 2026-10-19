# Aircraft Trajectory Explorer

本地离线航迹与模型评估查看器。当前版本同时读取 OpenSky 原始 Parquet、1 Hz 标准化轨迹 Parquet 和统一格式的模型评估输出，支持航迹检查以及多模型误差对比。

在项目根目录运行：

```powershell
conda run -n traj-pre python -s -m apps.experiment_viewer.app
```

浏览器访问：

```text
http://127.0.0.1:8050
```

默认数据文件：

```text
data/openskyDataset/subset_1/set_1.parquet
data/openskyDataset/subset_1/standardized_1hz/set_1_standardized_1hz.parquet
```

对比视图中：

- 蓝色曲线和实心点表示原始报告；
- 橙色曲线和实心点表示有原始观测支持的 1 Hz 标准化轨迹；
- 橙色空心点表示插值点；
- 红色叉号表示可疑运动区间的终点；
- 标准化轨迹严格按照 `segment_id` 断线。

采样间隔相关图仍然只分析原始报告。

如需使用其他同结构 Parquet，可以在启动前设置：

```powershell
$env:TRAJECTORY_DATA_PATH = "原始Parquet路径"
$env:TRAJECTORY_PROCESSED_DATA_PATH = "标准化Parquet路径"
```

标准化文件不存在时，应用会自动退化为仅原始数据模式。

## 模型评估

应用默认递归扫描 `outputs/` 下的评估 `manifest.yaml`。只要清单声明的
`metrics_by_horizon` 和 `metrics_by_segment` 文件存在，该次实验就会自动出现在
“Prediction evaluation”的模型多选框中。

当前评估区域包括：

- 不同模型的误差—预测时域曲线；
- 各预测时域的目标覆盖率和有效目标数量；
- 指定预测时域下的航迹段误差箱线图。

如需扫描其他输出根目录，可以在启动前设置：

```powershell
$env:TRAJECTORY_EVALUATION_ROOT = "模型评估输出根目录"
```
