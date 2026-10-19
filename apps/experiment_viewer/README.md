# Aircraft Trajectory Explorer

本地离线航迹查看器。当前版本同时读取 OpenSky 原始 Parquet 和 1 Hz 标准化轨迹 Parquet，支持按飞机编号筛选、原始/标准化/对比视图、经度、纬度、几何高度、气压高度时间序列和经纬二维轨迹显示。

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
