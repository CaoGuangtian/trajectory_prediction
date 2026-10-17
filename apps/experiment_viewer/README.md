# Aircraft Trajectory Explorer

本地离线航迹查看器。当前版本直接读取 OpenSky Parquet 数据，支持按飞机编号筛选、经纬高时间序列和经纬二维轨迹显示。

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
data/raw/openskyDataset/subset_1/set_1.parquet
```

如需使用其他同结构 Parquet，可以在启动前设置 `TRAJECTORY_DATA_PATH`。
