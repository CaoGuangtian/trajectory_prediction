# devTools 脚本工具使用说明

本目录存放项目开发过程中使用的辅助脚本。本说明统一记录这些脚本的用途、运行环境和调用方法。

除非另有说明，命令都应在 `trajectory_prediction` 项目根目录执行，并使用 Conda 虚拟环境 `traj-pre`：

```powershell
conda run -n traj-pre python <脚本路径> [参数]
```

## OpenSky CSV 转换为 Parquet

脚本：`devTools/convert_opensky_csv_to_parquet.py`

该脚本将 OpenSky LocaRDS CSV 流式转换为 Parquet，不会把完整 CSV 一次性加载到内存。它只改变存储格式，不改变字段含义，其中 `measurements` 仍保存为字符串。

脚本的默认输入为：

```text
data/raw/openskyDataset/subset_1/set_1.csv
```

默认输出到输入文件旁边：

```text
data/raw/openskyDataset/subset_1/set_1.parquet
```

运行默认转换：

```powershell
conda run -n traj-pre python devTools/convert_opensky_csv_to_parquet.py
```

指定输入和输出路径：

```powershell
conda run -n traj-pre python devTools/convert_opensky_csv_to_parquet.py `
  --input data/raw/openskyDataset/subset_1/set_1.csv `
  --output data/raw/openskyDataset/subset_1/set_1.parquet
```

常用参数：

- `--compression`：压缩算法，默认 `zstd`，还可使用 `snappy`、`gzip`、`brotli`、`lz4` 或 `none`。
- `--block-size-mib`：流式读取 CSV 的块大小，默认 64 MiB。
- `--row-group-rows`：每个 Parquet row group 的最大行数，默认 250,000。
- `--overwrite`：允许在转换成功并完成校验后替换已有输出文件。
- `--help`：显示全部参数。

脚本要求 `traj-pre` 环境中已安装 `pyarrow`。如尚未安装，可先执行：

```powershell
conda install -n traj-pre -c conda-forge pyarrow
```

写入过程中会先生成临时文件；只有转换完成且行数、字段校验通过后，才会生成或替换正式 Parquet 文件。如果转换失败或被中断，不会留下不完整的正式输出文件。

## OpenSky 实时演示数据生成

脚本 `prepare_opensky_realtime_demo.py` 会从 `subset_1` 流式筛选可信长轨迹，将其平滑并平移到中国区域，再合成一段 2 小时、同时活动飞机数保持在 10～20 架的回放 CSV。

`P1` 是由随机种子控制的唯一 4～6 位随机数，不是顺序编号。相同参数和种子会得到相同结果。

### 运行

在 `trajectory_prediction` 根目录执行：

```powershell
conda run -n traj-pre python devTools/prepare_opensky_realtime_demo.py
```

默认输入：

```text
data/raw/openskyDataset/subset_1
```

默认输出：

```text
data/processed/civil_data_for_realtime_web_show/20250510.csv
```

筛选、调度和并发统计保存在输出目录的 `report/` 子目录。脚本会完整扫描约 1.22 GB 的主文件两次，运行时间取决于磁盘速度，但不会把整份 CSV 一次性载入内存。

### 常用参数

```powershell
conda run -n traj-pre python devTools/prepare_opensky_realtime_demo.py `
  --seed 20250516 `
  --duration-hours 2 `
  --min-aircraft 10 `
  --max-aircraft 20 `
  --target-center-lon 112 `
  --target-center-lat 32
```

- `--seed`：控制轨迹调度和随机 `P1`，便于复现。
- `--replay-start`：合成回放开始时间，默认 `20250510 08:00:00.000000`。
- `--min-duration-minutes`：候选连续航段最短时长，默认 30 分钟。
- `--candidate-limit`：第二遍扫描最多提取的候选飞机数，默认 100。
- `--no-smoothing`：关闭异常点修复和平滑，不建议用于正式演示。
- `--help`：查看全部参数。

### 转换成 JSON

生成 CSV 后，在 `RealTimeTrajectory` 项目中执行：

```powershell
python tcp_dispatcher_demo/dataTransform/scripts/convert_extracted_csv_to_json.py `
  --input-dir ../trajectory_prediction/data/processed/civil_data_for_realtime_web_show `
  --output-dir tcp_dispatcher_demo/dataTransform/data/opensky_demo_json
```

输出目录顶层只有回放 CSV；其他报告放在 `report/` 中，因此不会被转换脚本误处理。
