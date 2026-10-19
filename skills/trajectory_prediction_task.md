# 飞行轨迹预测项目约定

本文件供后续参与项目的 Codex/AI 代理读取。除非用户明确修改，下列内容是实现、实验和结果解释的默认依据。

## 项目目标

根据历史 ADS-B 轨迹预测飞机未来三维轨迹，重点评价对训练阶段未见飞机的泛化能力。第一版模型不使用 `aircraft` ID 作为输入特征。

- 主标签：`longitude`、`latitude`、`geoAltitude`。
- 模型内部优先使用以预测起点为原点的局部 ENU 坐标。
- `baroAltitude` 可以作为辅助信息，但不能与 `geoAltitude` 混成同一标签。
- 不得跨 subset 或 `segment_id` 连接轨迹、生成历史窗口或预测目标。

## 固定输入输出协议

- 数据已经按 1 Hz 标准化。
- 输入 60 点：`t-59` 至 `t`，包含预测起点，共 60 点。
- 输出未来 60 点：`t+1` 至 `t+60`。
- 误差曲线至少报告 1、5、10、20、30、60 秒。
- 预测必须是因果、开环的；`t` 之后的观测不得用于状态更新、参数估计或模型选择。

## 数据划分与公平比较

- 必须先按 `aircraft` ID 划分 train、validation、test，再生成窗口；三个分区的飞机集合互不相交。
- train 用于训练、归一化统计和数据驱动参数拟合。
- validation 用于超参数、模型选择和早停。
- test 只用于冻结方案后的最终评价，不能根据 test 结果反复调参。
- 主对比统一使用完整 `test` 分区，不按 `trusted` 过滤，即同时保留 `true`、`false`、`unknown`。
- 可额外报告按 `trusted` 分组的辅助结果，但不能替代完整 test 主结果。
- 所有对比模型必须使用相同 subset、预测起点、60 点历史、60 秒未来窗口、目标点和质量过滤规则。
- 当前本地资源只允许实际运行 subset1；不得擅自启动全量实验。
- 模型失败或历史不足时不能静默删样本；使用预先定义的因果回退，并报告回退数量、回退率和覆盖率。

## 当前模型分类

1. Position Hold：位置保持基础基线。
2. OLS Constant Velocity：用最近 31 个历史点拟合速度的匀速直线外推基础基线。
3. CV Kalman Filter：固定参数的单 CV 模型 Kalman Filter，对照滤波本身的影响。
4. Standard IMM：固定参数、固定转移矩阵的 CV + 左转 CT + 右转 CT + Singer 模型库；不使用 OLS 门控，不针对测试轨迹自适应估计 IMM 参数。
5. Gated Motion IMM：保留原 gated v2 行为；非机动样本走最近 31 点 OLS-CV，因果检测到持续转弯或加速后启用 IMM。它是工程增强模型，不作为“标准 IMM”。
6. PatchTST Raw：独立深度学习对照模型；输入标准化后的未平滑轨迹，以预测起点为原点转换为局部 ENU，使用 60 点历史直接预测未来 60 点。
7. 后续模型：深度学习与 IMM 的融合模型。结构未确定前不要自行实现。

旧的自适应 IMM v1 不再作为独立实验模型、命令或前端条目保留。其内部公共数学函数若仍被 Gated Motion IMM 复用，不等于保留 v1 实验。

## IMM 研究约束

- Standard IMM 的模型参数、初始模型概率和 Markov 转移矩阵必须来自配置，并在 test 评估前冻结。
- Standard IMM 的 CT 必须同时包含固定左转率和固定右转率，避免单一带符号 CT 对相反方向转弯产生结构性偏差。
- 每个预测起点可以用截至 `t` 的历史观测执行滤波和更新；从 `t+1` 开始只能开环预测。
- Standard IMM 和 Gated Motion IMM 必须以不同 `model_type`、配置和输出目录注册，前端不得混为同一版本。

## 数据与质量处理

- 排序/分组至少使用 `subset + aircraft + timeAtServer`，并保留 `trajectory_id`、`segment_id`。
- 长时间缺口、时间非递增、不可解释的空间跳变必须切段；只允许对短缺口插值。
- 保存原始点/插值点标记以及异常速度标记。
- 归一化器和任何数据驱动阈值只能在 train 上拟合，再应用于 validation/test。
- `trusted=True` 表示通过独立验证，`false` 表示未通过或质量较低，缺失值为 `unknown`，不能将 unknown 当作 false。

## PatchTST 实验约束

- 第一版为 `patchtst_raw_v1`，不对高度或其他输入额外平滑；后续因果平滑增强必须建立独立版本，不能覆盖 Raw 对照。
- 每个窗口使用预测起点局部 ENU 三通道；不使用 `aircraft` ID，也不跨 `segment_id`。
- 输入变换只能使用截至预测起点的信息；监督标签和最终误差都以项目的标准化未来轨迹为准，不能换成模型平滑后的未来标签。
- 归一化尺度只能在 train 上拟合并冻结；validation 只用于早停/模型选择；训练命令不得读取 test。
- 损失使用目标质量掩码与 Huber loss；插值点或异常速度点可以从对应时刻的损失中排除，但不能因此静默删除整个预测起点。
- 设备默认 `auto`：CUDA 可用时选择 GPU，否则选择 CPU；也必须支持显式 `cpu`、`cuda` 和 CUDA 设备编号。
- 本地只允许 subset1；服务器训练也应先用 subset1 验证流程，扩大范围必须由用户明确决定。

## 评价与产物

- 所有模型产物与现有前端格式对齐：`manifest.yaml`、`metrics_by_horizon.csv`、`metrics_by_segment.parquet`。
- `Error by prediction horizon` 展示各预测时刻在有效目标上的聚合误差；具体是 mean、RMSE、median 或分位数，应以图例/字段名为准，不能笼统称为“平均误差”。
- 至少分别报告水平距离误差、垂直 MAE/RMSE 和三维误差，并报告样本数与覆盖率。
- 版本身份、配置哈希、数据来源、筛选条件、历史长度、预测时域和回退策略必须进入 manifest。
