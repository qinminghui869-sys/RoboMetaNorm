# RoboMetaNorm mini

RoboMetaNorm mini 读取机器人数据集的本地元数据与有限代表证据，生成符合
《数据转换标准》的命名建议。流程以保守、可复核为原则：只有证据完整且通过本地
确定性校验的字段才会修改；未确认字段保持原名，并在复核文件中记录原因。

## 工作流程

每个数据集最多执行两次 VLM 操作：

1. 联网硬件研究：综合 `info.json` 的 `robot_type`、
   `meta/common_record.json` 与 `meta/tasks.jsonl` 判断机器人身份，并查询相机与组件信息。
2. 全数据集映射：一次性把数据集中的相机、`action` 和 `observation.state` 字段映射到
   已研究的硬件画像。

VLM 只提供带来源的语义判断。最终名称、置信度、媒体属性、字段形状和机器组件顺序
均由本地代码按《数据转换标准》确定性校验。映射失败、来源不足、置信度不足或存在
歧义时，不修改源字段。

本地证据保持有界：

- 身份证据只读取上述三类元数据来源；
- Parquet 只采集排序后的首、末 episode 代表证据；
- 每个相机只采集排序后的首、末媒体代表证据；
- 抽取的临时帧在单次处理结束后删除。

## 输入与输出

数据集需包含 `meta/info.json`，并具有 `data/` 或 `videos/`。`--root` 可以是包含多个
数据集的目录，也可以是单个数据集目录本身；支持平铺和任务分组布局。

`normalize` 在每个数据集的 `meta/` 下生成：

- `info_norm.json`：保留全部源信息，只写入已确认的规范名称；
- `info_norm_review.json`：记录身份、字段映射、来源、决定、问题和
  `info_norm.json` 的 SHA-256。
- `robo_annotation.yaml`：仅当该数据集最终状态为 `PASS` 时生成的可执行描述文件。

不会改写源 `info.json`、Parquet 或媒体，也不会创建持久 cache、preview、日志或汇总文件。
`scan` 始终只读，不生成输出。

`robo_annotation.yaml` 使用固定顶层字段 `version`、`robot_type`、`adapter` 和
`robot_channel_schema`。相机键使用
`observation.images.cam_<位置>_<rgb|depth>`；原始相机字段仍作为值保留。已确认的机器
通道使用 `arm.left/right.joint`、`arm.left/right.eef` 和 `gripper.left/right`。仅当
`action` 与 `observation.state` 中同序、连续的 `main_follower_joint_*` 向量，经过 VLM
确认安全的单臂硬件身份及完整字段映射，且 `info.json` 的 `robot_type` 安全且非空时，才使用
`arm.main.joint`、`arm.main.eef` 和 `gripper.main`；双臂仍使用 `left/right` 键。

关节预检在 VLM 请求前执行。没有侧别的泛化名称，例如 `joint1`、`joint_1`、`j1`，或
无法构成连续单侧向量的关节布局，会直接阻止该数据集；复核文件会给出
`meta/info.json`、源字段、索引和原始名称。含 `left`/`right` 且连续的关节名称会进入
后续的硬件研究与映射确认，不会只因名称本身被丢弃。`main_follower_joint_*` 不完整、
两字段不一致、VLM 未能确认单臂硬件或映射不完整时，同样会阻止并记录复核原因；任何
泛化、无效或未确认的相机/机器组件路径都不会生成 YAML。

## 安装

需要 Python 3.10 或更高版本。

```bash
python3 -m pip install -e .
```

运行时依赖为 `pyarrow>=14.0` 和 `PyYAML>=6.0`。媒体证据采集需要系统可执行的
`ffprobe` 与 `ffmpeg`。

## CLI

只读扫描：

```bash
python3 -m robometanorm scan --root /path/to/datasets
# 或只扫描一个数据集
python3 -m robometanorm scan --root /path/to/dataset
```

生成规范化结果：

```bash
export DASHSCOPE_API_KEY='...'
python3 -m robometanorm normalize --root /path/to/datasets
```

安装后也可使用同名控制台命令：

```bash
robometanorm --help
```

两条子命令都接受：

- `--root`：数据集根目录；
- `--layout {auto,flat,task_grouped}`：目录布局。

`normalize` 还接受：

- `--vlm-endpoint`、`--vlm-model`、`--vlm-api-key-env`；
- `--confidence-threshold`；
- `--vlm-timeout-seconds`、`--vlm-max-retries`；
- `--vlm-retry-backoff-seconds`、`--vlm-max-tokens`。

缺少 API key 或网络查询失败时，程序保持原值并输出可复核原因，不会回退到硬编码型号规则。

## 当前边界

- 完整 URDF 不是转换前置条件，也不参与当前判断；
- 当前不处理触觉、声音或音频字段的命名；
- 不执行数值变换或媒体转码；
- 无法确认的字段始终保持源值。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.unit.test_architecture -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.unit.test_vlm -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```
