# RoboMetaNorm mini

RoboMetaNorm mini 读取机器人数据集的本地元数据与有限代表证据，生成符合
《数据转换标准》的命名建议。流程以保守、可复核为原则：只有证据完整且通过本地
确定性校验的字段才会修改；未确认字段保持原名，并在复核文件中记录原因。

## 工作流程

机器人身份仅取自 `meta/info.json` 的 `robot_type`，并原样保留。当前流程暂时跳过远程
硬件画像查询和联网研究，不会用其他元数据推断、补全或改写机器人型号。

每个数据集最多执行一次 VLM 操作：在同一次请求中完成数据结构理解，以及相机、
`action` 和 `observation.state` 字段映射。最终名称、置信度、媒体属性、字段形状和机器
组件顺序仍由本地代码按《数据转换标准》确定性校验。映射失败、置信度不足或存在歧义
时，不修改源字段，并将数据集标记为需要复核。

本地证据保持有界：

- 机器人身份只读取 `meta/info.json` 的 `robot_type`；
- Parquet 只采集排序后的首、末 episode 代表证据；
- 每个相机只采集排序后的首、末媒体代表证据；
- 抽取的临时帧在单次处理结束后删除。

## 输入与输出

### 输入

每个数据集必须包含 `meta/info.json`，并至少包含 `data/` 或 `videos/` 之一。`--root`
既可指向单个数据集，也可指向包含多个数据集的根目录；目录布局支持平铺、任务分组和
自动识别。

程序仅采集有限的本地证据：

- 从 `meta/info.json` 读取数据集元信息；机器人身份仅采用其中的 `robot_type`；
- 从排序后的首、末 episode 采集 Parquet 代表证据；
- 从每个相机排序后的首、末媒体采集代表证据。

该策略用于控制证据规模，不执行全量数据扫描。媒体分析产生的临时帧会在当前数据集
处理结束后删除。

### 输出

`normalize` 在每个数据集的 `meta/` 目录生成三个文件：

- `info_norm.json`：保留源元信息，仅更新已确认的规范名称；
- `info_norm_review.json`：记录身份、字段映射、证据、处理决定和详细问题，并保存
  `info_norm.json` 的 SHA-256 摘要；
- `robo_annotation.yaml`：描述适配器、相机和机器通道，同时标记结果是否需要人工复核。

正常处理以及可控的复核、预检阻断、API key 缺失和 VLM 调用失败路径均会生成上述
产物。证据采集、程序执行或文件写入发生异常时，数据集状态为 `ERROR`，输出可能不完整。

程序不会改写源 `info.json`、Parquet 或媒体文件，也不会创建持久缓存、预览、日志或
汇总文件。`scan` 始终只读，不生成任何输出。

## 输出语义与校验规则

`robo_annotation.yaml` 包含固定顶层字段 `version`、`robot_type`、`adapter`、
`robot_channel_schema` 和 `review`。其中：

- `review.required: false` 表示结果通过 VLM 语义判断和本地确定性校验；
- `review.required: true` 表示结果仍需人工复核；
- `review.issues` 提供紧凑的问题摘要，完整诊断记录在 `info_norm_review.json`；
- `robot_type` 缺失或非法时写为 `null`，并将结果标记为需要复核。

VLM 确认相机类别和位置时，YAML 使用确认后的标准相机键；VLM 不可用或映射未确认时，
REVIEW YAML 仅依据安全的源字段方位词与媒体模态生成标准格式的候选键，无法满足标准
语法的相机不写入映射。复核状态下还可保留根据 `action` 与 `observation.state` 的一致、
连续结构得到的本地尽力推导（best-effort）机器通道。这些候选映射仅用于支持后续复核，
不代表已经由 VLM 确认相机安装位置。

标准相机键采用 `observation.images.cam_<位置>_<rgb|depth>`，并由 VLM 结合抽帧图像
判断本体相机或外部相机类别；其 YAML 值保留原始相机字段。`videos/` 下 RGB 图像使用
`_rgb` 后缀并对应目标编码 `av1`，`depth/` 下深度图使用 `_depth` 后缀并对应目标编码
`ffv1`。机器
通道采用 `arm.left/right.joint`、`arm.left/right.eef` 和
`gripper.left/right`。REVIEW 输出仅在 `action` 与 `observation.state` 的 gripper 字段名
和索引一致时补充对应通道，并优先采用明确的 `*_gripper_open` 标量。对于同序、连续的
`main_follower_joint_*` 向量，使用 `arm.main.*` 与 `gripper.main`；双臂结构仍使用
`left/right`。

关节预检在 VLM 调用前执行。无侧别的泛化名称、索引不连续或 `action` 与
`observation.state` 布局不一致时，数据集将被阻断并记录源文件、字段、索引和原始名称。
无效路径不会写入 YAML；满足结构约束的本地尽力推导通道可写入 REVIEW YAML，并明确
标记为需要复核。

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

使用本地示例数据集快速验证：

```bash
cd /home/baai/qmh/RoboMetaNorm-mini-implementation

python3 -m robometanorm scan \
  --root /home/baai/qmh/dataset/collect_data \
  --layout auto

export DASHSCOPE_API_KEY='你的 API Key'
python3 -m robometanorm normalize \
  --root /home/baai/qmh/dataset/collect_data \
  --layout auto \
  --ignore-vlm-network-errors
```

处理完成后，每个数据集的 `meta/` 目录应包含 `info_norm.json`、
`info_norm_review.json` 和 `robo_annotation.yaml`。

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
- `--vlm-retry-backoff-seconds`、`--vlm-max-tokens`；
- `--ignore-vlm-network-errors`：VLM 联网异常转为 REVIEW 输出，而不是将该数据集标记为
  ERROR。

缺少 API key、VLM 返回可分类失败，或开启上述开关后的联网异常，程序保持原值并输出可
复核原因，不会回退到硬编码型号规则。

## 当前边界

- 完整 URDF 不是转换前置条件，也不参与当前判断；
- 当前不处理触觉、声音或音频字段的命名；
- 不执行数值变换或媒体转码；
- 无法确认的普通字段保持源值；YAML 相机目标键必须满足标准语法，否则不写入映射。
- 恢复远程硬件画像后，同型号只查询一次并供后续数据集复用，是计划中的长期缓存方案，当前尚未实现。
