# RoboMetaNorm

RoboMetaNorm 是面向机器人数据集的非破坏性元数据规范化工具。它扫描数据集，结合有限的 Parquet 样本和视频证据生成规范化建议与人工复核清单，不直接修改任何源数据。

## 设计原则

- **源数据只读**：不修改 `info.json`、Parquet、视频、图像或 Depth 文件。
- **规范内置**：相机类别、位置词表、字段模板和编码要求固化在代码中，运行时不依赖外部标准文件。
- **多证据推理**：机器人型号联网拓扑、本地画面、同数据集其他相机和元数据共同决定相机名称；VLM 不直接输出最终字段名。
- **兼容推理**：强证据不足但存在唯一兼容结论时生成可复核建议；只有真正无法区分或命名冲突时保留源字段。
- **结果可追溯**：规范建议、复核原因、候选项和证据分别写入数据集的 `meta/` 目录。
- **批处理隔离**：单个数据集失败不会中断同一批次中的其他数据集。

## 支持范围

| 能力 | 说明 |
| --- | --- |
| 数据集发现 | 支持平铺和任务分组目录，可自动识别或显式指定布局 |
| 前置检查 | 检查 RGB、action、observation、主摄像头等必要证据 |
| 相机字段 | 规范 RGB/Depth 字段名，检查媒体 FPS 与 shape，建议 AV1/FFV1 编码 |
| 机器字段 | 检查维度、跨 Episode 布局、父子切片、action/state 一致性及夹爪量程 |
| 机器人身份 | 合并机器人型号元数据，为相机拓扑和机器结构提供可追溯的强证据 |
| 默认 VLM | 默认使用 DashScope `qwen3.7-plus`，为相机拓扑、本地画面、机器字段和夹爪方向提供受约束的结构化判断 |
| 审核输出 | 生成规范化元数据建议和结构化人工复核清单 |

机器字段规范仅适用于末端执行器为夹爪的机械臂。数据集包含灵巧手、手指关节、骨架或关键点字段时，机器字段阶段保留整组源字段，不调用机器字段 VLM，并生成一条通用人工复核项。

本工具不改写数值，不推断未知单位，不依赖 URDF，也不执行四元数转欧拉角、骨架或关键点转换。需要夹爪数值归一化时，仅输出可审计的转换建议，源 Parquet 和源字段保持不变。

## 环境要求

- Python 3.10+
- NumPy 与 PyArrow：随主依赖安装，用于 Parquet 有限样本画像
- FFprobe 与 FFmpeg：用于视频探测和抽帧
- DashScope API Key：建议通过 `DASHSCOPE_API_KEY` 配置；缺少密钥时不会发送无认证请求，无法完成的语义判断会进入兼容推理或人工复核

安装项目：

```bash
python3 -m pip install -e .
```

如需 Depth 预览能力，可安装可选依赖：

```bash
python3 -m pip install -e '.[depth]'
```

## 输入目录

扫描入口固定为每个数据集的 `meta/info.json`。支持两种布局：

```text
# 平铺布局
/path/to/datasets/
└── <dataset>/
    ├── meta/info.json
    ├── data/
    ├── videos/
    └── depth/                 # 可选

# 任务分组布局
/path/to/datasets/
└── <task>/
    └── <dataset>/
        ├── meta/info.json
        ├── data/
        ├── videos/
        └── depth/             # 可选
```

数据集至少需要 `data/` 或 `videos/` 目录。扫描会忽略 `.git`、`.cache` 和 `__pycache__`。

规范化阶段还会按以下优先级读取可选身份元数据：

1. `meta/info.json` 的 `robot_type`，并兼容 `root_type`。
2. `meta/common_record.json` 的 `machine_id` 型号后缀。
3. `meta/tasks.jsonl` 中明确的机器人型号标识。

弱证据不能覆盖强证据；同一机器人系列的低优先级证据可以补充更具体的型号。不同系列发生冲突时采用高优先级结果，并生成 `ROBOT_IDENTITY_CONFLICT` 人工复核项。任务自然语言不会因包含 `hand`、`aloha` 等普通词而被当成机器人型号。

## 快速开始

先执行只读扫描，检查数据集状态和复核数量：

```bash
robometanorm scan --root /path/to/datasets
```

生成规范化建议：

```bash
robometanorm normalize --root /path/to/datasets
```

使用 `--layout auto|flat|task_grouped` 指定目录布局，默认值为 `auto`。如果尚未安装命令行入口，可直接从源码运行：

```bash
PYTHONPATH=src python3 -m robometanorm scan --root /path/to/datasets
```

### 默认 VLM

`normalize` 默认使用以下配置：

- endpoint：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- model：`qwen3.7-plus`
- API Key 环境变量：`DASHSCOPE_API_KEY`

API Key 通过环境变量注入后可直接运行：

```bash
export DASHSCOPE_API_KEY="<secret>"

robometanorm normalize --root /path/to/datasets
```

命令行参数用于覆盖默认连接，例如：

```bash
export CUSTOM_VLM_API_KEY="<secret>"

robometanorm normalize \
  --root /path/to/datasets \
  --vlm-endpoint https://vlm.example.com/v1 \
  --vlm-model your-model \
  --vlm-api-key-env CUSTOM_VLM_API_KEY
```

不要将密钥写入命令、配置文件或版本库。缺少密钥时，程序不会发送外部请求；确定性检查仍会执行，其他字段会结合现有证据推理，最终无法区分时保留源名并进入人工复核。可按需设置以下参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--vlm-endpoint` | DashScope compatible-mode URL | VLM 服务地址 |
| `--vlm-model` | `qwen3.7-plus` | VLM 模型名称 |
| `--vlm-api-key-env` | `DASHSCOPE_API_KEY` | 保存 API Key 的环境变量名 |
| `--confidence-threshold` | `0.85` | 相机 VLM 结果的自动采纳阈值 |
| `--vlm-timeout-seconds` | `120` | 单次请求超时秒数 |
| `--vlm-max-retries` | `2` | 网络异常、HTTP 429 和 5xx 的最大重试次数 |
| `--vlm-retry-backoff-seconds` | `1.0` | 指数退避的初始等待秒数 |
| `--vlm-max-tokens` | `1024` | 非特定兼容端点的最大输出 token 数 |

机器字段自动采纳使用更严格的固定门槛：置信度不低于 `0.92`、结果无歧义、无需数值转换且维度一致；有量纲字段还必须在源元数据中明确单位。复合字段必须由连续分段完整覆盖，任一分段不满足要求时保留整个源字段。

相机命名先区分本体相机与外部相机，再判断方位、部位和模态。`image_left`、`image_top`、`cam_high_rgb` 等源字段字符串只作为弱提示，不能单独决定最终名称：`left` 可能表示外部左侧，也可能表示左腕、左臂或机器人左侧的其他本体相机。

机器人型号明确时，程序通过 DashScope Responses API 的 `web_search` 查询该型号的本体相机拓扑；同一型号在单次运行中只查询一次。联网结果只限定候选槽位，不能直接确认改名。本地代表帧会独立判断相机类别、方位和部位，随后与同数据集的其他相机执行一对一约束推理：

- `CONFIRMED`：联网拓扑与本地画面高置信一致，目标名合法且无冲突。
- `INFERRED`：强证据不完整，但源字段、媒体元数据、已确认相机和唯一剩余拓扑槽位能形成唯一兼容结论；生成标准名称并进入人工复核。
- `UNRESOLVED`：全部证据不足、多个候选无法区分或目标冲突；保留源字段并进入人工复核。

例如 `airbot_mmk2` 的 `cam_high_rgb` 只有在拓扑与本地画面共同指向头部相机时才确认为 `observation.images.cam_head_rgb`；若左右腕相机已可靠占用对应槽位，剩余字段也可推断为头部相机，但等级为 `INFERRED`。相机位置无法确认不影响独立的模态与编码判断：可确认的 RGB 仍建议 AV1，Depth 仍建议 FFV1。

夹爪方向按以下顺序判定：明确表示开度或开口宽度的源名称、Parquet 低值/高值时刻对应的同侧视频帧、人工复核。同步视频结果置信度低于 `0.85`、夹爪被遮挡或两帧差异不足时不会自动采纳。

对于 DashScope compatible-mode endpoint，本地图片和其他结构化判断使用 Chat Completions 的非思考模式与 JSON Mode，并省略 `max_tokens`，避免结构化结果被截断；机器人拓扑联网查询使用 Responses API 的内置 `web_search`。其他 OpenAI-compatible endpoint 使用相同接口路径，服务不支持 Responses API 时记录为拓扑查询失败并继续兼容推理。

## Episode 采样策略

为控制大规模数据集的分析开销，规范化过程对每类数据最多读取 2 个 Episode：

- 文件按路径稳定排序；只有一个 Episode 时读取该文件，两个及以上时读取首个和末个。
- 普通机器字段仅对入选 Parquet 读取 schema、文件元数据和首个 batch；每个 batch 最多 `512` 行。
- 夹爪量程只投影 `action`、`observation.state` 中声明为 gripper 的目标维度，并扫描入选 Episode 的该维度；不会加载其他 Parquet 列。
- 视频探测、抽帧和 VLM 判断最多接收首、末两个 Episode；第一阶段证据充分时可提前结束，只读取首个视频。
- 每个待判定夹爪维度最多额外抽取两帧：接近 5% 与 95% 分位值的同步帧。优先使用 Parquet 时间戳，否则使用帧序号和 `fps` 换算。
- Parquet 缓存指纹只包含入选文件。未参与分析的中间 Episode 发生变化时不会重建缓存。
- 为确定末个 Episode，程序仍需遍历目录项；未入选文件不会进入 Parquet 内容读取、FFprobe、视频抽帧或 VLM 判断。

该策略以固定、可复现的有限样本换取更低的 I/O 成本。跨 Episode 一致性结论仅代表本次选中的首、末样本。

## 输出与复核

`normalize` 在每个数据集的 `meta/` 目录生成：

```text
<dataset>/meta/
├── info.json                    # 源元数据，保持不变
├── info_norm.json               # 规范化建议
├── info_norm_review.json        # 人工复核项与运行信息
└── .robometanorm_cache/         # 可删除的 Parquet 画像缓存
```

相机建议使用 `observation.images.cam_<位置>_<rgb|depth>` 格式。机器字段仅在声明、实测结构和语义证据全部满足要求时生成新名称。FPS、dtype、shape、数值内容和未覆盖字段保持原值。

`info_norm_review.json` 的 `robot_identity` 保存最终型号、选用来源、全部候选证据和冲突。相机复核项保存推理等级、规范化机器人拓扑、本地画面语义、候选项和未采纳原因；不会保存 API Key、Authorization 请求头或完整服务端错误响应。只有 `UNRESOLVED` 才保留源相机字段。

### 夹爪转换建议

标准目标字段为 `{left|right}_gripper_open`，目标范围为 `[0, 1]`，`0` 表示完全闭合，`1` 表示完全打开。

- 源数据已是 `[0, 1]`、数值增大表示打开且无需裁剪时，`info_norm.json` 可直接使用标准字段名。
- 源量程不同、方向相反或存在轻微越界时，`info_norm.json` 保留源字段名，`info_norm_review.json` 的 `gripper_transform_proposals` 给出 `source_closed`、`source_open`、归一化公式、裁剪策略和证据。
- 量程或方向无法确认时不生成转换建议，只生成定位到具体向量切片的人工复核项。
- 工具不会自动生成 `{dir}_gripper_open_scale`，也不会以样本最小值和最大值直接拉伸数据。

状态按严重程度排序为 `ERROR > BLOCKED > REVIEW > PASS`：

- `PASS`：检查通过，无待处理项。
- `REVIEW`：已生成可用建议，但存在需要人工确认的内容。
- `BLOCKED`：缺少继续判断所需的必要命名或结构证据。
- `ERROR`：元数据读取、输出或运行过程发生错误。

建议先运行 `scan`，处理 `BLOCKED` 项并确认 `REVIEW` 原因，再对生产数据运行 `normalize`。

## 验证

运行完整测试：

```bash
python3 -m unittest discover -s tests -v
```
