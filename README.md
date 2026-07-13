# RoboMetaNorm

面向机器人数据集的**规范化建议工具**。RoboMetaNorm 扫描数据集、检查转换前置条件，并依据《数据转换标准》生成可审核的元数据建议；它不直接改写原始数据。

## 核心原则

- **非破坏性**：不修改 `info.json`、Parquet、视频目录或媒体文件。
- **可审核**：每个数据集只在 `meta/` 中生成 `info_norm.json` 和 `info_norm_review.json`。
- **确定性优先**：相机与机器字段均由规则决定最终名称；VLM 只提供受限语义。
- **批处理隔离**：单个数据集异常不会中断其余数据集；全局结果仅输出到命令行，不落盘汇总文件。

## 当前能力

| 阶段 | 状态 | 能力 |
| --- | --- | --- |
| P0 | 已实现 | 递归发现、两种目录布局、前置条件检查、原子输出、命令行汇总 |
| P1 | 已实现 | 相机字段映射、固定 token 命名、媒体探测、冲突复核、可选 VLM 语义识别、RGB/Depth 编码建议 |
| P2 | 已实现 | Parquet 有限样本画像、action/state 去重、机器字段保守命名、机器复核与可选 VLM 语义 |
| P3 | 规划中 | 字段/目录物化、数值缩放和媒体转码；不属于当前版本 |

## 环境与安装

- Python 3.10+
- PyArrow 与 NumPy（随主依赖安装，用于 P2 Parquet 画像）
- 系统安装 FFprobe/FFmpeg（P1 媒体探测与抽帧）

```bash
python3 -m pip install -e .
```

Depth 预览需要可选依赖：

```bash
python3 -m pip install -e '.[depth]'
```

## 输入约定

支持以下两种布局，扫描入口均为 `meta/info.json`：

```text
# 平铺布局
collect_data/<dataset>/meta/info.json

# 任务分组布局
single_collect_data/<task>/<dataset>/meta/info.json
```

数据集目录至少包含 `data/` 或 `videos/`。扫描会忽略 `.git`、`.cache`、`.codex`、`.agents` 和 `__pycache__`。

## 快速开始

先只读扫描，确认状态与复核数量：

```bash
robometanorm scan --root /path/to/collect_data
```

确认后生成规范化建议：

```bash
robometanorm normalize --root /path/to/collect_data
```

可用 `--layout auto|flat|task_grouped` 限制目录布局，默认 `auto`。未安装命令行入口时，可改用：

```bash
PYTHONPATH=src python3 -m robometanorm scan --root /path/to/collect_data
```

### 可选：启用 VLM

默认不调用外部 VLM。配置后，未知或冲突相机可抽帧请求语义；机器字段仅提交结构、有限样本数值画像与声明名称，不提交目标字段名：

```bash
export OPENAI_API_KEY="<secret>"

robometanorm normalize --root /path/to/collect_data \
  --vlm-endpoint http://127.0.0.1:8002/v1 \
  --vlm-model qwen3-vl-30b-a3b-instruct-fp8
```

生产环境请通过环境变量或密钥管理系统注入密钥，禁止将密钥写入命令历史、配置文件或版本库。可按需设置 `--vlm-timeout-seconds`、`--vlm-max-retries`、`--vlm-retry-backoff-seconds`、`--vlm-max-tokens` 和 `--confidence-threshold`（P1 相机阈值）。P2 自动写入固定要求：置信度不低于 0.92、无歧义、无需转换、维度匹配；有量纲字段还必须在元数据中显式确认单位。

## 输出与状态

`normalize` 仅生成以下文件：

```text
<dataset>/meta/
├── info.json                 # 原始文件，保持不变
├── info_norm.json            # 规范化建议
└── info_norm_review.json     # 人工复核与运行信息
```

P1 中，确定性 RGB 相机建议使用 `observation.images.cam_<位置>_rgb` 与 `av1`；Depth 建议使用 `_depth` 与 `ffv1`。P2 对 `action`、`observation.state` 及其子字段只读取每个 Episode 的受限 Parquet 样本，验证维度、跨 Episode 布局、父子连续切片与 action/state 一致性后才建议名称。P2 不改变数值、不推断未知单位、不将 wrist 默认视为 EEF，也不处理四元数转欧拉角、夹爪量程/方向、骨架与关键点；这些内容保留源名并写入 `machine_review_items`。FPS、dtype、shape 和未涉及字段保持原值。

状态优先级：`ERROR > BLOCKED > REVIEW > PASS`。

- `PASS`：检查通过，无待处理项。
- `REVIEW`：可继续使用，但需要人工确认。
- `BLOCKED`：缺少 RGB、action、observation、主摄像头或 URDF 等关键前置条件。
- `ERROR`：元数据读取、输出或运行时发生错误。

## 验证

```bash
python3 -m unittest discover -s tests -v
```

建议在生产批处理前先对小规模副本运行 `scan`，审阅 `REVIEW`/`BLOCKED` 的原因后再执行 `normalize`。
