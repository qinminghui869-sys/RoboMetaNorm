# RoboMetaNorm

RoboMetaNorm 扫描机器人数据集，并根据《数据转换标准》生成规范化建议。当前完成 P0 和 P1：扫描、基础输出与相机规范化。

## P0 能力

- 递归发现 `**/meta/info.json`，兼容平铺和任务分组目录；
- 读取 `info.json`，检查 RGB、action、observation、视觉特征、URDF、采集程序和 LeRobot 转换脚本；
- `scan` 只输出命令行汇总表；
- `normalize` 在每个数据集的 `meta/` 中原子写入：
  - `info_norm.json`：原始 `info.json` 的深拷贝；P1 会将可确定的相机字段改为标准名，并建议 RGB 使用 AV1、Depth 使用 FFV1；
  - `info_norm_review.json`：状态、复核项和运行信息；
- 不修改原始 `info.json`、Parquet、视频目录或媒体文件，也不生成全局汇总文件。

P1 使用精确字典和固定 token 顺序（前后→垂直→左右→本体部位→模态）处理相机名。未知、低置信度或目标名冲突的字段保留原名，并写入 `camera_review_items`。P2 的机器字段与夹爪量程建议尚未实现。

## 安装

主流程依赖 Python 3.10+ 标准库和系统中的 FFprobe/FFmpeg：

```bash
python3 -m pip install -e .
```

若要生成 Depth 灰度和伪彩预览，请安装可选依赖：

```bash
python3 -m pip install -e '.[depth]'
```

也可以不安装，命令前加 `PYTHONPATH=src`。

## 使用

平铺目录示例：

```text
collect_data/
└── dataset_001/
    ├── meta/info.json
    ├── data/
    └── videos/
```

任务分组目录示例：

```text
single_collect_data/
└── task_a/
    └── task_a_001/
        ├── meta/info.json
        ├── data/
        └── videos/
```

只扫描，不写文件：

```bash
robometanorm scan --root /path/to/collect_data
```

生成 P0 基础输出：

```bash
robometanorm normalize --root /path/to/collect_data
```

默认不调用 VLM。若要为未知相机启用 OpenAI-compatible VLM 语义识别，可显式提供服务地址与模型；VLM 仅返回语义，最终字段名仍由规则生成：

```bash
export OPENAI_API_KEY="..."
robometanorm normalize --root /path/to/collect_data \
  --vlm-endpoint http://127.0.0.1:8002/v1 \
  --vlm-model qwen3-vl-30b-a3b-instruct-fp8 \
  --vlm-timeout-seconds 120 \
  --vlm-max-retries 2 \
  --vlm-retry-backoff-seconds 1.0 \
  --vlm-max-tokens 1024
```

客户端会按“显式密钥 → `--vlm-api-key-env` 指定变量 → `DASHSCOPE_API_KEY` → `OPENAI_API_KEY`”寻找密钥；对于 429、5xx 和网络异常，会按指数退避重试。其他 HTTP 错误或不合规 JSON 不会重试，而是进入人工复核。

可用 `--layout auto|flat|task_grouped` 限制输入布局；默认 `auto`。

状态优先级为 `ERROR > BLOCKED > REVIEW > PASS`。缺少 RGB、action、observation、主摄像头或 URDF 会标记为 `BLOCKED`；缺少采集程序或已有 LeRobot 脚本会标记为 `REVIEW`。

## 测试

```bash
python3 -m unittest discover -s tests -v
```
