# RoboMetaNorm P1 相机规范化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在规范建议模式下将可确定的 `observation.images.*` 字段写入 P1 标准相机名，并把未知、冲突和媒体不一致情况写入人工复核文件。

**Architecture:** 新增独立 `camera/` 领域模块，先使用全局精确映射和确定性 token 解析；只有字段未知或冲突且用户提供 VLM 配置时才抽帧并请求语义分类。应用层把相机结果合并到深拷贝的 `info_norm.json`，原始信息、媒体和 Parquet 均不修改。

**Tech Stack:** Python 3.10+ 标准库、FFprobe/FFmpeg 子进程、可选 OpenAI-compatible HTTP、已安装时使用 NumPy/OpenCV 生成 Depth 预览、`unittest`。

---

### Task 1: 相机命名规则与冲突检测

**Files:**
- Create: `src/robometanorm/camera/models.py`
- Create: `src/robometanorm/camera/mapping_registry.py`
- Create: `src/robometanorm/camera/name_parser.py`
- Create: `src/robometanorm/camera/name_builder.py`
- Create: `src/robometanorm/camera/collision_checker.py`
- Test: `tests/unit/test_camera_naming.py`

- [ ] **Step 1: Write the failing test**：精确映射 `image_left`、深度 `image_top_depth`、正则字段 `image_front_left_head` 分别生成标准字段；`image_left` 与 `camera_left` 重名时两者均不写入目标名。
- [ ] **Step 2: Run test to verify it fails**：运行 `python3 -m unittest tests.unit.test_camera_naming -v`；预期因 `robometanorm.camera` 不存在失败。
- [ ] **Step 3: Write minimal implementation**：定义标准词表与前后→垂直→左右→本体部位→模态的构造器；精确字典覆盖方案列出的字段；冲突检测返回待复核源字段。
- [ ] **Step 4: Run test to verify it passes**：运行相同命令；预期全部通过。

### Task 2: 媒体探测、抽帧和 VLM 语义接口

**Files:**
- Create: `src/robometanorm/camera/media_probe.py`
- Create: `src/robometanorm/camera/frame_sampler.py`
- Create: `src/robometanorm/camera/depth_preview.py`
- Create: `src/robometanorm/camera/prompt_builder.py`
- Create: `src/robometanorm/camera/vlm_classifier.py`
- Test: `tests/unit/test_camera_media.py`

- [ ] **Step 1: Write the failing test**：断言第一阶段抽样比例为 `10/50/90`，单 episode 第二阶段为 `10/30/50/70/90`，多 episode 第二阶段为 `20/50/80`；Prompt 不包含 `target_key`，且语义校验拒绝不在词表内的 token。
- [ ] **Step 2: Run test to verify it fails**：运行 `python3 -m unittest tests.unit.test_camera_media -v`；预期因模块不存在失败。
- [ ] **Step 3: Write minimal implementation**：FFprobe 提取 codec/FPS/分辨率/帧数，FFmpeg 在临时目录抽帧；Depth 以有效值 2%–98% 分位数生成灰度与伪彩预览；VLM 仅接收并输出语义字段，默认禁用。
- [ ] **Step 4: Run test to verify it passes**：运行相同命令；预期全部通过。

### Task 3: 将 P1 决策写入两个输出文件

**Files:**
- Create: `src/robometanorm/camera/normalizer.py`
- Modify: `src/robometanorm/domain/models.py`
- Modify: `src/robometanorm/application/pipeline.py`
- Modify: `src/robometanorm/writers/json_writer.py`
- Test: `tests/unit/test_camera_normalizer.py`
- Test: `tests/unit/test_json_writer.py`

- [ ] **Step 1: Write the failing test**：确定性相机字段在 `info_norm.json` 以目标键和目标 codec 保存，FPS/dtype/shape 保持；未知和重名字段保留源键，并在 `camera_review_items` 中含候选、原因码和待定人工决定。
- [ ] **Step 2: Run test to verify it fails**：运行 `python3 -m unittest tests.unit.test_camera_normalizer tests.unit.test_json_writer -v`；预期缺少 P1 normalizer 或 review 字段断言失败。
- [ ] **Step 3: Write minimal implementation**：按决策优先级更新深拷贝的 features；RGB 建议 `av1`、Depth 建议 `ffv1`；将相机复核项参与 `review_required` 和状态汇总；继续使用同目录临时文件加 `os.replace`。
- [ ] **Step 4: Run test to verify it passes**：运行相同命令；预期全部通过。

### Task 4: CLI 配置、文档和完整验证

**Files:**
- Modify: `src/robometanorm/cli/main.py`
- Modify: `README.md`
- Test: `tests/integration/test_cli.py`

- [ ] **Step 1: Write the failing test**：`normalize` 处理 legacy 相机字段时生成 P1 目标名和 codec；默认不调用 VLM，提供 endpoint/model 时才创建 OpenAI-compatible 分类器。
- [ ] **Step 2: Run test to verify it fails**：运行 `python3 -m unittest tests.integration.test_cli -v`；预期缺少参数或 P1 输出断言失败。
- [ ] **Step 3: Write minimal implementation**：添加 `--vlm-endpoint`、`--vlm-model`、`--vlm-api-key-env`、`--confidence-threshold`；README 说明 P1 行为和 VLM 默认禁用。
- [ ] **Step 4: Run complete verification**：运行 `python3 -m unittest discover -s tests -v`、`PYTHONPATH=src python3 -m robometanorm scan --root collect_data` 与 `PYTHONPATH=src python3 -m robometanorm normalize --root <temporary fixture root>`；预期测试通过、真实扫描不写文件、临时夹具只生成两个 JSON。
