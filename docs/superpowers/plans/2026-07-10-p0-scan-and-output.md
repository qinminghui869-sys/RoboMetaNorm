# RoboMetaNorm P0 扫描与基础输出 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现不修改原始机器人数据的 P0 命令行工具：发现数据集、检查转换前置条件，并在 `meta/` 中生成基础建议与人工复核文件。

**Architecture:** 使用 Python 标准库分离领域对象、文件系统适配器、前置条件检查、原子 JSON 写入和 CLI。`scan` 只汇总内存结果；`normalize` 在通过 JSON 结构校验后写入原始 `info.json` 的深拷贝和 review 文件，P1/P2 规则不进入本阶段。

**Tech Stack:** Python 3.10+、标准库 `argparse`/`dataclasses`/`json`/`pathlib`/`tempfile`、`unittest`。

---

## File structure

- `pyproject.toml`：包元数据和命令行入口。
- `src/robometanorm/domain/models.py`：P0 的数据集、复核项和扫描结果不可变领域对象。
- `src/robometanorm/adapters/filesystem.py`：递归发现、目录布局识别及不安全路径排除。
- `src/robometanorm/application/preconditions.py`：基于元数据和实际目录的前置条件检查与状态归并。
- `src/robometanorm/application/pipeline.py`：扫描、读取、检查、输出的编排。
- `src/robometanorm/writers/json_writer.py`：JSON 结构校验和同目录临时文件原子替换。
- `src/robometanorm/cli/main.py`：`scan` 与 `normalize` 子命令和表格输出。
- `tests/unit/`：发现、检查和写入行为测试。
- `tests/integration/test_cli.py`：两条 CLI 命令的端到端测试。
- `README.md`：安装、命令和 P0 边界说明。

### Task 1: 建立包与数据集发现

**Files:**
- Create: `pyproject.toml`
- Create: `src/robometanorm/__init__.py`
- Create: `src/robometanorm/domain/models.py`
- Create: `src/robometanorm/adapters/filesystem.py`
- Test: `tests/unit/test_discovery.py`

- [ ] **Step 1: Write the failing test**：构造直接数据集和 `task/dataset` 数据集，断言递归发现得到正确的 `DatasetCandidate`，并忽略 `.git`。
- [ ] **Step 2: Run test to verify it fails**：运行 `python3 -m unittest tests.unit.test_discovery -v`；预期因包不存在失败。
- [ ] **Step 3: Write minimal implementation**：定义候选对象、`auto|flat|task_grouped` 布局和 `**/meta/info.json` 发现，要求数据集根目录至少有 `data/` 或 `videos/`。
- [ ] **Step 4: Run test to verify it passes**：运行相同命令；预期全部通过。

### Task 2: 实现前置条件检查

**Files:**
- Create: `src/robometanorm/application/preconditions.py`
- Test: `tests/unit/test_preconditions.py`

- [ ] **Step 1: Write the failing test**：包含 action、observation、视频特征、视频文件、URDF、落盘程序和 LeRobot 脚本的数据集应为 `PASS`；删除 action 后应产生 `missing_action` 的阻塞复核项。
- [ ] **Step 2: Run test to verify it fails**：运行 `python3 -m unittest tests.unit.test_preconditions -v`；预期导入失败。
- [ ] **Step 3: Write minimal implementation**：检查 RGB、action、observation、视觉特征、URDF、落盘程序和转换脚本；按 `ERROR > BLOCKED > REVIEW > PASS` 汇总状态。
- [ ] **Step 4: Run test to verify it passes**：运行相同命令；预期全部通过。

### Task 3: 实现基础输出和扫描编排

**Files:**
- Create: `src/robometanorm/writers/json_writer.py`
- Create: `src/robometanorm/application/pipeline.py`
- Test: `tests/unit/test_json_writer.py`

- [ ] **Step 1: Write the failing test**：断言 `info_norm.json` 是原 `info.json` 的值相等深拷贝，`info_norm_review.json` 含运行信息和复核项，并且没有残留临时文件。
- [ ] **Step 2: Run test to verify it fails**：运行 `python3 -m unittest tests.unit.test_json_writer -v`；预期导入失败。
- [ ] **Step 3: Write minimal implementation**：在写入前验证两份可 JSON 序列化的对象，在 `meta/` 写入两个临时文件、刷盘并用 `os.replace` 原子替换；扫描失败只记录 `ERROR`，不中断其他数据集。
- [ ] **Step 4: Run test to verify it passes**：运行相同命令；预期全部通过。

### Task 4: CLI、README 与端到端验证

**Files:**
- Create: `src/robometanorm/__main__.py`
- Create: `src/robometanorm/cli/main.py`
- Create: `README.md`
- Test: `tests/integration/test_cli.py`

- [ ] **Step 1: Write the failing test**：`scan --root` 必须打印汇总且不写文件；`normalize --root` 必须打印汇总并生成两个文件。
- [ ] **Step 2: Run test to verify it fails**：运行 `python3 -m unittest tests.integration.test_cli -v`；预期导入失败。
- [ ] **Step 3: Write minimal implementation**：提供 `scan`、`normalize`、`--root`、`--layout`，用标准库生成表格；README 只说明 P0 行为、安装、运行和不修改原始数据的约束。
- [ ] **Step 4: Run test to verify it passes**：运行 `python3 -m unittest tests.integration.test_cli -v`；预期全部通过。

### Task 5: 完整回归验证

**Files:**
- Verify: `tests/`

- [ ] **Step 1: Run complete test suite**：运行 `python3 -m unittest discover -s tests -v`；预期所有测试通过。
- [ ] **Step 2: Run real-data smoke scan**：运行 `PYTHONPATH=src python3 -m robometanorm scan --root collect_data`；预期输出 10 个数据集的状态表，且不产生任何规范化文件。
- [ ] **Step 3: Check source tree**：运行 `git status --short`，确认仅新增程序、测试、README 和规划文档，不改动输入 PDF、`collect_data/` 的原始文件或 Parquet。
