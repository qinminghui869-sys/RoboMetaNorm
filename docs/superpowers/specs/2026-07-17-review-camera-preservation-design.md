# REVIEW Annotation 相机映射保留设计

## 问题与根因

完整 annotation 编译要求相机映射和机器映射同时通过确认。任一机器映射失败时，编译器会切换到整体 fallback。现有 fallback 仅保留源键本身已符合规范命名的相机，因此会丢弃已经通过 VLM 与本地校验确认、但源键尚未规范化的相机映射，最终产生空的 `adapter.cameras`。

该问题发生在 annotation 编译阶段，与 YAML 序列化和写入无关。

## 目标

在完整 annotation 因机器映射失败而进入 REVIEW fallback 时，保留独立有效的相机映射，同时不放宽相机的信任边界。

## 行为边界

REVIEW YAML 的 `adapter.cameras` 仅允许包含以下映射：

1. 已由 VLM 映射且通过现有本地确认规则的相机；
2. 源键本身已符合规范相机命名的本地相机。

未经确认且源键不规范的相机继续省略。编译器不得根据字段名称自行推断相机位置、安装部位或模态。

## 设计

完整 PASS 路径保持不变。`compile_annotation` 在完整编译失败后，独立尝试编译相机映射：

1. 当 `profile`、`mapping` 和置信度阈值有效时，复用现有 `_compile_cameras`；
2. `_compile_cameras` 继续负责来源完整性、来源唯一性、置信度、歧义、相机组件和规范目标名校验；
3. 相机编译成功时，将结果传入 best-effort document；
4. 相机编译失败或不存在有效 VLM 结果时，best-effort document 仅保留源键本身已规范的相机；
5. 机器通道仍按现有本地结构规则生成，`review.required` 与问题投影逻辑保持不变。

该设计不从 `normalized_info` 的键顺序反推来源，也不向 annotation compiler 引入 normalization records，从而避免新增跨层耦合。

## 错误处理

- 已确认相机、机器映射失败：保留相机，机器通道进入 fallback，并要求复核；
- 相机映射集合不完整、低置信度、歧义或重复：整组 VLM 相机结果均不保留；
- 无 VLM 结果：仅保留本来已规范的相机源键；
- 非法相机源键：继续省略，不写入不安全路径。

## 测试

新增最小回归测试：

1. 有效相机映射与不完整机器映射组合时，document 为 REVIEW，且 `adapter.cameras` 保留规范目标到原始来源的映射；
2. 未确认的非规范原始相机仍不会写入 REVIEW YAML；
3. 现有完整 PASS 相机映射与本地规范相机 fallback 测试继续通过。

## 范围

仅修改：

- `src/robometanorm/annotation.py`
- `tests/unit/test_annotation.py`

无需修改 YAML schema、writer、pipeline、VLM 协议或 README。
