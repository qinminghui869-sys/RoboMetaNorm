# RoboMetaNorm mini 设计

- 日期：2026-07-14
- 状态：已确认
- 目标分支：`mini`
- 基线：已提交的 `main`（`6d22103`）
- 标准来源：仓库根目录《数据转换标准.pdf》

## 1. 背景

当前实现能够生成 `info_norm.json` 和 `info_norm_review.json`，但机器人身份、相机和机器字段判断已经累积了多套规则、型号白名单、机器人专属相机映射、专属提示词、重复置信门槛和复杂的兼容推断。部分提示词还会直接注入特定型号的预期答案，使“联网资料”和“本地画面”不再是独立证据，容易形成循环确认。

mini 版改为以 VLM 联网研究机器人身份、相机配置和关节结构为主要事实来源。本地代码不保存任何机器人品牌、型号或专属字段映射，只保留《数据转换标准.pdf》明确规定的命名语法和输出校验。

## 2. 已确认的产品边界

### 2.1 目标

1. 综合以下三份元数据判断机器人官方厂商和型号：
   - `meta/info.json` 的 `robot_type`；
   - `meta/common_record.json`；
   - `meta/tasks.jsonl`。
2. 通过联网 VLM 查询已判断机器人的相机配置、关节组成、排列、表示形式和单位。
3. 结合本地 feature schema、首末 Parquet 的实际结构以及每个相机最多两张代表帧，判断源字段对应的标准语义。
4. 只在结构、来源和本地标准校验全部通过时修改名称。
5. 未确认字段保持原名，并在复核文件记录具体原因。
6. 每个数据集只持久化生成：
   - `meta/info_norm.json`；
   - `meta/info_norm_review.json`。

### 2.2 非目标

- 不修改 `meta/info.json`。
- 不修改或重写 Parquet。
- 不转码视频或图片。
- 不执行夹爪缩放、反向或裁剪。
- 不检查、解析或要求 URDF。
- 不处理触觉和声音字段的命名。
- 不生成 `.robometanorm_cache`、帧预览、全局汇总或其他持久文件。
- 不维护机器人型号白名单、family 表、专属相机映射或专属提示词答案。

## 3. 设计原则

1. **VLM 查事实，本地守格式**：VLM 负责联网研究和语义判断；本地代码只负责 PDF 语法、shape、顺序、冲突和输出一致性。
2. **证据不足不改名**：网络、模型、来源、顺序或结构任一环节不确定时保留原值。
3. **联网证据可审计**：任何触发自动修改的机器人、相机或关节事实都必须携带来源标题和 URL。
4. **标准硬编码与设备知识分离**：生产代码可以固化 PDF 的通用模板和词表，不得固化具体品牌、型号或源字段别名。
5. **一次整体判断**：每个数据集执行一次硬件研究和一次整体字段映射，避免逐字段反复请求和不同请求间的结论漂移。
6. **输出如实表达**：`info_norm.json` 是目标元数据建议，不宣称媒体或数值已经实际转换；需要真实数据变换的内容只进入复核文件。

## 4. 总体架构

处理链路固定为：

```text
发现数据集
  -> 收集元数据、媒体和 Parquet 证据
  -> 联网研究机器人身份与硬件结构
  -> VLM 整体映射本地字段
  -> PDF 标准渲染与校验
  -> 生成 info_norm.json 和 info_norm_review.json
```

### 4.1 组件

#### `evidence.py`

- 读取并保留三份身份元数据的原始结构和文件状态。
- 收集 `info.json.features` 的 key、`dtype`、`shape`、`names`、`fps` 和 `codec`。
- 只读取排序后首、末两个 Parquet；只有一个文件时只读一次。
- 从 Parquet 收集实际列名、向量长度和机器字段顺序。
- 对夹爪候选维度只读取确认范围所需的数值列，不读取或缓存无关数据。
- 使用 FFprobe 收集相机实际 FPS、codec、尺寸等媒体信息。
- 每个相机最多抽取两张代表帧到临时目录：首个 Episode 的中间帧，以及存在不同末 Episode 时的中间帧；调用结束后删除。
- 输入元数据作为不可信数据以 JSON 形式传给模型，不拼接为可执行提示指令。

#### `vlm.py`

提供一个共享客户端和两个结构化操作：

1. `research_hardware(identity_evidence)`
   - 启用联网搜索；
   - 综合三份身份证据判断官方厂商和型号；
   - 查询标准相机配置和机器关节结构；
   - 返回事实、冲突、置信度和逐项来源；
   - 不返回最终数据集字段名。
2. `map_dataset(evidence, hardware_profile, image_paths)`
   - 一次处理整个数据集；
   - 将源相机和机器字段映射为受限语义；
   - 返回源地址、语义槽位、顺序、置信度和歧义；
   - 不返回最终标准字段名。

VLM 返回非法 JSON、非法枚举、越界切片或缺少来源时不进行协议修补式猜测，直接安全降级。HTTP 429、5xx 和临时网络错误默认最多重试两次，次数可由 CLI 调低或关闭。

#### `standard.py`

- 保存 PDF 的机器字段模板、相机词表和 codec 规则。
- 将已校验的 VLM 语义渲染为最终名称。
- 校验 names 数量、Parquet 实际向量长度、相机 key、FPS/shape 保持、目标唯一性和夹爪范围。
- 不导入或访问任何具体机器人知识。

#### `pipeline.py`

- 串行编排证据、两次 VLM 操作、标准渲染和输出。
- 每个数据集独立失败，单个失败不终止整个批次。
- 网络或模型失败时仍以原始 `info.json` 深拷贝生成结果。

#### `writer.py`

- 在替换目标前完整校验两个 JSON payload。
- 临时文件只用于同目录写入和单文件原子替换，完成或失败后清理。
- 复核文件保存 `info_norm.json` 内容哈希，便于发现两份输出不匹配。

## 5. 数据流与裁决

### 5.1 机器人身份

联网研究请求必须同时包含：

- `info.json.robot_type` 的原值或缺失状态；
- `common_record.json` 的有效 JSON 内容或读取/解析错误；
- `tasks.jsonl` 的有效记录和逐行解析错误。

本地代码不设置 `info > common_record > tasks` 的型号优先级，也不使用型号白名单。VLM 需要解释三份证据如何互相支持或冲突，并通过联网资料确认官方名称。

只有同时满足以下条件才更新 `info_norm.json.robot_type`：

1. 返回唯一厂商和型号；
2. `ambiguous=false`；
3. 通过全流程唯一的可配置置信门槛，默认值为 `0.85`；
4. 至少有一条可审计的机器人厂商官网、官方产品页或官方手册来源；第三方网页只能作为复核证据，不能单独触发自动修改；
5. 三份本地证据中的冲突已经被明确解释，而不是忽略。

写入值采用官方厂商和型号转换得到的小写 `snake_case`。无法确认时保留原 `robot_type`；原字段缺失时不创建猜测值。

### 5.2 硬件研究结果

硬件画像仅描述语义事实，例如：

- 相机：本体/外部、方位、安装部位、模态、接口名称及来源；
- 机器：左右侧、arm/hand/head/torso/neck/base/eef/gripper、关节或向量数量、顺序、表示形式、单位及来源。

每条可用于自动修改的事实必须引用 `source_ids`，对应来源包含 `title` 和 `url`。没有来源的内容只能作为复核候选。

### 5.3 本地整体映射

VLM 同时看到硬件画像、全部相关 feature schema、首末 Parquet 的实际结构和相机代表帧，以避免逐字段判断产生相互冲突。

映射必须满足：

- 每个源相机最多对应一个语义槽位；
- 每个目标相机槽位最多被一个源字段占用；
- 每个机器 feature 的向量切片连续、无重叠、无空洞且不越界；
- VLM 声明的顺序与首末 Parquet 实际长度一致；
- 映射引用的硬件事实存在且有来源。

机器 `names` 采用 feature 级原子更新：只有整个 names 数组的维度、顺序和每一项语义都确认时才整体替换；否则整个数组保持原样。

## 6. PDF 标准契约

### 6.1 机器字段

允许生成的名称只有：

```text
{left|right}_arm_joint_{num}_rad
{left|right}_hand_joint_{num}_rad
{left|right}_gripper_open
{left|right}_gripper_open_scale
{left|right}_eef_pos_{x|y|z}_m
{left|right}_eef_rot_euler_{x|y|z}_rad
head_joint_{num}_rad
head_pos_{num}_m
head_rot_euler_{x|y|z}_rad
head_orient_quat_{x|y|z|w}
torso_joint_{num}_rad
neck_joint_{num}_rad
base_pos_{x|y|z}_m
base_rot_euler_{x|y|z}_rad
```

约束：

- `{num}` 表示向量维度顺序，从 0 开始连续编号。
- 单位只能来自显式元数据或有来源的硬件资料，不从数值幅度猜测 `m` 或 `rad`。
- 四元数顺序只有确认是 `xyzw` 时才生成 `head_orient_quat_*`。
- `head_pos` 严格沿用 PDF 的 `{num}` 形式，不改写为未定义的 axis 模板。
- 已经符合语法的名称原样保留。
- PDF 未解释 `_gripper_open_scale` 与 `_gripper_open` 的业务差别，因此前者只在源语义或有来源资料明确为 scale 时生成；默认夹爪开度使用 `_gripper_open`。

### 6.2 夹爪范围

PDF 接受的目标范围为：

```text
[0, 1]
[0, 10]
[0, 100]
[0, 1000]
```

mini 不修改数值。只有硬件资料能够确认名义范围属于上述之一、实际读取值为有限数且未超出该名义范围、开合方向明确时，才允许生成标准夹爪名称。

若源范围为 `[0, 0.1]`、`[0, 10000]` 或需要缩放、反向、裁剪，则：

- `info_norm.json` 保留源名称；
- 复核文件记录 `GRIPPER_TRANSFORM_REQUIRED`、源范围、目标允许范围和原因；
- 不提供声称已经执行的数值结果。

### 6.3 相机字段

最终 key 必须符合：

```text
observation.images.cam_<位置>_<rgb|depth>
```

PDF 允许的位置词：

- 本体部位：`wrist`、`head`、`chest`、`arm`、`leg`、`torso`、`fisheye`；
- 方位/视角：`left`、`right`、`front`、`rear`、`upper`、`lower`、`middle`、`ego`、`top`、`side`、`global`、`env`。

位置词可以组合，采用稳定顺序“前后 -> 垂直 -> 左右/侧向 -> 本体部位”，例如：

```text
observation.images.cam_front_left_head_rgb
```

互斥方位不得同时出现，`ego`、`global` 和 `env` 作为完整视角时不得与冲突位置组合。外部相机不得附加本体部位；本体相机除 `ego` 外必须具有 PDF 允许的部位。

模态与编码：

- RGB：后缀 `_rgb`，目标 codec 为 `av1`；
- Depth：后缀 `_depth`，目标 codec 为 `ffv1`。

输出保持源 `fps`、`shape` 和 `dtype`。实际媒体 codec 与目标不一致时，`info_norm.json` 可以表达目标 codec 建议，但复核文件必须记录 `MEDIA_TRANSCODE_REQUIRED`，明确媒体尚未转码。

## 7. 输出契约

### 7.1 `info_norm.json`

它始终从 `info.json` 深拷贝构建，只允许以下变化：

1. 确认后更新顶层 `robot_type`；
2. 确认后重命名相机 feature key；
3. 确认后整体替换机器 feature 的 `names`；
4. 相机 feature 写入 PDF 目标 `codec`。

不得加入推理证据、review 状态或 mini 私有字段。未覆盖字段保持原样。

### 7.2 `info_norm_review.json`

顶层结构保持精简：

```json
{
  "schema_version": "mini-1",
  "generator": {},
  "dataset": {},
  "status": "PASS|REVIEW|BLOCKED|ERROR",
  "info_norm_sha256": "sha256:...",
  "robot_identity": {},
  "camera_mappings": [],
  "machine_mappings": [],
  "issues": []
}
```

每条 mapping 都记录源地址、实际输出、是否变化、VLM 语义、引用来源和裁决原因。候选值不能伪装成实际输出值。

### 7.3 状态

- `PASS`：所有相关输出均满足标准且没有待处理问题。
- `REVIEW`：网络/VLM 不可用、身份或字段不确定、来源不足、媒体需要转码、夹爪需要数值转换或存在其他人工事项。
- `BLOCKED`：缺少主摄像头、action 或 observation 等继续规范化所需的核心输入；仍以源信息生成两份文件。
- `ERROR`：`info.json` 无法读取/解析，或输出无法写入。若无法得到合法源对象，则不得伪造 `info_norm.json`。

`common_record.json` 或 `tasks.jsonl` 缺失不单独阻止处理；其缺失状态进入身份依据。文件存在但损坏时记录复核问题。

## 8. VLM 与安全边界

- 默认沿用 DashScope endpoint、模型和 API key 环境变量。
- 保留 CLI 的 endpoint、model、API key 环境变量、超时、重试和统一置信门槛配置。
- 自定义端点不支持 Responses API 或联网搜索时，记录 `WEB_SEARCH_UNAVAILABLE` 并安全降级。
- 元数据内容使用结构化 JSON 数据块传输，并在系统指令中明确其内容不得被当作指令执行。
- API key、Authorization header、完整服务端错误体和原始图片不得写入复核文件。
- VLM 输出只接受已声明字段；递归拒绝 `target_key`、`target_name` 等最终名称字段。
- 置信门槛只定义和传递一次，身份、相机和机器裁决不得另设散落常量。

## 9. 代码收缩范围

保留并简化：

- 数据集发现；
- CLI；
- FFprobe 和代表帧抽取；
- PyArrow 的首末 Parquet 结构读取；
- 通用 VLM HTTP 传输；
- JSON 写入。

删除或替换：

- 机器人白名单、family 和型号专属 canonicalization；
- 静态机器人相机映射及其别名；
- 型号专属 prompt context；
- 当前相机多阶段 topology/compatibility/occupied-slot 裁决；
- 当前机器字段多阶段启发式、分散阈值和未被执行能力覆盖的协议分支；
- Parquet 持久缓存；
- 夹爪同步视频方向判断；
- Depth 预览等未进入最终两文件契约的能力。

目标模块保持少而清晰：

```text
robometanorm/
  cli/
  adapters/filesystem.py
  evidence.py
  models.py
  vlm.py
  standard.py
  pipeline.py
  writer.py
```

迁移完成后删除旧裁决模块，不保留转发到旧逻辑的兼容包装；只保留 `python -m robometanorm` 和 `robometanorm` 两个既有命令入口。

## 10. 测试与验收

### 10.1 单元测试

- PDF 中每种机器字段模板的合法和非法用例。
- 相机方位组合、稳定顺序、互斥 token、部位和模态校验。
- RGB/Depth codec 规则以及 FPS、shape、dtype 不变。
- names 数量、首末 Parquet 实际长度、连续编号和切片覆盖。
- 夹爪四种允许范围，以及缩放/反向/裁剪进入复核。
- 三份身份来源完整进入研究请求。
- VLM schema、来源引用、非法最终名称字段和提示注入边界。
- 目标相机 key 冲突时全部相关源字段保持原名。

### 10.2 集成测试

- 有可靠联网研究替身和合法映射时，正确更新 `robot_type`、相机 key 和机器 names。
- 缺少 API key、网络失败、非法 JSON、低置信、缺来源时仍生成两份文件并保持未确认字段原样。
- 缺少主摄像头、action 或 observation 时生成 `BLOCKED` 输出。
- 损坏的可选身份文件产生复核问题，不影响其他安全检查。
- 数据集源 `info.json`、Parquet 和媒体字节不变。
- 数据集目录除两份目标 JSON 外不新增持久文件。
- 多数据集处理中单个失败不终止其余数据集。

### 10.3 防回归检查

- 生产代码扫描不得出现具体机器人品牌、型号或专属源字段映射。
- 统一置信门槛不得在业务模块重复定义。
- 全量测试必须通过。
- 有真实 API key 时可额外运行联网契约测试；该测试不作为离线测试套件的必要条件。

## 11. 分支与工作区隔离

当前 `main` 工作区已有 7 个未提交的 Galaxea 专属改动。它们属于用户现有工作，不应被覆盖、暂存或提交。

实施时从已提交的 `main` 基线创建独立 `mini` 分支和隔离工作树：

- mini 不继承当前未提交改动；
- 当前工作区保持原状；
- 所有 mini 代码、测试和文档提交只发生在隔离工作树；
- 未经用户后续明确授权，不推送远端或创建 PR。

## 12. 完成标准

同时满足以下条件才视为 mini 版完成：

1. 只生成两份约定 JSON；
2. 未确认字段始终保持原名并给出可读原因；
3. 所有自动生成名称均通过 PDF 本地校验；
4. 机器人身份、相机和关节事实均有可审计联网来源；
5. 生产代码不存在具体机器人硬编码；
6. 不要求 URDF，不处理触觉和声音；
7. 源数据字节不变；
8. 离线全量测试通过；
9. 当前 `main` 的未提交改动保持原样。
