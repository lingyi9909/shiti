# Question Builder V1 设计规范

- 日期：2026-08-29
- 状态：已批准（2026-08-29）
- 运行环境：macOS 本地单机
- 技术栈：Python 3.12+
- 输入：批量 `.docx`
- 正式输出：`questions.jsonl + image/`

## 1. 背景与目标

本项目用于将来源、排版、语言和答案组织方式高度不统一的 K12 Word 试题资料，自动转换为统一结构的 JSONL 试题数据。

V1 不把问题定义为“让大模型把 Word 转 JSON”，而定义为：

> 一套带证据链、可拒绝、可追踪、可重跑、可更换 OCR/模型供应商的 K12 试题结构化生产流水线。

核心质量原则：

1. 正式结果准确率优先，宁可降低召回率，也不允许低置信度题目或答案错配进入正式数据。
2. 大模型不得自行解题、补答案或纠正原始答案。
3. 能确定性解析的内容不交给模型猜测。
4. OCR/LLM 结果只能形成候选或语义判断，不能绕过程序化质量门禁直接写入正式 JSONL。
5. 所有关键结果必须可追溯到原始 Word、原始内容块和识别/匹配证据。
6. 任何关键内容解析失败不得静默丢失；必须显式标记并触发拒绝策略。

## 2. V1 范围

### 2.1 输入范围

V1 只接受 `.docx` 文件，不接受 `.doc`、PDF、Excel、网页或独立图片目录。

一次任务可输入 10～100 个 Word 文件、几千道题。文件可能属于：

- 单套试卷及答案；
- 多套可大致区分的试卷；
- 多年级、多学科、多试卷混合文件；
- 题目与答案位于同一文件；
- 题目与答案位于不同文件；
- 一个答案文件覆盖多个题目文件；
- 文件名、题号、排版和答案组织方式不统一。

DOCX 内部必须支持：

- 普通文本；
- Word 自动编号；
- 图片与图文混排；
- 整题截图；
- Word 原生表格；
- 图片表格；
- OMML 公式；
- MathType/OLE 公式；
- 公式截图；
- 文本框；
- 页眉页脚；
- 多语言及混合语言。

### 2.2 正式输出

```text
result/
├── questions.jsonl
└── image/
    ├── <sha256-prefix>.png
    ├── <sha256-prefix>.jpg
    └── ...
```

内部运行产物额外包括：

```text
result/
├── rejected.jsonl
├── run_report.json
├── logs/
└── workspace/
    ├── documents/
    ├── recognition/
    ├── candidates/
    ├── answers/
    └── matches/
```

正式对外交付仅包含 `questions.jsonl` 与 `image/`。

### 2.3 V1 明确不做

- PDF / `.doc` 输入；
- Web UI；
- 分布式集群；
- 人工审核后台；
- AI 自主解题；
- AI 纠正原始答案；
- 试题生成；
- 知识图谱；
- 向量数据库平台；
- 大数据基础设施。

## 3. 规模与演进目标

- V1：单次 10～100 个 Word、几千道题；
- 稳定期：单次数百～上千 Word、数万道题；
- 最终：长期批量生产、累计百万级试题。

因此 V1 采用本地单机实现，但从第一版就具备：

- Provider 抽象；
- 幂等；
- 缓存；
- 断点续跑；
- 任务状态持久化；
- 可配置并发；
- 可追溯证据链；
- 可迁移服务器的存储边界。

不在 V1 提前引入 Kafka、Kubernetes、Flink 等重型基础设施。

## 4. 总体架构

```text
DOCX Files
   ↓
1. DOCX Ingest
   ↓
2. Document Parser
   ↓
Document IR
   ↓
3. Recognition Router
   ├─ Text OCR
   ├─ Formula OCR
   ├─ Table Recognition
   └─ Vision / Multimodal
   ↓
4. Document Understanding
   ├─ 文件分类
   ├─ 文档元信息提取
   └─ 试卷聚类
   ↓
5. Question Builder
   ↓
Question IR
   ↓
6. Answer Extractor
   ↓
Answer IR
   ↓
7. Answer Matcher + Independent Verifier
   ↓
Matched Question
   ↓
8. Metadata Normalizer
   ↓
9. Multi-stage Quality Gates
   ├─ PASS → Final Question
   └─ REJECT → rejected.jsonl
   ↓
10. JSONL Export
```

架构硬原则：

- Parser 只回答“Word 中有什么”，不负责拆题；
- Question Builder 只负责题目结构，不负责答案生成；
- Answer Extractor 只提取原始答案，不建立归属；
- Answer Matcher 负责归属判断，但不得修改答案内容；
- Metadata Normalizer 可推断辅助元数据，不得生成答案；
- Exporter 只做最终 Schema 转换，不包含业务推断。

## 5. 数据模型

系统采用三层核心数据模型。

### 5.1 Document IR

用于忠实表示 Word 原始内容，必须保留内容顺序、原始来源和原始资源。

示意：

```json
{
  "document_id": "doc_f82a...",
  "source_file": "数学试卷.docx",
  "source_sha256": "...",
  "blocks": [
    {
      "block_id": "b001",
      "order": 1,
      "type": "paragraph",
      "raw_text": "1. 已知函数",
      "normalized_text": "1. 已知函数",
      "style_id": "Normal",
      "numbering": {
        "resolved_label": "1."
      }
    },
    {
      "block_id": "b002",
      "order": 2,
      "type": "formula",
      "source_type": "omml",
      "latex": "f(x)=x^2"
    },
    {
      "block_id": "b003",
      "order": 3,
      "type": "image",
      "asset_id": "img_..."
    }
  ]
}
```

Document IR 必须满足：

- 原始 Block Order 不可破坏；
- 原始证据不可被 OCR/LLM 结果覆盖；
- 后续识别只能追加 recognized/normalized 值；
- 任一内容块都可反查源文档及源 XML/资源关系。

### 5.2 Question IR / Answer IR

Question Candidate 只引用 Document IR Block，不要求模型重写题干：

```json
{
  "question_candidate_id": "qc_001",
  "document_id": "doc_001",
  "content_blocks": ["b100", "b101", "b102"],
  "question_number": "12",
  "question_type_candidate": "选择题",
  "split_score": 0.991
}
```

Answer Candidate：

```json
{
  "answer_candidate_id": "ac_018",
  "document_id": "doc_answer_03",
  "question_number": "12",
  "answer": "C",
  "analysis": "因为……",
  "source_blocks": ["b300", "b301"],
  "extract_score": 0.997
}
```

拆题与答案匹配必须是独立阶段。

### 5.3 Final Question

只有通过全部质量门禁的 Matched Question 才转换为最终 19 字段结构。

内部调试信息、模型调用信息、Match Evidence 不直接污染正式 Schema；相关信息保存在内部 workspace 和 `static_info` 的允许范围内。

## 6. DOCX 解析引擎

### 6.1 技术实现

- Python 3.12+
- `python-docx`
- `lxml`
- `zipfile`
- 自定义 OOXML Parser

禁止依赖 Microsoft Office COM 或 macOS 专有 Office 自动化组件，以保证后续可迁移 Linux 服务器。

重点解析：

```text
word/document.xml
word/numbering.xml
word/styles.xml
word/_rels/document.xml.rels
word/media/*
word/header*.xml
word/footer*.xml
word/embeddings/*
```

### 6.2 内容顺序

必须直接遍历 `w:body` 及段落内部 Run/Drawing/Formula 子节点，不能使用“先 paragraphs、再 tables”的读取方式导致原始顺序丢失。

必须支持：

```text
文字 → 公式 → 文字 → 图片 → 文字 → 表格
```

按原顺序重建。

### 6.3 Word 自动编号

解析 `numbering.xml` 中的 `numId / ilvl / abstractNum / lvlText / start`，还原：

- `1.`
- `（1）`
- `①`
- `A.`
- `一、`
- 多级编号。

题号属于后续拆题的强证据。

### 6.4 表格

Word 原生表格直接解析 XML，不调用 OCR。

- 简单表格 → Markdown；
- 复杂合并单元格 → HTML；
- 单元格内部继续支持文本、图片、公式；
- 保留行列、合并关系和原始顺序。

### 6.5 OMML 公式

处理链路：

```text
OMML → MathML/AST → LaTeX → Formula Validator
```

原生公式成功转换时不调用外部 OCR。

### 6.6 MathType / OLE

优先尝试解析嵌入对象；无法解析时提取预览图并调用 Formula OCR。仍无法可靠恢复时标记 `FORMULA_UNRESOLVED`。

关键公式 unresolved 时整题不得进入正式结果。

### 6.7 文本框、页眉页脚与噪声

- 文本框必须被发现并作为独立 Block 保存；
- 页眉页脚不进入题干，但可用于试卷标题、年份、学科、城市等 Metadata Candidate；
- 重复页码、水印、网站地址等先标记为 Noise Candidate，不能在 Parser 阶段无证据删除。

## 7. 图片与识别路由

### 7.1 图片资产

图片以原始二进制 SHA-256 作为稳定身份，文件名使用 hash 前缀，避免重名并支持去重。

图片原件永远保留；OCR/Vision 输出不能覆盖原图。

### 7.2 图像类型路由

先分类，再选择服务：

- `TEXT_IMAGE` → Text OCR
- `FORMULA_IMAGE` → Formula OCR
- `TABLE_IMAGE` → Table Recognition
- `QUESTION_SCREENSHOT` → Vision OCR + Multimodal LLM
- `DIAGRAM / GEOMETRY / CHART / MAP / CHEMISTRY` → 保留原图 + Vision Understanding
- `MIXED / UNKNOWN` → Multimodal fallback

几何图、函数图、实验图等即使可提取部分文字，也必须保留原图引用。

### 7.3 Provider 抽象

V1 不绑定具体供应商，至少定义：

```text
TextOCRProvider
FormulaOCRProvider
TableRecognitionProvider
VisionProvider
LLMProvider
```

业务层只能依赖统一 Provider Contract。

配置支持 Primary / Fallback / Verifier：

```yaml
providers:
  text_ocr:
    primary: provider_a
    fallback: provider_b
  formula_ocr:
    primary: provider_c
    fallback: provider_d
  vision:
    primary: provider_e
  llm:
    primary: provider_f
    verifier: provider_g
```

### 7.4 识别结果校验

模型或供应商自报 confidence 不可作为唯一依据。

系统维护 provider-specific calibration，将供应商原始 confidence、结构校验、结果一致性和黄金集表现转换为内部 `normalized_score`。

V1 默认策略：

- 非关键内容：`normalized_score >= 0.95` 可接受；
- 关键 OCR/公式/表格：`>= 0.98` 才可单路接受；
- `0.90～0.98`：触发第二供应商或独立验证；
- `< 0.90`：直接拒绝；
- 两个高分结果互相冲突：拒绝，不选“更高的那个”。

这些阈值必须配置化、版本化；真实黄金集上线后只能通过版本化 benchmark 调整。

## 8. Document Understanding 与文件聚类

### 8.1 文档分类

每个 Word 分类为：

- `QUESTION`
- `ANSWER`
- `QUESTION_AND_ANSWER`
- `MIXED`
- `UNKNOWN`

同时生成文档级 Metadata Candidate：

- 试卷标题；
- 学科；
- 年级；
- 年份；
- 城市；
- 考试类型；
- 题号序列；
- 答案号序列。

### 8.2 Exam Cluster

禁止在全部文件之间直接进行答案匹配。

先根据以下证据聚类：

- 文件名；
- 文档标题；
- 学科；
- 年级；
- 年份；
- 城市；
- 考试类型；
- 题号数量和序列；
- 文本语义特征。

Answer Matcher 默认只能在同一 Exam Cluster 中匹配。跨 Cluster 仅允许在满足“唯一强关联”的显式规则时发生；V1 默认关闭跨 Cluster 匹配，以精度优先。

## 9. Question Builder

### 9.1 拆题策略

采用“结构规则 + LLM 语义判断”。

结构证据优先：

- Word 编号；
- 题号模式；
- 章节/题型标题；
- 选项 A/B/C/D；
- 换行和段落样式；
- 分值；
- 子问题编号；
- Block 连续性。

LLM 仅处理结构证据无法消歧的边界。

### 9.2 禁止模型重写原题

LLM 最好只返回：

```json
{
  "content_blocks": ["b100", "b101", "b102"]
}
```

最终 `text_question` 必须从 Document IR Block 原文/识别结果按顺序拼装，避免模型重写造成漏字、数字改变、公式改变。

### 9.3 复合题

V1 默认保留“共享材料 + 多个小问”为一道复合题，避免拆开后丢失公共题干。只有证据明确且答案结构独立时才允许拆分。

### 9.4 拆题 Gate

V1 默认：

- `split_score >= 0.98`；
- 不存在未归属的关键 Block；
- 不存在明显截断；
- 选项结构无矛盾；
- 关键图片/公式/表格均可追溯。

任一失败 → `QUESTION_SPLIT_LOW_CONFIDENCE` 或 `QUESTION_CONTENT_INCOMPLETE`。

## 10. Answer Extractor

答案只允许从原始 Word 内容中抽取。

允许：

- 答案提取；
- 答案/解析分离；
- 格式清洗；
- 原始答案文本标准化；
- 跨文件归属判断。

禁止：

- 解题；
- 补答案；
- 因“模型认为原答案错误”而改答案。

规则：

- 原始答案不存在 → `ANSWER_NOT_FOUND`；
- 答案与解析无法可靠区分 → 全部放 `text_answer`，`answer_analysis="略"`；
- 解答题必须从原文抽取最终答案；若原始材料只有过程且无法可靠抽取最终答案，拒绝。

## 11. Answer Matcher

这是 V1 最高风险模块。

### 11.1 多证据匹配

不得使用“题号相同=匹配”。Match Evidence 至少包括：

- Exam Cluster；
- 文件关系；
- 题号；
- 题号序列；
- 题型；
- 题目数/答案数；
- 答案格式；
- 上下文顺序；
- 文本语义一致性；
- 文件名/标题证据。

### 11.2 序列对齐

匹配以整份题目序列和答案序列为基本单位，而不是逐题独立贪心匹配。使用动态规划/序列对齐，允许：

- 缺题；
- 缺答案；
- 编号跳号；
- 局部插入内容。

缺少 Q3 的答案不能导致 Q4、Q5 整体错位。

### 11.3 Abstention

V1 默认通过条件：

- `match_score >= 0.995`；
- `top1 - top2 >= 0.10`；
- Cluster 关系无冲突；
- 序列对齐无明显矛盾；
- 独立 Answer Verifier = PASS；
- Verifier `normalized_score >= 0.995`。

若 Top1 与 Top2 都很高但接近，必须 `ANSWER_MATCH_AMBIGUOUS`，不能选择 Top1。

### 11.4 Independent Verifier

Verifier 不重新生成答案，只判断：

> 是否有足够证据证明该 Answer Candidate 属于该 Question Candidate？

Matcher 与 Verifier 必须使用独立 Prompt；配置允许使用不同模型/供应商。

## 12. Metadata Normalizer

### 12.1 来源优先级

元数据按以下优先级取值：

1. 原始文档明确字段；
2. 文件名；
3. 标题/页眉；
4. 同 Exam Cluster 其他文件；
5. LLM 推断。

每个内部 Metadata Value 保存 `value/source/score`。

### 12.2 核心与非核心字段

核心内容：

- `text_question`
- `text_answer`
- 题目—答案对应关系
- 关键公式
- 关键表格
- 关键图片及位置

核心不可靠 → 整题拒绝。

非核心字段允许未知或空值：

- 知识点；
- 考点；
- 出版社；
- 教材版本；
- 年份；
- 城市；
- 赛事信息等。

字段规范本身提供 `未知` 枚举时使用 `未知`；非必填自由文本无可靠值时使用空字符串 `""`，不得编造。

## 13. 最终 19 字段 Contract

以下规则以用户提供的 Excel 字段规范为唯一 V1 对外交付基线。

| 字段 | 类型 | 必填 | V1 规则 |
|---|---|---:|---|
| `text_question` | string | 是 | Markdown；公式 LaTeX；简单表格 Markdown；复杂表格 HTML；图片 `<img src="image/...">`；保持原顺序 |
| `is_pic_included` | int | 是 | 仅允许 `0/1`，由最终题干是否含图片引用程序计算 |
| `text_answer` | string | 是 | 只能来自原始资料，不允许 AI 生成 |
| `answer_analysis` | string | 是 | 原文无解析时填 `略` |
| `text_course` | string | 是 | `语文/数学/英语/科学/政治/历史/地理/物理/化学/生物/未知` |
| `text_grade_level` | string | 是 | 小学一～六、初中一～三、高中一～三、未知 |
| `text_grade` | string | 是 | `小学/初中/高中/未知`，必须与 `text_grade_level` 一致 |
| `knowledge_points` | string | 否 | 无可靠值填空字符串 |
| `exam_points` | string | 否 | 无可靠值填空字符串 |
| `publisher` | string | 否 | 无可靠值填空字符串 |
| `text_paper` | string | 是 | 无来源时允许 `未知` |
| `textbook_version` | string | 否 | 无可靠值填空字符串 |
| `static_info` | string | 是 | **合法 JSON 对象序列化后的字符串**，至少含 `slim_question_md5` 与 `copyright` |
| `language` | string | 是 | ISO 639-1 小写码；无法可靠映射时整题拒绝 |
| `text_year` | string | 否 | 年份必须字符串，如 `"2024"`；无可靠值为空字符串 |
| `entrance_exam_type` | string | 是 | `小升初/中考/高考/未知` |
| `text_city` | string | 否 | 无可靠值为空字符串 |
| `question_type` | string | 是 | `判断题/选择题/填空题/问答题/其他题型/未知` |
| `competition_event` | string | 否 | 无可靠值为空字符串 |

### 13.1 `static_info`

Excel 将 `static_info` 定义为 `string`，因此内部使用对象，Export 时执行严格 JSON 序列化。

最小内容：

```json
{
  "slim_question_md5": "1aa3f723b09edd2d912f0bb5b06a8e7b",
  "copyright": "0"
}
```

建议同时允许加入：

- `source_files`
- `source_question_blocks`
- `source_answer_blocks`
- `pipeline_version`
- `original_metadata`

不得放 API Key、请求密钥等敏感信息。

Excel 示例中 `static_info` 样例存在疑似多余引号/转义错误；实现以“字段类型为 string 且字符串内容必须是合法 JSON”作为正式规则。

### 13.2 `language`

- 遵循 ISO 639-1；
- 全部小写；
- 中英混合时取主要作答语言；
- 内部可保存 `detected_languages`；
- 无法可靠映射为 ISO 639-1 时拒绝，不能创造 `unknown` 等非标准代码。

### 13.3 `slim_question_md5`

基于最终 `text_question` 做稳定 canonicalization：

- Unicode 规范化；
- 换行统一为 LF；
- 去除首尾空白；
- 压缩无意义连续空行；
- 不删除数字、公式、图片引用、题目实际内容。

对 canonicalized string 计算 MD5。

算法版本固定为 `slim_md5_v1`，任何算法变更必须升版本，不允许静默修改。

## 14. Quality Gates

### Gate 0：File Gate

检查：DOCX 可读取、ZIP/XML 完整、媒体关系可解析、文件未加密损坏。

### Gate 1：Document IR Gate

检查：

- 内容块无静默丢失；
- 关键关系无断链；
- 图片可追溯；
- 公式 unresolved 有明确标记；
- Block Order 稳定。

### Gate 2：Recognition Gate

检查：

- 关键 OCR/Formula/Table 满足阈值；
- 低置信度已走 fallback；
- 多供应商高置信度冲突时拒绝；
- 关键图片实际存在。

### Gate 3：Question Split Gate

检查：

- 拆题边界唯一；
- 关键内容完整；
- 选项/小问结构无明显矛盾；
- 不存在未归属关键 Block。

### Gate 4：Answer Match Gate

检查：

- 原始答案存在；
- 答案来源可追溯；
- Match Score 达标；
- Margin 达标；
- 序列对齐无冲突；
- Independent Verifier 通过。

### Gate 5：Final Contract Gate

检查：

- 19 字段类型；
- 必填字段；
- 枚举；
- 年级/年级段一致性；
- ISO 639-1；
- 图片引用文件存在；
- `static_info` 可二次 JSON parse；
- MD5 格式与算法正确；
- JSONL 每行独立可解析。

## 15. Rejected 数据

所有拒绝项进入 `rejected.jsonl`，不得静默丢弃。

统一 reason code：

- `DOCUMENT_PARSE_FAILED`
- `DOCUMENT_RELATION_BROKEN`
- `QUESTION_SPLIT_LOW_CONFIDENCE`
- `QUESTION_CONTENT_INCOMPLETE`
- `OCR_LOW_CONFIDENCE`
- `FORMULA_UNRESOLVED`
- `TABLE_UNRESOLVED`
- `IMAGE_MISSING`
- `ANSWER_NOT_FOUND`
- `ANSWER_MATCH_AMBIGUOUS`
- `ANSWER_VERIFICATION_FAILED`
- `LANGUAGE_UNRESOLVED`
- `SCHEMA_VALIDATION_FAILED`

示意：

```json
{
  "candidate_id": "qc_9281",
  "stage": "answer_matching",
  "reason_code": "ANSWER_MATCH_AMBIGUOUS",
  "details": {
    "top1": 0.996,
    "top2": 0.991
  },
  "source_files": ["试卷.docx", "答案A.docx", "答案B.docx"]
}
```

## 16. 缓存、幂等与断点续跑

### 16.1 缓存

所有外部高成本调用必须缓存。

Cache Key：

```text
content_hash + provider + model_version + prompt_version + recognition_task
```

V1 使用 SQLite 索引 + 文件缓存。

### 16.2 Stage 状态

每个文档/资产/题目阶段保存：

- `PENDING`
- `RUNNING`
- `COMPLETED`
- `FAILED`
- `REJECTED`

任务中断后从最后未完成阶段恢复，不重复已成功且缓存仍有效的 OCR/LLM 调用。

### 16.3 Run Fingerprint

由以下内容计算：

```text
input_hash + config_hash + pipeline_version
```

相同输入、相同配置、相同流水线版本应产生稳定 ID 并最大程度复用结果。

## 17. 外部 API 调用策略

### 17.1 Retry

可重试：

- timeout；
- HTTP 429；
- HTTP 5xx；
- connection reset。

使用指数退避 + jitter。

不可重试并应快速失败：

- 401；
- 403；
- 明确参数错误；
- Provider Contract 不满足。

### 17.2 并发

- DOCX/CPU 解析：线程池或进程池；
- OCR/LLM HTTP：`asyncio`；
- 每个 Provider 独立 `Semaphore`；
- 并发量配置化，避免供应商限流。

## 18. 本地存储

V1：

- SQLite：任务、Stage、Provider Call、Cache Index、状态；
- JSON 文件：IR、Candidate、Match Evidence；
- 文件系统：图片和中间资产。

后续服务器化替换边界：

- SQLite → PostgreSQL；
- 文件 → Object Storage；
- 本地 Cache → Redis；
- 本地 Worker → Queue + Worker。

核心 Domain 与 Pipeline 不依赖具体基础设施。

## 19. CLI

V1 主命令：

```bash
qbuilder run \
  --input ./input \
  --output ./result \
  --config ./config.yaml
```

其他命令：

```bash
qbuilder resume <run_id>
qbuilder report <run_id>
qbuilder config validate
```

API Key 只能通过环境变量或系统 Secret 注入，不写入 YAML、日志或 `static_info`。

## 20. 配置

示意：

```yaml
pipeline_version: "0.1.0"

quality:
  precision_first: true
  noncritical_recognition_accept: 0.95
  critical_recognition_accept: 0.98
  recognition_fallback_floor: 0.90
  split_accept: 0.98
  answer_match_accept: 0.995
  answer_match_margin: 0.10
  answer_verify_accept: 0.995
  reject_on_missing_answer: true
  reject_on_unresolved_formula: true
  reject_on_ambiguous_match: true

providers:
  text_ocr:
    primary: provider_a
    fallback: provider_b
  formula_ocr:
    primary: provider_c
    fallback: provider_d
  table:
    primary: provider_e
  vision:
    primary: provider_f
  llm:
    primary: provider_g
    verifier: provider_h

concurrency:
  document_parse: 4
  text_ocr: 8
  formula_ocr: 4
  table: 4
  vision: 4
  llm: 6

output:
  copyright_default: "0"
  slim_md5_version: "slim_md5_v1"
```

阈值属于 Versioned Configuration；修改阈值必须记录到 Run Metadata，并触发黄金集回归。

## 21. Prompt 管理

所有 Prompt 文件化、版本化，例如：

```text
prompts/
├── document_classify/v1.txt
├── question_split/v1.txt
├── answer_extract/v1.txt
├── answer_verify/v1.txt
└── metadata/v1.txt
```

每次调用记录：

- provider；
- model；
- request id；
- prompt version；
- latency；
- token/计费信息；
- normalized score；
- fallback 原因。

## 22. 可追溯性

任一正式试题必须支持：

```text
Final Question
  ↓
Matched Question
  ↓
Question Candidate / Answer Candidate
  ↓
Document IR Blocks
  ↓
原始 DOCX + 资源
```

即能够回答：

- 题干来自哪个 Word、哪些 Block；
- 答案来自哪个 Word、哪些 Block；
- 哪个模型参与拆题/验证；
- 哪个 OCR 识别了哪张图；
- 为什么匹配通过；
- 当时使用的配置与流水线版本。

## 23. 成本与运行报告

每次任务生成 `run_report.json`，至少包含：

- Word 数；
- Question Candidate 数；
- Accepted / Rejected；
- Acceptance Rate；
- 各 Reject Reason 数量；
- Text OCR / Formula OCR / Table / Vision / LLM 调用次数；
- Cache Hit Rate；
- Provider 错误与 fallback 次数；
- 总耗时、平均耗时；
- Token 与估算费用。

该报告用于后续同时优化准确率、召回率、稳定性和成本。

## 24. 测试体系

### 24.1 Synthetic Fixtures

开发阶段程序化构造 DOCX：

- 纯文本；
- Word 自动编号；
- 图片题；
- OMML 公式；
- MathType/OLE fallback；
- Word 表格；
- 图片表格；
- 图文混排；
- 同文件题目+答案；
- 跨文件答案；
- 答案缺失；
- 题号错乱；
- 重复题号；
- 两套试卷混合；
- 多语言；
- 复合题；
- 关键公式识别失败；
- 两个答案候选接近造成歧义。

### 24.2 四层测试

1. Unit Test：编号、OMML、Markdown、MD5、枚举、Schema 等；
2. Golden Parser Test：固定 DOCX → Document IR 必须稳定；
3. Provider Contract Test：所有供应商适配器行为一致；
4. E2E：DOCX 目录 → `questions.jsonl + image/`。

### 24.3 真实黄金集

真实样本不阻塞开工。

- 中期：20～50 个真实 Word 用于暴露真实格式问题；
- 稳定期：500～1000 道人工确认试题形成 Gold Dataset；
- 任何 Parser/Prompt/Model/OCR/Matcher/Threshold 变更都必须跑 Gold Regression。

## 25. 验收标准

本项目不以“尽量多输出”为首要目标。

验收指标优先级：

1. Answer Wrong-Match Rate；
2. Accepted Precision；
3. Question Integrity；
4. 核心字段准确率；
5. Recall / Acceptance Rate；
6. 成本和吞吐。

V1 的产品原则是：

> 在黄金集上，发现一道“题 A 配成题 B 答案”都视为严重缺陷；可以通过提高拒绝率来消除错配。

正式发布 Gate：

- 所有自动测试通过；
- Gold Dataset 中正式 Accepted 数据不得出现已知答案错配；
- 所有关键公式/图片/表格完整；
- Final Contract 100% 合法；
- 所有 rejected 均有 reason code；
- 任一 Accepted 可追溯到原始证据。

## 26. 推荐工程目录

```text
question-builder/
├── pyproject.toml
├── README.md
├── config/
├── prompts/
├── src/question_builder/
│   ├── cli/
│   ├── domain/
│   ├── parser/docx/
│   ├── recognition/providers/
│   ├── understanding/
│   ├── splitter/
│   ├── answer/
│   ├── matching/
│   ├── metadata/
│   ├── quality/
│   ├── export/
│   ├── storage/
│   ├── cache/
│   └── pipeline/
├── tests/
└── fixtures/
```

每个模块只承担一个明确职责，通过 Domain Contract 通信；Provider DTO 不得渗透到业务 Domain。

## 27. 服务器化演进

V1 稳定后：

```text
CLI → HTTP API / Web
SQLite → PostgreSQL
Local Files → Object Storage
Local Cache → Redis
Single Process → Queue + Worker
macOS → Linux Server
```

Document IR、Question/Answer Domain、Quality Gate、Provider Contract 与 Final Schema 不需要重写。

## 28. Design Decisions Summary

本设计已明确以下关键决策：

- V1 只支持 `.docx`；
- macOS 本地 CLI 首发；
- Python 3.12+；
- OCR/LLM 均使用外部服务；
- 多供应商可切换；
- 支持 DOCX 内图片、表格、公式、整题截图及多语言；
- 不限制输入文件内部组织方式；
- 全自动，无人工审核环节；
- 正式结果精度优先，允许降低召回率；
- 核心字段必须可靠，非核心字段允许未知/空；
- 原始答案缺失即拒绝，AI 不得自行解题；
- Document IR 是不可跳过的中间层；
- 先 Exam Cluster，再做答案匹配；
- Answer Matcher 必须支持 Abstain；
- Matcher 后增加独立 Verifier；
- `static_info` 对外类型严格为 string，内容为合法 JSON；
- `language` 严格 ISO 639-1 小写；
- 正式交付只包含 `questions.jsonl + image/`。

## 29. 设计结论

Question Builder V1 的核心不是单次模型推理，而是：

```text
Document IR
+ Recognition Routing
+ Question Boundary Reconstruction
+ Exam Clustering
+ Answer Sequence Matching
+ Abstention
+ Independent Verification
+ Multi-stage Quality Gates
+ Traceability
```

只要这些边界在实现阶段不被破坏，V1 可以先在 macOS 单机稳定处理几千题，再逐步扩展到数万和百万级生产，而无需推倒核心架构。
