# Question Builder V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a macOS-first Python 3.12+ CLI that turns heterogeneous `.docx` question/answer files into high-precision `questions.jsonl + image/` with source traceability, provider abstraction, abstention, quality gates, caching, and resume.

**Architecture:** DOCX is parsed deterministically into immutable Document IR. OCR/LLM services only add recognition or semantic evidence; question splitting, answer extraction, matching, normalization, and final export remain separate stages. Formal output is produced only after multi-stage programmatic gates pass.

**Tech Stack:** Python 3.12+, Pydantic v2, Typer, python-docx, lxml, PyYAML, httpx, tenacity, aiosqlite, pytest, pytest-asyncio, pytest-cov, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-08-29-question-builder-v1-design.md`

## Global Constraints

- Input is `.docx` only; PDF/`.doc`/Excel/Web are out of scope.
- macOS is the first runtime; no Office COM or macOS-only Office automation.
- Formal delivery is only `questions.jsonl + image/`.
- Original answers are authoritative; AI may extract/match but must never solve, invent, or correct answers.
- Precision beats recall; ambiguous critical evidence is rejected.
- Cross-Exam-Cluster answer matching is disabled in V1.
- External fields follow the approved 19-field contract exactly.
- `static_info` is a JSON string; `language` is lowercase ISO 639-1; `text_year` is string; `is_pic_included` is 0/1.
- Default thresholds: noncritical recognition 0.95, critical recognition 0.98, fallback floor 0.90, split 0.98, answer match 0.995, margin 0.10, verifier 0.995.
- Provider/prompt/model/calibration/threshold/pipeline versions must be recorded when they can affect accepted output.
- Every task uses TDD and ends with an independent review Gate.

## Locked Structure

```text
shiti/
├── pyproject.toml
├── README.md
├── config/default.yaml
├── prompts/{document_classify,question_split,answer_extract,answer_verify,metadata}/v1.txt
├── docs/superpowers/{specs,plans}/
├── src/question_builder/
│   ├── cli/app.py
│   ├── config/models.py
│   ├── domain/{document,question,answer,matching,final,quality}.py
│   ├── parser/docx/{package,body,numbering,table,formula,textbox,assets}.py
│   ├── recognition/{contracts,calibration,router}.py
│   ├── recognition/providers/fake.py
│   ├── understanding/{classifier,clustering}.py
│   ├── splitter/{rules,builder}.py
│   ├── answer/extractor.py
│   ├── matching/{alignment,scoring,verifier}.py
│   ├── metadata/normalizer.py
│   ├── quality/gates.py
│   ├── export/{markdown,jsonl}.py
│   ├── storage/{db,workspace}.py
│   ├── cache/service.py
│   ├── pipeline/orchestrator.py
│   └── report/run_report.py
├── tests/{unit,contract,golden,e2e}/
└── fixtures/{synthetic,gold}/
```

---

### Task 0: Foundation, Config, and Domain Contracts

**Files:** `pyproject.toml`, `README.md`, `config/default.yaml`, `src/question_builder/config/models.py`, `src/question_builder/domain/*.py`, `tests/unit/test_config.py`, `tests/unit/test_domain_models.py`.

**Produces:** `AppConfig`, `QualityThresholds`, `ProviderConfig`, `DocumentIR`, `ContentBlock`, `QuestionCandidate`, `AnswerCandidate`, `MatchEvidence`, `MatchedQuestion`, `FinalQuestionRecord`, `RejectReason`, `RejectedRecord`.

- [ ] Write `pyproject.toml` with Python `>=3.12` and the Tech Stack dependencies above; expose `qbuilder = question_builder.cli.app:app`.
- [ ] Write failing config tests asserting all seven approved thresholds and `recognition_fallback_floor <= critical_recognition_accept`.

```python
def test_defaults(config):
    assert config.quality.noncritical_recognition_accept == 0.95
    assert config.quality.answer_match_accept == 0.995
    assert config.quality.answer_match_margin == 0.10
```

- [ ] Run `python -m pytest tests/unit/test_config.py -v`; expected FAIL because models do not exist.
- [ ] Implement Pydantic config models plus exact `config/default.yaml`; API keys are not config fields.
- [ ] Write failing domain tests for unique block order, exact 19-field shape, binary picture flag, grade consistency, JSON-string `static_info`, and exact reject codes from the Spec.
- [ ] Implement domain types. `ContentBlock.type` must include paragraph/formula/image/table/textbox/header/footer/noise_candidate/unresolved.
- [ ] Run:

```bash
python -m pytest tests/unit/test_config.py tests/unit/test_domain_models.py -v
python -m ruff check src tests
python -m mypy src/question_builder/config src/question_builder/domain
```

- [ ] Commit: `feat: establish question builder domain contracts`.

**Gate:** Freeze Domain Contract and Final 19-field Contract before Task 1.

---

### Task 1: Core DOCX Package, Ordering, Numbering, and Media

**Files:** `parser/docx/{package,body,numbering,assets}.py`, parser unit tests, `fixtures/synthetic/basic_ordered.docx`, `tests/golden/test_basic_document_ir.py`.

**Produces:** `parse_docx(path: Path, asset_dir: Path) -> DocumentIR`; stable source/document/block/asset identities.

- [ ] Generate and commit a DOCX fixture containing paragraphs, Word numbering, inline image, and table boundaries.
- [ ] Write failing ZIP/package tests: invalid ZIP fails; `word/document.xml` is mandatory; source SHA-256 is stable.
- [ ] Implement package inspection with `zipfile` and explicit `DocxPackageError`.
- [ ] Write failing numbering tests covering `1.`, `（1）`, `A.`, Chinese/multilevel numbering from `numbering.xml`.
- [ ] Implement `numId/abstractNum/ilvl/lvlText/start` resolution; keep label separate from `raw_text`.
- [ ] Write golden test asserting exact body order and SHA-based image materialization.
- [ ] Implement direct `w:body` traversal plus run-child ordering. `asset_id = "img_" + sha256[:16]`.
- [ ] Run `python -m pytest tests/unit/parser tests/golden/test_basic_document_ir.py -v`.
- [ ] Commit: `feat: parse ordered docx content and numbering`.

**Gate:** No identifiable source element may disappear silently; every block keeps provenance.

---

### Task 2: Tables, OMML, Text Boxes, Headers/Footers, Noise, and OLE Evidence

**Files:** `parser/docx/{table,formula,textbox}.py`, modify `body.py`, advanced parser tests, `fixtures/synthetic/advanced_content.docx`.

**Produces:** structured table/formula/textbox/header/footer evidence and explicit unresolved formula evidence.

- [ ] Write failing native-table tests for plain cells, horizontal/vertical merges, and image/formula inside cell.
- [ ] Implement `w:tbl`, `w:gridSpan`, `w:vMerge`; simple tables expose Markdown, merged tables expose HTML.
- [ ] Write failing OMML tests for fraction, superscript, subscript, root, grouping; unsupported constructs raise `FormulaConversionError`.
- [ ] Implement tested OMML→LaTeX subset plus brace/empty validation; preserve source OMML.
- [ ] Write failing tests that `w:txbxContent`, header, and footer are discovered but headers/footers do not enter body question text.
- [ ] Implement text-box anchor hints and header/footer metadata blocks.
- [ ] Mark repeated page/footer/watermark strings as `noise_candidate` only with structural evidence; never delete them in Parser.
- [ ] For unparsed `word/embeddings/*`, emit unresolved formula evidence with embedding path and preview-image relation when available.
- [ ] Run `python -m pytest tests/unit/parser tests/golden/test_advanced_document_ir.py -v`.
- [ ] Commit: `feat: preserve advanced docx content and unresolved evidence`.

**Gate:** Unsupported critical content is explicit, never converted to empty text.

---

### Task 3: Provider Contracts, Calibration, Recognition Router, and Retry

**Files:** `recognition/{contracts,calibration,router}.py`, `recognition/providers/fake.py`, contract/router tests.

**Produces:** typed async protocols for Text OCR, Formula OCR, Table Recognition, Vision, and LLM plus normalized `RecognitionResult`.

- [ ] Write provider-contract tests requiring provider, model, request id, latency, raw score/reference, normalized score, and content.
- [ ] Implement `typing.Protocol` contracts; provider SDK DTOs remain inside adapters.
- [ ] Write failing calibration tests: missing calibration cannot blindly trust provider score.
- [ ] Implement `normalize_score(provider, model, task, raw_score)` with versioned calibration identity.
- [ ] Write router tests: `>=0.98` critical primary accepts; `0.90–0.98` falls back; `<0.90` rejects; conflicting high-score results reject; noncritical acceptance is 0.95.
- [ ] Implement image classes: TEXT_IMAGE, FORMULA_IMAGE, TABLE_IMAGE, QUESTION_SCREENSHOT, DIAGRAM, GEOMETRY, CHART, MAP, CHEMISTRY, MIXED, UNKNOWN.
- [ ] Implement retry classification: timeout/429/5xx/reset retry with exponential backoff+jitter; 401/403/schema-contract errors fail fast.
- [ ] Run `python -m pytest tests/contract tests/unit/recognition -v`.
- [ ] Commit: `feat: add calibrated recognition provider routing`.

**Gate:** Provider DTOs cannot enter Domain; normalized scoring/fallback cannot be bypassed.

**Production Provider Gate:** Vendors are not invented before selection. Tasks may use contract-compliant fake providers. Before production, selected real provider adapters must each pass these contracts and calibration tests, with secrets supplied outside YAML/logs. Provider-specific work receives a separate implementation plan once vendor API contracts are known.

---

### Task 4: Document Understanding and Conservative Exam Clustering

**Files:** `understanding/{classifier,clustering}.py`, `prompts/document_classify/v1.txt`, understanding tests.

**Produces:** `DocumentUnderstanding` and `ExamCluster`.

- [ ] Write deterministic classification tests for `QUESTION`, `ANSWER`, `QUESTION_AND_ANSWER`, `MIXED`, `UNKNOWN` using headings, answer density, numbering, and option structure.
- [ ] Implement rule-first feature extraction: filename/title/header metadata, subject, grade, year, city, exam type, question/answer number sequences.
- [ ] Write LLM-fallback contract test requiring one allowed enum plus cited block ids; reject invented enum values.
- [ ] Implement classification fusion without overwriting source evidence.
- [ ] Write clustering tests: obvious question+answer pair joins; subject/year/grade conflicts prevent merge; ambiguous doc stays singleton instead of force-merge.
- [ ] Implement conservative clustering from filename/title/course/grade/year/city/exam type/number sequences/semantic evidence.
- [ ] Run `python -m pytest tests/unit/understanding -v`.
- [ ] Commit: `feat: classify documents and build conservative exam clusters`.

**Gate:** V1 Answer Matcher receives candidates only from the same accepted Exam Cluster.

---

### Task 5: Question Splitter and Source-Preserving Reconstruction

**Files:** `splitter/{rules,builder}.py`, `export/markdown.py`, `prompts/question_split/v1.txt`, split/golden tests.

**Produces:** `build_question_candidates(...) -> list[QuestionCandidate]` and source-derived Markdown reconstruction.

- [ ] Write structural split tests for numeric/Chinese numbering, choices, section headings, subquestions, and answer sections.
- [ ] Implement rule-generated candidate ranges with evidence; rules do not auto-accept weak boundaries.
- [ ] Write LLM-disambiguation test: output may contain block ids/ranges and confidence only; model-rewritten question text is not authoritative.
- [ ] Implement LLM-assisted boundary selection validating all referenced blocks exist and preserve order.
- [ ] Write reconstruction test for text→formula→text→image→text→table and unresolved critical content.
- [ ] Implement rendering: normalized source text, LaTeX, `<img src="image/<hash-name>">`, Markdown simple table, HTML complex table.
- [ ] Implement conservative compound-question policy: shared material + subquestions stays one candidate unless structure and answer independence are both explicit.
- [ ] Emit `QUESTION_SPLIT_LOW_CONFIDENCE` for uncertain boundary and `QUESTION_CONTENT_INCOMPLETE` for missing critical blocks.
- [ ] Run `python -m pytest tests/unit/splitter tests/golden/test_question_reconstruction.py -v`.
- [ ] Commit: `feat: build traceable question candidates from source blocks`.

**Gate:** LLM paraphrase can never become `text_question` source truth.

---

### Task 6: Source-Backed Answer Extraction

**Files:** `answer/extractor.py`, `prompts/answer_extract/v1.txt`, answer tests.

**Produces:** `extract_answer_candidates(...) -> list[AnswerCandidate]` where each nonempty answer cites source blocks.

- [ ] Write tests for compact answer lists, per-line answers, answer+analysis, long-form solution final answer, and missing answer.
- [ ] Write prompt contract explicitly forbidding solving/correcting/filling missing answers and requiring source block ids.
- [ ] Implement deterministic numbering/answer-section extraction first.
- [ ] Implement LLM fallback and source-span verification; unsupported model output is rejected.
- [ ] Implement answer/analysis split: if inseparable, keep all in answer and set analysis `略`; if no reliable original final answer exists, reject.
- [ ] Add regression where fake LLM returns a plausible answer absent from source; expected reject.
- [ ] Run `python -m pytest tests/unit/answer -v`.
- [ ] Commit: `feat: extract source-backed answer candidates`.

**Gate:** No code path may create an answer without original source evidence.

---

### Task 7: Sequence Alignment, Answer Matching, Abstention, and Independent Verifier

**Files:** `matching/{alignment,scoring,verifier}.py`, `prompts/answer_verify/v1.txt`, matching/golden tests.

**Produces:** `AlignmentResult`, `MatchEvidence`, `MatchedQuestion`, structured ambiguity/verifier rejects.

- [ ] Write dynamic-alignment tests: perfect sequence, missing A3 without shifting Q4, duplicate number ambiguity, inserted commentary.
- [ ] Implement DP alignment with match/skip-question/skip-answer operation trace.
- [ ] Write scoring tests covering cluster, document relation, number, sequence, type, counts, answer format, filename/title, semantic evidence.
- [ ] Implement versioned weighted scoring; conflicting cluster evidence cannot pass.
- [ ] Write exact abstention cases:

```text
0.996 vs 0.80 + verifier 0.997 => PASS
0.996 vs 0.991 => ANSWER_MATCH_AMBIGUOUS
0.994 vs 0.20 => reject below match threshold
0.999 + verifier 0.994 => ANSWER_VERIFICATION_FAILED
```

- [ ] Implement threshold + margin policy; never select top-1 when margin fails.
- [ ] Implement verifier with output limited to PASS/FAIL, score, reason, cited blocks; replacement answer is not an allowed field.
- [ ] Build golden fixture containing duplicate numbers and tempting wrong mappings; expected accepted mappings are exact.
- [ ] Run `python -m pytest tests/unit/matching tests/golden/test_answer_matching_precision.py -v`.
- [ ] Commit: `feat: add precision-first answer matching and verification`.

**Gate:** Any known wrong accepted mapping fails Task 7 even if recall is high.

---

### Task 8: Metadata Normalization and Final 19-Field Builder

**Files:** `metadata/normalizer.py`, `domain/final.py`, metadata/final-contract tests.

**Produces:** `NormalizedMetadata` and `FinalQuestionRecord` with exactly 19 external fields.

- [ ] Write enum normalization tests (`初一→初中一年级`, `单选题→选择题`, supported grade/course/exam/question enums).
- [ ] Implement source priority: explicit document → filename → title/header → same cluster → LLM inference; preserve source/score internally.
- [ ] Write contract tests for 0/1 picture flag, ISO 639-1, string year, grade consistency, mandatory answer, and parseable `static_info`.
- [ ] Implement `slim_md5_v1`: Unicode normalization, LF, trim, collapse meaningless blank lines, preserve actual content/formulas/image refs, MD5 lowercase hex.
- [ ] Implement `static_info = json.dumps(internal_provenance, ensure_ascii=False, sort_keys=True)` containing MD5, copyright, source files/blocks, pipeline version, md5 version.
- [ ] Ensure optional non-core free text is `""` when unknown; contract enums use `未知` only where allowed.
- [ ] Run `python -m pytest tests/unit/metadata tests/unit/export/test_final_contract.py -v`.
- [ ] Commit: `feat: normalize metadata into final question contract`.

**Gate:** Internal debug/provider evidence does not leak into external fields except approved provenance serialized in `static_info`.

---

### Task 9: Multi-Stage Quality Gates and Structured Rejects

**Files:** `quality/gates.py`, `domain/quality.py`, quality tests.

**Produces:** deterministic Gate chain: File → Document IR → Recognition → Split → Answer Match → Final Contract.

- [ ] Write gate-order test where multiple failures exist; primary reason must be stable and earliest applicable.
- [ ] Implement File/Document gates for unreadable source, broken relations, missing critical block, unresolved critical formula/table.
- [ ] Implement Recognition/Split gates using normalized thresholds and fallback evidence.
- [ ] Implement Answer Gate requiring original answer, match score, margin, cluster/sequence consistency, verifier pass.
- [ ] Implement Final Gate: validate 19 fields, parse `static_info`, verify image files, recompute MD5.
- [ ] Assert every Spec reason code serializes with candidate id, stage, reason, details, source files.
- [ ] Run `python -m pytest tests/unit/quality -v`.
- [ ] Commit: `feat: enforce multi-stage precision quality gates`.

**Gate:** Every nonfatal data rejection is structured; critical failure cannot silently reach accepted output.

---

### Task 10: SQLite State, Workspace, Cache, Idempotency, and Resume

**Files:** `storage/{db,workspace}.py`, `cache/service.py`, storage/cache tests.

**Produces:** `RunStore`, `WorkspaceStore`, `CacheService`; states PENDING/RUNNING/COMPLETED/FAILED/REJECTED.

- [ ] Write DB tests requiring runs, stage states, provider calls, cache index, version/config/input hashes.
- [ ] Implement transactional `RunStore` with validated state transitions.
- [ ] Write workspace round-trip tests for Document IR, candidates, answers, match evidence.
- [ ] Implement atomic JSON writes via temp file + `os.replace`.
- [ ] Write cache tests: identical content/provider/model/prompt/task hits; any version change misses.
- [ ] Implement immutable payload cache + SQLite index; never persist API headers/secrets.
- [ ] Write resume test: parse/recognition completed, split pending; resuming must not call parse/OCR fake again.
- [ ] Implement run fingerprint from input hash + config hash + pipeline version and first-incomplete-stage selection.
- [ ] Run `python -m pytest tests/unit/storage tests/unit/cache -v`.
- [ ] Commit: `feat: persist resumable runs and versioned provider cache`.

**Gate:** Interrupted work reuses completed paid-provider calls when all cache-version inputs match.

---

### Task 11: Pipeline Orchestrator and CLI

**Files:** `pipeline/orchestrator.py`, `cli/app.py`, pipeline/CLI tests.

**Produces:** `QuestionBuilderPipeline.run(...) -> RunSummary`; CLI `run`, `resume`, `report`, `config validate`.

- [ ] Write exact stage-order test using fakes: ingest→parse→recognition→understanding→split→answer extract→match/verify→metadata→quality→export/report.
- [ ] Implement stage-isolated orchestrator; persist each stage before downstream execution.
- [ ] Write provider semaphore tests; configure separate concurrency for parse/text OCR/formula/table/vision/LLM.
- [ ] Implement bounded `asyncio.Semaphore` use; do not create unbounded thousands of tasks.
- [ ] Write CLI tests for:

```text
qbuilder run --input DIR --output DIR --config FILE
qbuilder resume RUN_ID
qbuilder report RUN_ID
qbuilder config validate --config FILE
```

- [ ] Implement Typer commands; invalid path/config exits nonzero; secrets are never printed.
- [ ] Run `python -m pytest tests/unit/pipeline tests/unit/cli -v && qbuilder config validate --config config/default.yaml`.
- [ ] Commit: `feat: orchestrate resumable question building from cli`.

**Gate:** Tests and production CLI use the same orchestrator path.

---

### Task 12: JSONL/Image Export, Rejected Output, Traceability, and Run Report

**Files:** `export/jsonl.py`, `report/run_report.py`, export/report tests.

**Produces:** `questions.jsonl`, `image/`, `rejected.jsonl`, `run_report.json`.

- [ ] Write export tests: one valid JSON object/line, exactly 19 fields, UTF-8, all `image/<sha-name>` refs exist, rejected items never enter accepted file.
- [ ] Implement atomic JSONL export and SHA-deduplicated image copy.
- [ ] Write traceability tests: each accepted `static_info` resolves source question blocks, answer blocks, pipeline and MD5 version; no secret leakage.
- [ ] Write run-report tests for document/candidate/accepted/rejected counts, reasons, provider calls/fallbacks, cache hit rate, latency, tokens, estimated cost.
- [ ] Implement report aggregation from persisted run/provider/export data.
- [ ] Run `python -m pytest tests/unit/export tests/unit/report -v`.
- [ ] Commit: `feat: export traceable jsonl results and run reports`.

**Gate:** Formal delivery is consumable using only `questions.jsonl + image/`.

---

### Task 13: Synthetic E2E Matrix and Gold Regression Harness

**Files:** `fixtures/synthetic/*.docx`, `fixtures/gold/README.md`, `tests/e2e/*.py`, `tests/golden/test_gold_dataset_harness.py`.

**Produces:** provider-free deterministic certification and future real Gold Dataset harness.

- [ ] Create committed fixtures for pure text, numbering, inline image, OMML, unresolved formula, native/image table, mixed blocks, same/cross-file answer, missing answer, duplicate numbers, mixed exams, multilingual, compound question, OCR failure, ambiguous answer candidates.
- [ ] Write full-pipeline E2E with fake providers asserting exact accepted ids, exact reject reasons, valid JSONL, images present.
- [ ] Write precision E2E containing a tempting wrong answer mapping; expected reject.
- [ ] Write resume E2E that interrupts after recognition and confirms provider call count does not increase on resume.
- [ ] Add Gold layout:

```text
fixtures/gold/<case>/input/*.docx
fixtures/gold/<case>/expected/questions.jsonl
fixtures/gold/<case>/expected/rejected.jsonl
```

- [ ] Implement Gold metrics: accepted precision, wrong-answer count, integrity failures, recall, acceptance rate. Empty real-Gold set skips explicitly.
- [ ] Run:

```bash
python -m pytest -v
python -m pytest --cov=question_builder --cov-report=term-missing
python -m ruff check src tests
python -m mypy src/question_builder
```

- [ ] Commit: `test: certify synthetic question builder pipeline`.

**Gate:** Synthetic accepted set has zero known wrong answer mappings. Lower recall from conservative rejection is acceptable and reported.

---

### Task 14: Operations Documentation and V1 Release Certification

**Files:** modify `README.md`; create `docs/operations/{provider-adapters,gold-dataset,v1-acceptance-checklist}.md`; `tests/unit/test_docs_examples.py`.

- [ ] Write tests that documented config loads, CLI help exposes all four commands, and examples contain no embedded secrets.
- [ ] Document macOS setup, Python 3.12 environment, install, environment variables, config validation, run/resume/report, and formal delivery.
- [ ] Document Provider Adapter Contract, retry categories, calibration/versioning, payload storage, secret rules.
- [ ] Document real-data process: 20–50 Word tuning set, then 500–1000 manually confirmed questions as Gold Dataset.
- [ ] Create acceptance checklist requiring green automated suite, valid 19 fields/assets, structured rejects, traceability, zero known wrong mappings, no AI-generated answers, and Gold evidence before production.
- [ ] Run fresh:

```bash
python -m pytest -v
python -m ruff check src tests
python -m mypy src/question_builder
qbuilder config validate --config config/default.yaml
```

- [ ] Commit: `docs: add question builder v1 operations and acceptance`.

**Gate:** Development certification needs fresh automated evidence; production certification additionally needs real Gold Dataset evidence and selected real provider adapters that pass Task 3 contracts.

---

## Dependency Order and Milestones

```text
Task 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14
```

- **M1 Parsing Baseline:** Tasks 0–2 — stable DOCX → Document IR.
- **M2 Semantic Baseline:** Tasks 3–6 — recognition, clustering, split, source-backed answers.
- **M3 Precision Baseline:** Tasks 7–9 — matcher, abstention, final normalization, gates.
- **M4 Runnable Product:** Tasks 10–12 — resume/cache/CLI/export/report.
- **M5 V1 Certification:** Tasks 13–14 — synthetic certification green; production waits for real Gold + provider adapter evidence.

## Definition of Done for V1 Development

All must hold on the same accepted commit:

1. `python -m pytest -v` passes.
2. `python -m ruff check src tests` passes.
3. `python -m mypy src/question_builder` passes.
4. `qbuilder config validate --config config/default.yaml` passes.
5. Synthetic accepted output contains zero known wrong answer mappings.
6. Every reject has a stable reason code.
7. Every accepted question traces to source question and answer blocks.
8. Every final record contains exactly 19 approved fields and passes schema validation.
9. Every accepted image reference exists in `image/`.
10. No code path creates `text_answer` without source-backed evidence.
11. Resume reuses completed paid-provider calls when cache versions match.
12. Repository contains approved Design Spec, this Plan, operations docs, and acceptance checklist.

Production approval is a later Gate: real labelled Gold regression + real selected provider adapters passing Task 3 contracts.
