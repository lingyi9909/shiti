# Question Builder V1

Question Builder converts heterogeneous K12 `.docx` question and answer material into a precision-first, traceable structured-data pipeline. V1 is macOS-first, targets Python 3.12+, and keeps OCR/LLM providers outside the domain contracts.

## Task 0 foundation

The current foundation freezes:

- Python `>=3.12` packaging and the `qbuilder` CLI entry point;
- versioned YAML configuration with no API-key fields;
- provider-independent Document, Question, Answer, Matching, Final, and Rejection contracts;
- the approved seven quality thresholds;
- the exact 19-field external question record;
- the exact rejection-reason contract and source-block traceability primitives.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/unit/test_config.py tests/unit/test_domain_models.py -v
python -m ruff check src tests
python -m mypy src/question_builder/config src/question_builder/domain
```

The formal pipeline must never solve, invent, or correct source answers. Critical ambiguity is rejected rather than guessed.
