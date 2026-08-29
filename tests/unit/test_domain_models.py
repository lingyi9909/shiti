import json

import pytest
from pydantic import ValidationError

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.final import FinalQuestionRecord
from question_builder.domain.matching import MatchedQuestion, MatchEvidence
from question_builder.domain.quality import RejectedRecord, RejectReason
from question_builder.domain.question import QuestionCandidate

FINAL_FIELDS = {
    "text_question",
    "is_pic_included",
    "text_answer",
    "answer_analysis",
    "text_course",
    "text_grade_level",
    "text_grade",
    "knowledge_points",
    "exam_points",
    "publisher",
    "text_paper",
    "textbook_version",
    "static_info",
    "language",
    "text_year",
    "entrance_exam_type",
    "text_city",
    "question_type",
    "competition_event",
}

REJECT_CODES = {
    "DOCUMENT_PARSE_FAILED",
    "DOCUMENT_RELATION_BROKEN",
    "QUESTION_SPLIT_LOW_CONFIDENCE",
    "QUESTION_CONTENT_INCOMPLETE",
    "OCR_LOW_CONFIDENCE",
    "FORMULA_UNRESOLVED",
    "TABLE_UNRESOLVED",
    "IMAGE_MISSING",
    "ANSWER_NOT_FOUND",
    "ANSWER_MATCH_AMBIGUOUS",
    "ANSWER_VERIFICATION_FAILED",
    "LANGUAGE_UNRESOLVED",
    "SCHEMA_VALIDATION_FAILED",
}


def block(block_id: str, order: int) -> ContentBlock:
    return ContentBlock(block_id=block_id, order=order, type="paragraph", raw_text=block_id)


def final_record(**overrides: object) -> FinalQuestionRecord:
    payload: dict[str, object] = {
        "text_question": "1. 1+1=?",
        "is_pic_included": 0,
        "text_answer": "2",
        "answer_analysis": "略",
        "text_course": "数学",
        "text_grade_level": "小学一年级",
        "text_grade": "小学",
        "knowledge_points": "",
        "exam_points": "",
        "publisher": "",
        "text_paper": "未知",
        "textbook_version": "",
        "static_info": json.dumps(
            {
                "slim_question_md5": "1aa3f723b09edd2d912f0bb5b06a8e7b",
                "copyright": "0",
                "source_question_blocks": ["b1"],
                "source_answer_blocks": ["a1"],
            },
            ensure_ascii=False,
        ),
        "language": "zh",
        "text_year": "2024",
        "entrance_exam_type": "未知",
        "text_city": "",
        "question_type": "填空题",
        "competition_event": "",
    }
    payload.update(overrides)
    return FinalQuestionRecord.model_validate(payload)


def test_document_ir_requires_unique_strictly_increasing_block_order() -> None:
    with pytest.raises(ValidationError):
        DocumentIR(
            document_id="doc_1",
            source_file="paper.docx",
            source_sha256="a" * 64,
            blocks=[block("b1", 1), block("b2", 1)],
        )

    with pytest.raises(ValidationError):
        DocumentIR(
            document_id="doc_1",
            source_file="paper.docx",
            source_sha256="a" * 64,
            blocks=[block("b2", 2), block("b1", 1)],
        )


def test_content_block_type_contract_includes_all_approved_values() -> None:
    allowed = {
        "paragraph",
        "formula",
        "image",
        "table",
        "textbox",
        "header",
        "footer",
        "noise_candidate",
        "unresolved",
    }
    for index, block_type in enumerate(sorted(allowed), start=1):
        parsed = ContentBlock(block_id=f"b{index}", order=index, type=block_type)
        assert parsed.type == block_type


def test_question_answer_and_matching_keep_source_block_traceability() -> None:
    question = QuestionCandidate(
        question_candidate_id="qc_1",
        document_id="doc_q",
        content_blocks=["b1", "b2"],
        question_number="1",
        question_type_candidate="选择题",
        split_score=0.999,
    )
    answer = AnswerCandidate(
        answer_candidate_id="ac_1",
        document_id="doc_a",
        question_number="1",
        answer="C",
        analysis="略",
        source_blocks=["a1"],
        extract_score=0.997,
    )
    evidence = MatchEvidence(
        question_candidate_id="qc_1",
        answer_candidate_id="ac_1",
        match_score=0.999,
        second_best_score=0.5,
        verifier_score=0.999,
        question_source_blocks=["b1", "b2"],
        answer_source_blocks=["a1"],
    )
    matched = MatchedQuestion(question=question, answer=answer, evidence=evidence)

    assert matched.question.content_blocks == ("b1", "b2")
    assert matched.answer.source_blocks == ("a1",)


def test_domain_contract_is_immutable_including_nested_block_collections() -> None:
    content = ContentBlock(
        block_id="b1",
        order=1,
        type="paragraph",
        raw_text="original",
        numbering={"resolved_label": "1.", "nested": {"level": 1}},
        metadata={"tags": ["source"]},
    )
    document = DocumentIR(
        document_id="doc_1",
        source_file="paper.docx",
        source_sha256="a" * 64,
        blocks=[content],
    )
    question = QuestionCandidate(
        question_candidate_id="qc_1",
        document_id="doc_1",
        content_blocks=["b1"],
        split_score=0.999,
    )
    answer = AnswerCandidate(
        answer_candidate_id="ac_1",
        document_id="doc_a",
        answer="A",
        source_blocks=["a1"],
        extract_score=0.999,
    )
    evidence = MatchEvidence(
        question_candidate_id="qc_1",
        answer_candidate_id="ac_1",
        match_score=0.999,
        question_source_blocks=["b1"],
        answer_source_blocks=["a1"],
    )

    with pytest.raises(ValidationError):
        content.raw_text = "tampered"
    with pytest.raises(TypeError):
        content.metadata["new"] = "tampered"
    with pytest.raises((AttributeError, TypeError)):
        content.metadata["tags"].append("tampered")
    with pytest.raises(TypeError):
        content.numbering["resolved_label"] = "2."  # type: ignore[index]
    with pytest.raises(TypeError):
        content.numbering["nested"]["level"] = 2  # type: ignore[index]

    assert isinstance(document.blocks, tuple)
    with pytest.raises(AttributeError):
        document.blocks.append(block("b2", 2))  # type: ignore[attr-defined]
    assert isinstance(question.content_blocks, tuple)
    assert isinstance(answer.source_blocks, tuple)
    assert isinstance(evidence.question_source_blocks, tuple)
    assert isinstance(evidence.answer_source_blocks, tuple)


def test_content_block_json_round_trip_preserves_content_and_immutability() -> None:
    content = ContentBlock(
        block_id="b1",
        order=1,
        type="paragraph",
        numbering={"resolved_label": "1.", "nested": {"level": 1}},
        recognized={"text": "题干", "confidence": 0.99},
        metadata={"tags": ["source"], "origin": {"page": 3}},
    )

    payload = content.model_dump_json()
    restored = ContentBlock.model_validate_json(payload)

    assert json.loads(payload)["metadata"] == {"tags": ["source"], "origin": {"page": 3}}
    assert restored == content
    with pytest.raises(TypeError):
        restored.metadata["origin"]["page"] = 4  # type: ignore[index]


def test_document_ir_json_round_trip_preserves_nested_block_content() -> None:
    content = ContentBlock(
        block_id="b1",
        order=1,
        type="paragraph",
        metadata={"tags": ["source"], "origin": {"page": 3}},
    )
    document = DocumentIR(
        document_id="doc_1",
        source_file="paper.docx",
        source_sha256="a" * 64,
        blocks=[content],
    )

    payload = document.model_dump_json()
    restored = DocumentIR.model_validate_json(payload)

    assert restored == document
    with pytest.raises(TypeError):
        restored.blocks[0].metadata["origin"]["page"] = 4  # type: ignore[index]


def test_match_evidence_json_round_trip_preserves_immutable_evidence() -> None:
    evidence = MatchEvidence(
        question_candidate_id="qc_1",
        answer_candidate_id="ac_1",
        match_score=0.999,
        question_source_blocks=["b1"],
        answer_source_blocks=["a1"],
        evidence={"rule": "same_number", "verified": True, "margin": 0.2},
    )

    payload = evidence.model_dump_json()
    restored = MatchEvidence.model_validate_json(payload)

    assert restored == evidence
    with pytest.raises(TypeError):
        restored.evidence["rule"] = "tampered"


def test_rejected_record_json_round_trip_preserves_nested_immutable_details() -> None:
    rejected = RejectedRecord(
        candidate_id="qc_1",
        stage="answer_matching",
        reason_code=RejectReason.ANSWER_MATCH_AMBIGUOUS,
        details={"scores": {"top1": 0.996, "top2": 0.991}, "candidates": ["a1", "a2"]},
        source_files=["paper.docx", "answer.docx"],
    )

    payload = rejected.model_dump_json()
    restored = RejectedRecord.model_validate_json(payload)

    assert restored == rejected
    with pytest.raises(TypeError):
        restored.details["scores"]["top1"] = 1.0  # type: ignore[index]


def test_static_info_enforces_md5_and_binary_copyright_contract() -> None:
    assert final_record().static_info
    assert final_record(
        static_info=json.dumps(
            {
                "slim_question_md5": "0123456789abcdef0123456789abcdef",
                "copyright": "1",
                "source_question_blocks": ["b1"],
                "source_answer_blocks": ["a1"],
            }
        )
    ).static_info

    invalid_md5_values = [
        "",
        "a" * 31,
        "a" * 33,
        "A" * 32,
        "g" * 32,
    ]
    for invalid_md5 in invalid_md5_values:
        with pytest.raises(ValidationError):
            final_record(
                static_info=json.dumps(
                    {
                        "slim_question_md5": invalid_md5,
                        "copyright": "0",
                        "source_question_blocks": ["b1"],
                        "source_answer_blocks": ["a1"],
                    }
                )
            )

    for invalid_copyright in ("2", "", 0, 1, True, None):
        with pytest.raises(ValidationError):
            final_record(
                static_info=json.dumps(
                    {
                        "slim_question_md5": "0123456789abcdef0123456789abcdef",
                        "copyright": invalid_copyright,
                        "source_question_blocks": ["b1"],
                        "source_answer_blocks": ["a1"],
                    }
                )
            )


def test_final_question_has_exact_19_field_shape_and_forbids_extras() -> None:
    record = final_record()
    assert set(record.model_dump()) == FINAL_FIELDS
    assert len(record.model_dump()) == 19

    with pytest.raises(ValidationError):
        final_record(provider_payload={"vendor": "must-not-leak"})


def test_picture_flag_is_strictly_binary_integer() -> None:
    assert final_record(is_pic_included=0).is_pic_included == 0
    assert final_record(is_pic_included=1).is_pic_included == 1
    with pytest.raises(ValidationError):
        final_record(is_pic_included=2)
    with pytest.raises(ValidationError):
        final_record(is_pic_included=True)


def test_grade_and_grade_level_must_be_consistent() -> None:
    with pytest.raises(ValidationError):
        final_record(text_grade="初中", text_grade_level="小学六年级")


def test_static_info_is_a_json_object_serialized_as_string() -> None:
    record = final_record()
    parsed = json.loads(record.static_info)
    assert parsed["source_question_blocks"] == ["b1"]
    assert parsed["source_answer_blocks"] == ["a1"]

    with pytest.raises(ValidationError):
        final_record(static_info={"slim_question_md5": "x", "copyright": "0"})
    with pytest.raises(ValidationError):
        final_record(static_info="not-json")
    with pytest.raises(ValidationError):
        final_record(static_info='["not", "an", "object"]')


def test_final_contract_rejects_non_lowercase_or_non_iso_language() -> None:
    with pytest.raises(ValidationError):
        final_record(language="ZH")
    with pytest.raises(ValidationError):
        final_record(language="zz")


def test_final_contract_accepts_valid_iso_languages() -> None:
    for language in ("bm", "de", "en", "zh"):
        assert final_record(language=language).language == language


def test_reject_reason_contract_is_exact() -> None:
    assert {reason.value for reason in RejectReason} == REJECT_CODES
    rejected = RejectedRecord(
        candidate_id="qc_1",
        stage="answer_matching",
        reason_code=RejectReason.ANSWER_MATCH_AMBIGUOUS,
        details={"top1": 0.996, "top2": 0.991},
        source_files=["paper.docx", "answer.docx"],
    )
    assert rejected.reason_code.value == "ANSWER_MATCH_AMBIGUOUS"
