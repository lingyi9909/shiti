from __future__ import annotations

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.matching import MatchedQuestion, MatchEvidence
from question_builder.domain.question import QuestionCandidate
from question_builder.storage.workspace import WorkspaceStore


def _document() -> DocumentIR:
    return DocumentIR(
        document_id="doc-1",
        source_file="questions.docx",
        source_sha256="a" * 64,
        blocks=(
            ContentBlock(
                block_id="q1",
                order=1,
                type="paragraph",
                raw_text="1. 选择正确答案",
                metadata={"source": "word"},
            ),
        ),
    )


def _question() -> QuestionCandidate:
    return QuestionCandidate(
        question_candidate_id="qc-1",
        document_id="doc-1",
        content_blocks=("q1",),
        question_number="1",
        question_type_candidate="选择题",
        split_score=0.999,
    )


def _answer() -> AnswerCandidate:
    return AnswerCandidate(
        answer_candidate_id="ac-1",
        document_id="answer-doc",
        question_number="1",
        answer="A",
        analysis="原文解析",
        source_blocks=("a1",),
        extract_score=0.999,
    )


def _matched() -> MatchedQuestion:
    question = _question()
    answer = _answer()
    return MatchedQuestion(
        question=question,
        answer=answer,
        evidence=MatchEvidence(
            question_candidate_id=question.question_candidate_id,
            answer_candidate_id=answer.answer_candidate_id,
            match_score=0.999,
            second_best_score=0.5,
            verifier_score=0.999,
            question_source_blocks=question.content_blocks,
            answer_source_blocks=answer.source_blocks,
            evidence={"same_cluster": True, "verifier_decision": "PASS"},
        ),
    )


def test_workspace_round_trips_document_candidate_answer_and_match_evidence(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "workspace")
    document = _document()
    question = _question()
    answer = _answer()
    matched = _matched()

    store.write_document(document)
    store.write_question_candidate(question)
    store.write_answer_candidate(answer)
    store.write_matched_question(matched)

    assert store.read_document(document.document_id) == document
    assert store.read_question_candidate(question.question_candidate_id) == question
    assert store.read_answer_candidate(answer.answer_candidate_id) == answer
    assert store.read_matched_question(question.question_candidate_id) == matched


def test_workspace_writes_leave_only_complete_json_artifacts(tmp_path) -> None:
    root = tmp_path / "workspace"
    store = WorkspaceStore(root)

    store.write_document(_document())
    store.write_question_candidate(_question())
    store.write_answer_candidate(_answer())
    store.write_matched_question(_matched())

    json_files = sorted(root.rglob("*.json"))
    assert len(json_files) == 4
    assert not list(root.rglob("*.tmp"))
    assert all(path.read_text(encoding="utf-8").strip().startswith("{") for path in json_files)


def test_workspace_identity_cannot_escape_workspace_root(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "workspace")
    question = QuestionCandidate(
        question_candidate_id="../../outside",
        document_id="doc-1",
        content_blocks=("q1",),
        question_number="1",
        question_type_candidate="选择题",
        split_score=0.999,
    )

    store.write_question_candidate(question)

    assert store.read_question_candidate("../../outside") == question
    assert not (tmp_path / "outside.json").exists()
