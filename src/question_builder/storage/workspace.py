from __future__ import annotations

from pathlib import Path

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.document import DocumentIR
from question_builder.domain.matching import MatchedQuestion
from question_builder.domain.question import QuestionCandidate


class WorkspaceStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write_document(self, document: DocumentIR) -> Path:
        raise NotImplementedError

    def read_document(self, document_id: str) -> DocumentIR:
        raise NotImplementedError

    def write_question_candidate(self, question: QuestionCandidate) -> Path:
        raise NotImplementedError

    def read_question_candidate(self, question_candidate_id: str) -> QuestionCandidate:
        raise NotImplementedError

    def write_answer_candidate(self, answer: AnswerCandidate) -> Path:
        raise NotImplementedError

    def read_answer_candidate(self, answer_candidate_id: str) -> AnswerCandidate:
        raise NotImplementedError

    def write_matched_question(self, matched: MatchedQuestion) -> Path:
        raise NotImplementedError

    def read_matched_question(self, question_candidate_id: str) -> MatchedQuestion:
        raise NotImplementedError
