from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.document import DocumentIR
from question_builder.domain.matching import MatchedQuestion
from question_builder.domain.question import QuestionCandidate


class WorkspaceStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write_document(self, document: DocumentIR) -> Path:
        path = self._artifact_path("documents", document.document_id)
        self._atomic_write(path, document.model_dump_json())
        return path

    def read_document(self, document_id: str) -> DocumentIR:
        path = self._artifact_path("documents", document_id)
        return DocumentIR.model_validate_json(path.read_text(encoding="utf-8"))

    def write_question_candidate(self, question: QuestionCandidate) -> Path:
        path = self._artifact_path("candidates", question.question_candidate_id)
        self._atomic_write(path, question.model_dump_json())
        return path

    def read_question_candidate(self, question_candidate_id: str) -> QuestionCandidate:
        path = self._artifact_path("candidates", question_candidate_id)
        return QuestionCandidate.model_validate_json(path.read_text(encoding="utf-8"))

    def write_answer_candidate(self, answer: AnswerCandidate) -> Path:
        path = self._artifact_path("answers", answer.answer_candidate_id)
        self._atomic_write(path, answer.model_dump_json())
        return path

    def read_answer_candidate(self, answer_candidate_id: str) -> AnswerCandidate:
        path = self._artifact_path("answers", answer_candidate_id)
        return AnswerCandidate.model_validate_json(path.read_text(encoding="utf-8"))

    def write_matched_question(self, matched: MatchedQuestion) -> Path:
        path = self._artifact_path("matches", matched.question.question_candidate_id)
        self._atomic_write(path, matched.model_dump_json())
        return path

    def read_matched_question(self, question_candidate_id: str) -> MatchedQuestion:
        path = self._artifact_path("matches", question_candidate_id)
        return MatchedQuestion.model_validate_json(path.read_text(encoding="utf-8"))

    def _artifact_path(self, category: str, identity: str) -> Path:
        if not identity:
            raise ValueError("artifact identity must be non-empty")
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.root / category / f"{digest}.json"

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
