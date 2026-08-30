from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from question_builder.understanding.classifier import (
    DocumentClass,
    DocumentUnderstanding,
)


@dataclass(frozen=True, slots=True)
class ExamCluster:
    cluster_id: str
    document_ids: tuple[str, ...]
    accepted: bool
    evidence: tuple[str, ...]


_QUESTION_CLASSES = {DocumentClass.QUESTION}
_ANSWER_CLASSES = {DocumentClass.ANSWER}


def _cluster_id(document_ids: tuple[str, ...]) -> str:
    material = "\n".join(sorted(document_ids)).encode("utf-8")
    return "cluster_" + hashlib.sha256(material).hexdigest()[:16]


def _normalized_family(value: str | None) -> str:
    if not value:
        return ""
    text = Path(value).stem
    for token in ("参考答案", "答案", "含答案", "试卷", "试题", "解析"):
        text = text.replace(token, "")
    return re.sub(r"[\s_\-—（）()【】\[\]]+", "", text).lower()


def _hard_conflict(left: DocumentUnderstanding, right: DocumentUnderstanding) -> bool:
    for attribute in ("subject", "year", "grade"):
        left_value = getattr(left.features, attribute)
        right_value = getattr(right.features, attribute)
        if left_value and right_value and left_value != right_value:
            return True
    return False


def _role_pair(left: DocumentUnderstanding, right: DocumentUnderstanding) -> bool:
    return (
        left.document_class in _QUESTION_CLASSES
        and right.document_class in _ANSWER_CLASSES
    ) or (
        right.document_class in _QUESTION_CLASSES
        and left.document_class in _ANSWER_CLASSES
    )


def _question_answer_pair(
    left: DocumentUnderstanding,
    right: DocumentUnderstanding,
) -> tuple[DocumentUnderstanding, DocumentUnderstanding]:
    if left.document_class in _QUESTION_CLASSES:
        return left, right
    return right, left


def _number_sequence_compatible(
    question: DocumentUnderstanding,
    answer: DocumentUnderstanding,
) -> bool:
    question_numbers = question.features.question_number_sequence
    answer_numbers = answer.features.answer_number_sequence
    if not question_numbers or not answer_numbers:
        return False
    overlap = set(question_numbers) & set(answer_numbers)
    minimum = min(len(question_numbers), len(answer_numbers))
    return len(overlap) == minimum


def _pair_evidence(
    left: DocumentUnderstanding,
    right: DocumentUnderstanding,
) -> tuple[str, ...] | None:
    if _hard_conflict(left, right) or not _role_pair(left, right):
        return None

    question, answer = _question_answer_pair(left, right)
    if not _number_sequence_compatible(question, answer):
        return None

    evidence: list[str] = ["number_sequence_compatible"]
    metadata_matches = 0
    for attribute in ("subject", "grade", "year", "city", "exam_type"):
        left_value = getattr(left.features, attribute)
        right_value = getattr(right.features, attribute)
        if left_value and right_value and left_value == right_value:
            metadata_matches += 1
            evidence.append(f"{attribute}_match")

    left_family = _normalized_family(left.source_file)
    right_family = _normalized_family(right.source_file)
    if left_family and left_family == right_family:
        evidence.append("filename_family_match")

    left_title = _normalized_family(left.features.title)
    right_title = _normalized_family(right.features.title)
    if left_title and left_title == right_title:
        evidence.append("title_family_match")

    strong_family = "filename_family_match" in evidence or "title_family_match" in evidence
    if metadata_matches >= 3 or (strong_family and metadata_matches >= 2):
        return tuple(evidence)
    return None


def _can_join_cluster(
    candidate: DocumentUnderstanding,
    members: list[DocumentUnderstanding],
) -> tuple[str, ...] | None:
    pair_evidence: list[str] = []
    found_strong_pair = False
    for member in members:
        if _hard_conflict(candidate, member):
            return None
        candidate_evidence = _pair_evidence(candidate, member)
        if candidate_evidence is not None:
            found_strong_pair = True
            pair_evidence.extend(candidate_evidence)
    if not found_strong_pair:
        return None
    return tuple(dict.fromkeys(pair_evidence))


def build_exam_clusters(
    understandings: tuple[DocumentUnderstanding, ...],
) -> tuple[ExamCluster, ...]:
    groups: list[list[DocumentUnderstanding]] = []
    group_evidence: list[list[str]] = []

    for understanding in understandings:
        joined = False
        for index, members in enumerate(groups):
            candidate_evidence = _can_join_cluster(understanding, members)
            if candidate_evidence is None:
                continue
            members.append(understanding)
            group_evidence[index].extend(candidate_evidence)
            joined = True
            break
        if not joined:
            groups.append([understanding])
            group_evidence.append(["singleton_conservative"])

    clusters: list[ExamCluster] = []
    for members, cluster_evidence in zip(groups, group_evidence, strict=True):
        document_ids = tuple(member.document_id for member in members)
        normalized_evidence = list(dict.fromkeys(cluster_evidence))
        if len(document_ids) > 1:
            normalized_evidence = [
                item for item in normalized_evidence if item != "singleton_conservative"
            ]
        clusters.append(
            ExamCluster(
                cluster_id=_cluster_id(document_ids),
                document_ids=document_ids,
                accepted=True,
                evidence=tuple(normalized_evidence),
            )
        )

    return tuple(clusters)
