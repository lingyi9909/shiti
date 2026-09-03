from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from question_builder.domain.document import DomainModel

Course = Literal[
    "语文",
    "数学",
    "英语",
    "科学",
    "政治",
    "历史",
    "地理",
    "物理",
    "化学",
    "生物",
    "未知",
]
GradeLevel = Literal[
    "小学一年级",
    "小学二年级",
    "小学三年级",
    "小学四年级",
    "小学五年级",
    "小学六年级",
    "初中一年级",
    "初中二年级",
    "初中三年级",
    "高中一年级",
    "高中二年级",
    "高中三年级",
    "未知",
]
Grade = Literal["小学", "初中", "高中", "未知"]
EntranceExamType = Literal["小升初", "中考", "高考", "未知"]
QuestionType = Literal["判断题", "选择题", "填空题", "问答题", "其他题型", "未知"]

_ISO_639_1_CODES: frozenset[str] = frozenset(
    """
    aa ab ae af ak am an ar as av ay az ba be bg bi
    bm bn bo br bs ca ce ch co cr cs cu cv cy da de
    dv dz ee el en eo es et eu fa ff fi fj fo fr fy
    ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia
    id ie ig ii ik io is it iu ja jv ka kg ki kj kk
    kl km kn ko kr ks ku kv kw ky la lb lg li ln lo
    lt lu lv mg mh mi mk ml mn mr ms mt my na nb nd
    ne ng nl nn no nr nv ny oc oj om or os pa pi pl
    ps pt qu rm rn ro ru rw sa sc sd se sg sh si sk
    sl sm sn so sq sr ss st su sv sw ta te tg th ti
    tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo
    wa wo xh yi yo za zh zu
    """.split()
)

_GRADE_PREFIX: dict[str, str] = {
    "小学一年级": "小学",
    "小学二年级": "小学",
    "小学三年级": "小学",
    "小学四年级": "小学",
    "小学五年级": "小学",
    "小学六年级": "小学",
    "初中一年级": "初中",
    "初中二年级": "初中",
    "初中三年级": "初中",
    "高中一年级": "高中",
    "高中二年级": "高中",
    "高中三年级": "高中",
    "未知": "未知",
}


class FinalQuestionRecord(DomainModel):
    text_question: str = Field(min_length=1)
    is_pic_included: Literal[0, 1]
    text_answer: str = Field(min_length=1)
    answer_analysis: str = Field(min_length=1)
    text_course: Course
    text_grade_level: GradeLevel
    text_grade: Grade
    knowledge_points: str = ""
    exam_points: str = ""
    publisher: str = ""
    text_paper: str = Field(min_length=1)
    textbook_version: str = ""
    static_info: str
    language: str
    text_year: str = ""
    entrance_exam_type: EntranceExamType
    text_city: str = ""
    question_type: QuestionType
    competition_event: str = ""

    @field_validator("is_pic_included", mode="before")
    @classmethod
    def validate_picture_flag(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            raise ValueError("is_pic_included must be integer 0 or 1")
        return value

    @field_validator("static_info", mode="before")
    @classmethod
    def validate_static_info(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("static_info must be a JSON object serialized as a string")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("static_info must contain valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("static_info JSON must be an object")
        required_keys = {
            "slim_question_md5",
            "copyright",
            "source_question_blocks",
            "source_answer_blocks",
        }
        missing = required_keys - parsed.keys()
        if missing:
            raise ValueError(f"static_info missing required provenance keys: {sorted(missing)}")
        slim_question_md5 = parsed["slim_question_md5"]
        if (
            not isinstance(slim_question_md5, str)
            or re.fullmatch(r"[0-9a-f]{32}", slim_question_md5) is None
        ):
            raise ValueError("static_info slim_question_md5 must be 32 lowercase hex characters")
        if parsed["copyright"] not in ("0", "1") or not isinstance(parsed["copyright"], str):
            raise ValueError('static_info copyright must be string "0" or "1"')
        for key in ("source_question_blocks", "source_answer_blocks"):
            block_ids = parsed[key]
            if (
                not isinstance(block_ids, list)
                or not block_ids
                or any(not isinstance(item, str) or not item for item in block_ids)
            ):
                raise ValueError(f"static_info {key} must be a non-empty list of block ids")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in _ISO_639_1_CODES:
            raise ValueError("language must be a lowercase ISO 639-1 code")
        return value

    @field_validator("text_year")
    @classmethod
    def validate_text_year(cls, value: str) -> str:
        if value and re.fullmatch(r"[0-9]{4}", value) is None:
            raise ValueError("text_year must be empty or a four-digit year string")
        return value

    @model_validator(mode="after")
    def validate_grade_consistency(self) -> FinalQuestionRecord:
        expected_grade = _GRADE_PREFIX[self.text_grade_level]
        if self.text_grade != expected_grade:
            raise ValueError("text_grade must be consistent with text_grade_level")
        return self
