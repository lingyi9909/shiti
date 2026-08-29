from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from zipfile import BadZipFile, ZipFile


class DocxPackageError(ValueError):
    """Raised when a source cannot be inspected as the required DOCX package."""


@dataclass(frozen=True, slots=True)
class DocxPackage:
    path: Path
    source_sha256: str
    members: frozenset[str]

    @classmethod
    def open(cls, path: Path) -> DocxPackage:
        try:
            source_bytes = path.read_bytes()
            with ZipFile(path) as archive:
                members = frozenset(archive.namelist())
        except (BadZipFile, OSError) as exc:
            raise DocxPackageError(f"{path} is not a valid DOCX ZIP") from exc

        if "word/document.xml" not in members:
            raise DocxPackageError("DOCX package is missing mandatory word/document.xml")

        return cls(path=path, source_sha256=sha256(source_bytes).hexdigest(), members=members)

    def read(self, member: str) -> bytes:
        if member not in self.members:
            raise DocxPackageError(f"DOCX package member is missing: {member}")
        try:
            with ZipFile(self.path) as archive:
                return archive.read(member)
        except (BadZipFile, OSError, KeyError) as exc:
            raise DocxPackageError(f"cannot read DOCX package member: {member}") from exc
