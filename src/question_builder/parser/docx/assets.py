from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath

from question_builder.parser.docx.package import DocxPackage


@dataclass(frozen=True, slots=True)
class MaterializedAsset:
    asset_id: str
    sha256: str
    filename: str
    path: Path
    source_member: str


def materialize_asset(
    package: DocxPackage,
    member: str,
    asset_dir: Path,
) -> MaterializedAsset:
    content = package.read(member)
    digest = sha256(content).hexdigest()
    asset_id = f"img_{digest[:16]}"
    suffix = PurePosixPath(member).suffix.lower()
    filename = f"{digest[:16]}{suffix}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    target = asset_dir / filename
    if not target.exists():
        target.write_bytes(content)
    return MaterializedAsset(
        asset_id=asset_id,
        sha256=digest,
        filename=filename,
        path=target,
        source_member=member,
    )
