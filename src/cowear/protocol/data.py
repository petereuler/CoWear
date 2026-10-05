"""Manifest-first access to the final 363-session Hugging Face release."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROLES = ("mobile", "watch", "rokid")


def paper_role(storage_role: str) -> str:
    """Return the manuscript name for a stored dataset role."""
    from .. import PAPER_ROLE_FOR_STORAGE

    try:
        return PAPER_ROLE_FOR_STORAGE[storage_role]
    except KeyError as exc:
        raise ValueError(f"unknown storage role: {storage_role}") from exc


def storage_role(paper_name: str) -> str:
    """Return the Hugging Face/session-manifest name for a paper role."""
    from .. import STORAGE_ROLE_FOR_PAPER

    try:
        return STORAGE_ROLE_FOR_PAPER[paper_name]
    except KeyError as exc:
        raise ValueError(f"unknown paper role: {paper_name}") from exc


@dataclass(frozen=True)
class SessionRow:
    base_id: str
    split: str
    available_roles: tuple[str, ...]
    same_hand: str
    sample_quality: str


def read_manifest(path: Path) -> list[SessionRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CoWear manifest: {path}")
    output = []
    seen: set[str] = set()
    for row in rows:
        base_id = row.get("base_id", "").strip()
        if not base_id or base_id in seen:
            raise ValueError(f"invalid or duplicate base_id: {base_id!r}")
        seen.add(base_id)
        roles = tuple(role for role in row.get("available_roles", "").split(",") if role)
        if not set(roles).issubset(ROLES):
            raise ValueError(f"unknown device role in {base_id}: {roles}")
        output.append(SessionRow(base_id, row.get("split", ""), roles, row.get("same_hand", ""), row.get("sample_quality", "")))
    return output


def validate_paper_split(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in read_manifest(path):
        counts[row.split] = counts.get(row.split, 0) + 1
    expected = {"train": 217, "val": 31, "test": 115}
    if counts != expected:
        raise ValueError(f"expected split counts {expected}, got {counts}")
    return counts
