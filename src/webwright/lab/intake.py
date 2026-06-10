from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class TaskTier(IntEnum):
    TIER_1 = 1  # Public research, no artist identity
    TIER_2 = 2  # Artist-adjacent, identity present in request
    TIER_3 = 3  # Artist-facing output, accuracy and citation required


_TIER_2_PATTERNS = re.compile(
    r"\b(artist|client|roster|musician|band|performer|their\s+artist|my\s+client)\b",
    re.IGNORECASE,
)
_TIER_3_PATTERNS = re.compile(
    r"\b(send|report\s+to|tell|brief|share\s+with|deliver\s+to|present\s+to)\b",
    re.IGNORECASE,
)
_GOTO_PATTERN = re.compile(
    r"""(?:page\.goto|goto)\s*\(\s*[\"'](https?://[^\"']+)[\"']"""
)


def classify_task(task: str) -> TaskTier:
    """Classify a task string into a governance tier.

    Tier 3 wins over Tier 2 if both patterns match.
    Default is Tier 1 (public research, no special handling).
    """
    if _TIER_3_PATTERNS.search(task):
        return TaskTier.TIER_3
    if _TIER_2_PATTERNS.search(task):
        return TaskTier.TIER_2
    return TaskTier.TIER_1


def check_domain_allowlist(code: str, allowlist: list[str]) -> str | None:
    """Scan generated code for goto() calls and block non-allowlisted domains.

    Returns a violation string if a domain is blocked, None if clean.
    Only consulted when the LAB config has a non-empty allowlist.
    """
    if not code or not allowlist:
        return None
    normalised_allowlist = [h.removeprefix("www.") for h in allowlist]
    for url in _GOTO_PATTERN.findall(code):
        host = (urlparse(url).hostname or "").removeprefix("www.")
        if not any(
            host == allowed or host.endswith("." + allowed)
            for allowed in normalised_allowlist
        ):
            return f"domain not in LAB allowlist: {host} (from {url})"
    return None


def build_audit_record(
    task: str,
    tier: TaskTier,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Build the initial audit record written at intake time."""
    return {
        "lab_audit": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id or "",
            "task_text": task,
            "tier": tier.value,
            "tier_name": tier.name,
            "flags": _tier_flags(tier),
            "domains_visited": [],
            "redactions": [],
            "gate_verdicts": [],
        }
    }


def _tier_flags(tier: TaskTier) -> list[str]:
    flags: list[str] = []
    if tier >= TaskTier.TIER_2:
        flags.append("IDENTITY_IN_REQUEST")
    if tier >= TaskTier.TIER_3:
        flags.append("ARTIST_FACING_OUTPUT")
        flags.append("CITATION_REQUIRED")
    return flags


def append_audit_log(log_path: str | Path | None, record: dict[str, Any]) -> None:
    """Append one JSON line to the LAB audit log."""
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
