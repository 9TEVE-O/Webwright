from __future__ import annotations

import json
from pathlib import Path

import pytest

from webwright.lab.intake import (
    TaskTier,
    append_audit_log,
    build_audit_record,
    check_domain_allowlist,
    classify_task,
)


# ---------------------------------------------------------------------------
# classify_task
# ---------------------------------------------------------------------------


class TestClassifyTask:
    def test_generic_task_is_tier_1(self) -> None:
        assert classify_task("Check what AI tools shipped this week on musictech.net") == TaskTier.TIER_1

    def test_empty_string_is_tier_1(self) -> None:
        assert classify_task("") == TaskTier.TIER_1

    def test_artist_keyword_gives_tier_2(self) -> None:
        assert classify_task("Find the latest release for this artist") == TaskTier.TIER_2

    def test_client_keyword_gives_tier_2(self) -> None:
        assert classify_task("Research new plugin releases for my client") == TaskTier.TIER_2

    def test_musician_keyword_gives_tier_2(self) -> None:
        assert classify_task("Look up chart positions for this musician") == TaskTier.TIER_2

    def test_band_keyword_gives_tier_2(self) -> None:
        assert classify_task("Find press coverage for the band") == TaskTier.TIER_2

    def test_performer_keyword_gives_tier_2(self) -> None:
        assert classify_task("Check streaming numbers for the performer") == TaskTier.TIER_2

    def test_roster_keyword_gives_tier_2(self) -> None:
        assert classify_task("Compile a report on our roster") == TaskTier.TIER_2

    def test_send_keyword_gives_tier_3(self) -> None:
        assert classify_task("Find the new releases and send them to me") == TaskTier.TIER_3

    def test_tell_keyword_gives_tier_3(self) -> None:
        assert classify_task("Tell the team what was released this week") == TaskTier.TIER_3

    def test_brief_keyword_gives_tier_3(self) -> None:
        assert classify_task("Brief the artist on new competition") == TaskTier.TIER_3

    def test_deliver_to_gives_tier_3(self) -> None:
        assert classify_task("Gather data and deliver to the client") == TaskTier.TIER_3

    def test_present_to_gives_tier_3(self) -> None:
        assert classify_task("Compile results and present to stakeholders") == TaskTier.TIER_3

    def test_report_to_gives_tier_3(self) -> None:
        assert classify_task("Find insights and report to management") == TaskTier.TIER_3

    def test_share_with_gives_tier_3(self) -> None:
        assert classify_task("Find the data and share with the team") == TaskTier.TIER_3

    def test_tier_3_wins_over_tier_2_when_both_match(self) -> None:
        # "artist" matches TIER_2, "send" matches TIER_3; TIER_3 should win
        assert classify_task("Find info about this artist and send it over") == TaskTier.TIER_3

    def test_tier_2_pattern_is_case_insensitive(self) -> None:
        assert classify_task("Research this ARTIST") == TaskTier.TIER_2
        assert classify_task("Research this Artist") == TaskTier.TIER_2

    def test_tier_3_pattern_is_case_insensitive(self) -> None:
        assert classify_task("SEND the report") == TaskTier.TIER_3
        assert classify_task("Send the report") == TaskTier.TIER_3

    def test_my_client_phrase_gives_tier_2(self) -> None:
        assert classify_task("Compile streaming data for my client") == TaskTier.TIER_2

    def test_tier_values_are_correct(self) -> None:
        assert TaskTier.TIER_1 == 1
        assert TaskTier.TIER_2 == 2
        assert TaskTier.TIER_3 == 3

    def test_tier_ordering(self) -> None:
        assert TaskTier.TIER_1 < TaskTier.TIER_2 < TaskTier.TIER_3


# ---------------------------------------------------------------------------
# check_domain_allowlist
# ---------------------------------------------------------------------------


ALLOWLIST = ["musictech.net", "billboard.com", "youtube.com"]


class TestCheckDomainAllowlist:
    def test_empty_code_returns_none(self) -> None:
        assert check_domain_allowlist("", ALLOWLIST) is None

    def test_empty_allowlist_returns_none(self) -> None:
        code = "page.goto('https://evil.com')"
        assert check_domain_allowlist(code, []) is None

    def test_none_code_returns_none(self) -> None:
        assert check_domain_allowlist(None, ALLOWLIST) is None  # type: ignore[arg-type]

    def test_code_with_no_goto_returns_none(self) -> None:
        code = "x = 1 + 1\nprint(x)"
        assert check_domain_allowlist(code, ALLOWLIST) is None

    def test_allowed_domain_returns_none(self) -> None:
        code = 'page.goto("https://musictech.net/news")'
        assert check_domain_allowlist(code, ALLOWLIST) is None

    def test_blocked_domain_returns_violation_string(self) -> None:
        code = 'page.goto("https://evil.com/steal")'
        result = check_domain_allowlist(code, ALLOWLIST)
        assert result is not None
        assert "evil.com" in result
        assert "LAB allowlist" in result

    def test_violation_includes_full_url(self) -> None:
        url = "https://notallowed.org/path"
        code = f'page.goto("{url}")'
        result = check_domain_allowlist(code, ALLOWLIST)
        assert result is not None
        assert url in result

    def test_www_prefix_stripped_from_url(self) -> None:
        # www.musictech.net should match allowlist entry musictech.net
        code = 'page.goto("https://www.musictech.net/article")'
        assert check_domain_allowlist(code, ALLOWLIST) is None

    def test_www_prefix_stripped_from_allowlist(self) -> None:
        allowlist_with_www = ["www.musictech.net"]
        code = 'page.goto("https://musictech.net/article")'
        assert check_domain_allowlist(code, allowlist_with_www) is None

    def test_subdomain_of_allowed_domain_passes(self) -> None:
        # sub.musictech.net should be allowed (endswith(".musictech.net"))
        code = 'page.goto("https://sub.musictech.net/page")'
        assert check_domain_allowlist(code, ALLOWLIST) is None

    def test_subdomain_of_blocked_domain_is_blocked(self) -> None:
        code = 'page.goto("https://sub.evil.com/page")'
        result = check_domain_allowlist(code, ALLOWLIST)
        assert result is not None
        assert "evil.com" in result or "sub.evil.com" in result

    def test_page_goto_syntax(self) -> None:
        code = 'page.goto("https://musictech.net")'
        assert check_domain_allowlist(code, ALLOWLIST) is None

    def test_bare_goto_syntax(self) -> None:
        code = 'goto("https://musictech.net")'
        assert check_domain_allowlist(code, ALLOWLIST) is None

    def test_single_quote_urls(self) -> None:
        code = "page.goto('https://musictech.net')"
        assert check_domain_allowlist(code, ALLOWLIST) is None

    def test_double_quote_urls(self) -> None:
        code = 'page.goto("https://musictech.net")'
        assert check_domain_allowlist(code, ALLOWLIST) is None

    def test_first_violation_returned_for_multiple_gotos(self) -> None:
        code = (
            'page.goto("https://evil1.com")\n'
            'page.goto("https://evil2.com")\n'
        )
        result = check_domain_allowlist(code, ALLOWLIST)
        assert result is not None
        assert "evil1.com" in result

    def test_all_gotos_allowed_returns_none(self) -> None:
        code = (
            'page.goto("https://musictech.net/a")\n'
            'page.goto("https://billboard.com/b")\n'
        )
        assert check_domain_allowlist(code, ALLOWLIST) is None

    def test_mixed_allowed_and_blocked_returns_first_blocked(self) -> None:
        code = (
            'page.goto("https://musictech.net/ok")\n'
            'page.goto("https://blocked.example.com/bad")\n'
        )
        result = check_domain_allowlist(code, ALLOWLIST)
        assert result is not None
        assert "blocked.example.com" in result

    def test_partial_domain_name_does_not_bypass(self) -> None:
        # "notmusictech.net" should NOT match "musictech.net"
        code = 'page.goto("https://notmusictech.net/page")'
        result = check_domain_allowlist(code, ALLOWLIST)
        assert result is not None


# ---------------------------------------------------------------------------
# build_audit_record
# ---------------------------------------------------------------------------


class TestBuildAuditRecord:
    def test_record_has_lab_audit_key(self) -> None:
        record = build_audit_record("some task", TaskTier.TIER_1)
        assert "lab_audit" in record

    def test_tier_1_record_fields(self) -> None:
        record = build_audit_record("research task", TaskTier.TIER_1, task_id="t123")
        audit = record["lab_audit"]
        assert audit["task_text"] == "research task"
        assert audit["task_id"] == "t123"
        assert audit["tier"] == 1
        assert audit["tier_name"] == "TIER_1"
        assert audit["flags"] == []
        assert audit["domains_visited"] == []
        assert audit["redactions"] == []
        assert audit["gate_verdicts"] == []

    def test_tier_2_record_has_identity_flag(self) -> None:
        record = build_audit_record("artist task", TaskTier.TIER_2)
        assert "IDENTITY_IN_REQUEST" in record["lab_audit"]["flags"]

    def test_tier_3_record_has_all_flags(self) -> None:
        record = build_audit_record("delivery task", TaskTier.TIER_3)
        flags = record["lab_audit"]["flags"]
        assert "IDENTITY_IN_REQUEST" in flags
        assert "ARTIST_FACING_OUTPUT" in flags
        assert "CITATION_REQUIRED" in flags

    def test_task_id_defaults_to_empty_string(self) -> None:
        record = build_audit_record("task", TaskTier.TIER_1)
        assert record["lab_audit"]["task_id"] == ""

    def test_task_id_none_becomes_empty_string(self) -> None:
        record = build_audit_record("task", TaskTier.TIER_1, task_id=None)
        assert record["lab_audit"]["task_id"] == ""

    def test_timestamp_is_iso_format(self) -> None:
        from datetime import datetime

        record = build_audit_record("task", TaskTier.TIER_1)
        ts = record["lab_audit"]["timestamp"]
        # Should parse without raising
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_timestamp_includes_timezone_info(self) -> None:
        record = build_audit_record("task", TaskTier.TIER_1)
        ts = record["lab_audit"]["timestamp"]
        # UTC offset should be present (+00:00 or Z)
        assert "+" in ts or ts.endswith("Z")

    def test_tier_2_record_does_not_have_citation_flag(self) -> None:
        record = build_audit_record("artist task", TaskTier.TIER_2)
        flags = record["lab_audit"]["flags"]
        assert "CITATION_REQUIRED" not in flags
        assert "ARTIST_FACING_OUTPUT" not in flags


# ---------------------------------------------------------------------------
# append_audit_log
# ---------------------------------------------------------------------------


class TestAppendAuditLog:
    def test_no_op_when_log_path_is_none(self, tmp_path: Path) -> None:
        # Should not raise and should not create any file
        record = {"lab_audit": {"task_text": "test"}}
        append_audit_log(None, record)
        assert list(tmp_path.iterdir()) == []

    def test_no_op_when_log_path_is_empty_string(self, tmp_path: Path) -> None:
        record = {"lab_audit": {"task_text": "test"}}
        append_audit_log("", record)
        assert list(tmp_path.iterdir()) == []

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nested" / "deep" / "audit.jsonl"
        record = {"lab_audit": {"task_text": "test"}}
        append_audit_log(log_path, record)
        assert log_path.exists()

    def test_writes_valid_json_line(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        record = {"lab_audit": {"task_text": "research", "tier": 1}}
        append_audit_log(log_path, record)
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed == record

    def test_appends_multiple_records(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        record1 = {"lab_audit": {"task_text": "task one", "tier": 1}}
        record2 = {"lab_audit": {"task_text": "task two", "tier": 2}}
        append_audit_log(log_path, record1)
        append_audit_log(log_path, record2)
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == record1
        assert json.loads(lines[1]) == record2

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        log_path = str(tmp_path / "audit.jsonl")
        record = {"lab_audit": {"task_text": "test"}}
        append_audit_log(log_path, record)
        assert Path(log_path).exists()

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        record = {"lab_audit": {"task_text": "test"}}
        append_audit_log(log_path, record)
        assert log_path.exists()

    def test_each_line_ends_with_newline(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        record = {"lab_audit": {"task_text": "test"}}
        append_audit_log(log_path, record)
        content = log_path.read_text(encoding="utf-8")
        assert content.endswith("\n")

    def test_unicode_in_record_is_preserved(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        record = {"lab_audit": {"task_text": "Ünïcödé artist: 音楽"}}
        append_audit_log(log_path, record)
        lines = log_path.read_text(encoding="utf-8").splitlines()
        parsed = json.loads(lines[0])
        assert parsed["lab_audit"]["task_text"] == "Ünïcödé artist: 音楽"

    def test_existing_file_is_appended_not_overwritten(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        # Pre-populate the file
        log_path.write_text(json.dumps({"pre": "existing"}) + "\n", encoding="utf-8")
        record = {"lab_audit": {"task_text": "new"}}
        append_audit_log(log_path, record)
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"pre": "existing"}
        assert json.loads(lines[1]) == record


# ---------------------------------------------------------------------------
# __init__.py re-exports
# ---------------------------------------------------------------------------


class TestLabPackageExports:
    def test_all_symbols_importable_from_package(self) -> None:
        from webwright.lab import (  # noqa: F401
            TaskTier,
            append_audit_log,
            build_audit_record,
            check_domain_allowlist,
            classify_task,
        )

    def test_task_tier_from_package_is_same_class(self) -> None:
        from webwright.lab import TaskTier as PackageTier
        from webwright.lab.intake import TaskTier as IntakeTier

        assert PackageTier is IntakeTier
