# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for check_card.py bootstrap rules and false positive guards."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
CHECK_CARD_PATH = SCRIPTS_DIR / "check_card.py"

spec = importlib.util.spec_from_file_location("check_card", CHECK_CARD_PATH)
assert spec and spec.loader
check_card: ModuleType = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_card)

check_dependencies = check_card.check_dependencies
check_spec_edits = check_card.check_spec_edits
check_line_patterns = check_card.check_line_patterns
collect_decisions = check_card.collect_decisions
is_implementation_source = check_card.is_implementation_source
is_checker_script = check_card.is_checker_script
lockfile_has_external_sources = check_card.lockfile_has_external_sources


def test_spec_edits_a01_allowed_when_pure_additions_and_claimed():
    failures = check_spec_edits(
        card="A0.1",
        spec_edits=["specs/spec-01-op-ir.md", "specs/spec-14-build-ci.md"],
        spec_statuses=[("A", "specs/spec-01-op-ir.md"), ("A", "specs/spec-14-build-ci.md")],
        deliverables_text="Cargo.toml workspace, specs/ copied in, DECISIONS.md",
    )
    assert failures == []


def test_spec_edits_a01_rejected_when_not_claimed_in_deliverables():
    failures = check_spec_edits(
        card="A0.1",
        spec_edits=["specs/spec-01-op-ir.md"],
        spec_statuses=[("A", "specs/spec-01-op-ir.md")],
        deliverables_text="Cargo.toml workspace, deny.toml",
    )
    assert len(failures) == 1
    assert "does not claim specs/ seeding" in failures[0]


def test_spec_edits_a01_rejected_when_modified():
    failures = check_spec_edits(
        card="A0.1",
        spec_edits=["specs/spec-01-op-ir.md"],
        spec_statuses=[("M", "specs/spec-01-op-ir.md")],
        deliverables_text="specs/ copied in",
    )
    assert len(failures) == 1
    assert "specs/ modified or deleted" in failures[0]


def test_spec_edits_a01_rejected_when_deleted():
    failures = check_spec_edits(
        card="A0.1",
        spec_edits=["specs/spec-01-op-ir.md"],
        spec_statuses=[("D", "specs/spec-01-op-ir.md")],
        deliverables_text="specs/ copied in",
    )
    assert len(failures) == 1
    assert "specs/ modified or deleted" in failures[0]


def test_spec_edits_non_a01_prohibits_any_spec_edits():
    failures = check_spec_edits(
        card="A0.2",
        spec_edits=["specs/spec-14-build-ci.md"],
        spec_statuses=[("A", "specs/spec-14-build-ci.md")],
        deliverables_text="specs/ copied in",
    )
    assert len(failures) == 1
    assert "specs/ edited:" in failures[0]


def test_dependencies_a01_none_allowed_without_external_sources():
    failures = check_dependencies(
        card="A0.1",
        files=["Cargo.lock", "crates/r9v-common/Cargo.toml"],
        new_deps_text="none",
        has_external_sources=False,
    )
    assert failures == []

    # Also test empty or "- none"
    assert check_dependencies("A0.1", ["Cargo.lock"], "- none", False) == []
    assert check_dependencies("A0.1", ["Cargo.lock"], "", False) == []


def test_dependencies_a01_none_rejected_with_external_sources():
    failures = check_dependencies(
        card="A0.1",
        files=["Cargo.lock"],
        new_deps_text="none",
        has_external_sources=True,
    )
    assert len(failures) == 1
    assert "lockfile changed but PR body's ## New dependencies is empty or 'none'" in failures[0]


def test_dependencies_non_a01_none_rejected_when_lockfile_changed():
    failures = check_dependencies(
        card="A0.2",
        files=["Cargo.lock"],
        new_deps_text="none",
        has_external_sources=False,
    )
    assert len(failures) == 1
    assert "lockfile changed but PR body's ## New dependencies is empty or 'none'" in failures[0]


def test_dependencies_with_justification_allowed():
    failures = check_dependencies(
        card="A0.2",
        files=["Cargo.lock"],
        new_deps_text="thiserror = '2.0': required for typed error mapping",
        has_external_sources=True,
    )
    assert failures == []


def test_lockfile_has_external_sources_detection(tmp_path):
    # Pure internal workspace lockfile
    clean_lock = tmp_path / "Cargo.lock"
    clean_lock.write_text(
        'version = 4\n\n[[package]]\nname = "r9v"\nversion = "0.1.0"\n\n[[package]]\nname = "r9v-common"\nversion = "0.1.0"\n'
    )
    assert not lockfile_has_external_sources(tmp_path, "Cargo.lock")

    # Lockfile with crates.io registry dependency
    registry_lock = tmp_path / "Cargo_reg.lock"
    registry_lock.write_text(
        'version = 4\n\n[[package]]\nname = "thiserror"\nversion = "2.0.0"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n'
    )
    assert lockfile_has_external_sources(tmp_path, "Cargo_reg.lock")

    # Lockfile with git dependency
    git_lock = tmp_path / "Cargo_git.lock"
    git_lock.write_text(
        'version = 4\n\n[[package]]\nname = "foo"\nversion = "0.1.0"\nsource = "git+https://github.com/example/foo"\n'
    )
    assert lockfile_has_external_sources(tmp_path, "Cargo_git.lock")


def test_checker_scripts_and_docs_excluded_from_implementation_sources():
    # Checker copies and its regression test file excluded
    assert is_checker_script("scripts/check_card.py")
    assert is_checker_script("skills/r9v-card-work/scripts/check_card.py")
    assert is_checker_script(".agents/skills/r9v-card-work/scripts/check_card.py")
    assert is_checker_script("tests/test_check_card.py")
    assert not is_implementation_source("scripts/check_card.py")
    assert not is_implementation_source("skills/r9v-card-work/scripts/check_card.py")
    assert not is_implementation_source(".agents/skills/r9v-card-work/scripts/check_card.py")
    assert not is_implementation_source("tests/test_check_card.py")

    # Other tests remain implementation sources for asm enforcement
    assert is_implementation_source("tests/test_r9v_cli.py")
    assert is_implementation_source("tests/test_release_contract.py")

    # Documentation excluded
    assert not is_implementation_source("skills/r9v-card-work/SKILL.md")
    assert not is_implementation_source(".agents/skills/r9v-engineering-standards/SKILL.md")
    assert not is_implementation_source("skills/r9v-card-work/references/pr-template.md")
    assert not is_implementation_source("DECISIONS.md")

    # Implementation sources included
    assert is_implementation_source("crates/r9v-sched/src/lib.rs")
    assert is_implementation_source("crates/r9v/src/main.rs")
    assert is_implementation_source("spikes/dot4-gemv/dot4_gemv.hip")
    assert is_implementation_source("tools/helper.py")


def test_checker_scripts_self_patterns_ignored_without_false_positives():
    checker_pattern_lines = [
        (16, '#   7. No TODO/todo!()/unimplemented!() without an owning card id.'),
        (17, '#   8. No thread_rng()/random() outside tests.'),
        (98, 'if re.search(r"\\basm!\\s*\\(|__asm__|\\basm\\s+volatile", t) and not f.startswith(ASM_OK):'),
        (101, 'if re.search(r"\\btodo!\\(|\\bunimplemented!\\(", t):'),
        (103, 'if re.search(r"(//|#)\\s*TODO\\b", t):'),
        (105, 'if re.search(r"thread_rng\\(\\)|\\brandom\\(\\)", t):'),
    ]
    for script_path in (
        "scripts/check_card.py",
        "skills/r9v-card-work/scripts/check_card.py",
        ".agents/skills/r9v-card-work/scripts/check_card.py",
    ):
        failures = check_line_patterns(script_path, checker_pattern_lines)
        assert failures == []


def test_markdown_documentation_decisions_ignored_without_false_positives():
    doc_entries = [
        ("skills/r9v-card-work/SKILL.md", [(69, "Example: DECISION(A2.8) in markdown body")]),
        (".agents/skills/r9v-engineering-standards/SKILL.md", [(291, "Use DECISION(A2.8) format")]),
        ("skills/r9v-card-work/references/pr-template.md", [(15, "<!-- DECISION(A2.8) -->")]),
    ]
    decisions = collect_decisions(doc_entries)
    assert decisions == []


def test_real_source_violations_caught():
    # Unsafe outside allowed crates
    f1 = check_line_patterns("crates/r9v-ir/src/lib.rs", [(10, "let x = unsafe { *ptr };")])
    assert len(f1) == 1
    assert "`unsafe` outside allowed crates" in f1[0]

    # Inline asm outside leaf
    f2 = check_line_patterns("crates/r9v-sched/src/lib.rs", [(20, 'asm!("nop")')])
    assert len(f2) == 1
    assert "inline asm outside r9v-kgen/src/leaf/" in f2[0]

    # Untagged stubs
    f3 = check_line_patterns("crates/r9v-loader/src/lib.rs", [(30, "todo!()")])
    assert len(f3) == 1
    assert "stub without a card id" in f3[0]

    # Untagged TODO
    f4 = check_line_patterns("crates/r9v-loader/src/lib.rs", [(35, "// TODO: implement parsing")])
    assert len(f4) == 1
    assert "TODO without a card id" in f4[0]

    # Unseeded randomness in python implementation
    f5 = check_line_patterns("tools/bench.py", [(40, "val = random()")])
    assert len(f5) == 1
    assert "unseeded randomness outside tests" in f5[0]

    # Tagged TODO / stub allowed
    f6 = check_line_patterns("crates/r9v-loader/src/lib.rs", [
        (50, "todo!(A0.1)"),
        (51, "// TODO(A0.1): finish staging ring"),
    ])
    assert f6 == []


def test_real_source_decisions_caught():
    source_entries = [
        ("crates/r9v-sched/src/lib.rs", [(12, "// DECISION(A0.1): unified step graph")]),
        ("spikes/dot4-gemv/dot4_gemv.hip", [(42, "// DECISION(A0.1): batch tile size")]),
    ]
    decisions = collect_decisions(source_entries)
    assert decisions == [
        ("crates/r9v-sched/src/lib.rs", 12, "A0.1"),
        ("spikes/dot4-gemv/dot4_gemv.hip", 42, "A0.1"),
    ]
