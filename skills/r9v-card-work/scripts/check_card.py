#!/usr/bin/env python3
"""Mechanical checks for an R9V card PR.

Usage:
    python scripts/check_card.py --card A3.4 --base main --pr-body pr.md [--repo .]

Checks (each prints PASS/FAIL and a reason):
  1. No files under specs/ changed (for A0.1 only: pure addition allowed if claimed in Deliverables).
  2. `unsafe` only in crates/r9v-hip and crates/r9v-t0 SIMD modules; inline asm only in crates/r9v-kgen/src/leaf/.
  3. Generated or measured artifacts not hand-edited unless the PR body claims ownership
     (kernels/gen, tune, bench/baselines, docs/config.md, SUPPORT.md, support/).
  4. Every DECISION(<card>) comment in the diff appears in the PR body's "## Decisions" section.
  5. Every commit since base starts with "<card>:" and carries a Signed-off-by trailer.
  6. Cargo.lock changes are matched by a "## New dependencies" section that is not "none"
     (for A0.1 only: "none" allowed when Cargo.lock has no registry/git source entries).
  7. No TODO/todo!()/unimplemented!() without an owning card id.
  8. No thread_rng()/random() outside tests.

Exit code 0 when everything passes, 1 otherwise. This script checks rules, not judgment;
the acceptance checklist covers the rest.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

GENERATED = ("kernels/gen/", "tune/", "bench/baselines/", "docs/config.md", "SUPPORT.md", "support/")
UNSAFE_OK = ("crates/r9v-hip/", "crates/r9v-t0/src/simd/")
ASM_OK = ("crates/r9v-kgen/src/leaf/",)
SOURCE_EXTS = (".rs", ".hip", ".cpp", ".h", ".cu", ".py")
CARD_RE = re.compile(r"^[A-D]\d+\.(?:S)?\d+[a-z]?$")


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def changed_files(repo, base):
    return [f for f in git(repo, "diff", "--name-only", f"{base}...HEAD").splitlines() if f]


def git_spec_statuses(repo, base):
    out = git(repo, "diff", "--name-status", f"{base}...HEAD", "--", "specs")
    statuses = []
    for line in out.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            status, path = parts[0], parts[1]
            if status.startswith("R") and "\t" in path:
                path = path.split("\t")[1]
            statuses.append((status, path))
    return statuses


def lockfile_has_external_sources(repo, path="Cargo.lock"):
    lock_path = Path(repo) / path
    if not lock_path.exists():
        return False
    content = lock_path.read_text()
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("source = "):
            src = line.split("=", 1)[1].strip().strip('"')
            if src.startswith("registry+") or src.startswith("git+"):
                return True
    return False


def is_checker_script(path):
    p = Path(path)
    return (
        p.name == "check_card.py"
        or path.endswith("check_card.py")
        or path == "tests/test_check_card.py"
        or p.name == "test_check_card.py"
        or path.endswith("test_check_card.py")
    )


def is_implementation_source(path):
    """Return True if path is an implementation source file for content/decision scans.

    Excludes the card checker scripts and non-code documentation files.
    """
    if not path.endswith(SOURCE_EXTS):
        return False
    if is_checker_script(path):
        return False
    return True


def check_spec_edits(card, spec_edits, spec_statuses, deliverables_text):
    """Validate specs/ edits according to card rules.

    For card A0.1 only, permit specs/ files only when every spec change is an addition
    (never modification/deletion) and the PR Deliverables section claims specs/ seeding.
    For every other card, any edit under specs/ is prohibited.
    """
    failures = []
    if not spec_edits:
        return failures
    if card == "A0.1":
        claims_specs = "specs" in deliverables_text.lower()
        if not claims_specs:
            failures.append("specs/ files added but PR body's ## Deliverables does not claim specs/ seeding")
        non_additions = [path for status, path in spec_statuses if status != "A"]
        if non_additions:
            failures.append(f"specs/ modified or deleted (only additions permitted for A0.1): {', '.join(non_additions)}")
    else:
        failures.append(f"specs/ edited: {', '.join(spec_edits)}")
    return failures


def check_dependencies(card, files, new_deps_text, has_external_sources):
    """Validate dependencies in lockfiles according to card rules.

    For card A0.1 only, permit 'New dependencies: none' when Cargo.lock contains no
    registry/git source entries. Otherwise, require dependency justification.
    """
    failures = []
    lockfile_changed = "Cargo.lock" in files or any(f.endswith("uv.lock") for f in files)
    if lockfile_changed:
        deps = (new_deps_text or "").strip()
        is_none = deps.lower() in ("", "none", "- none")
        if is_none:
            a01_allowed = (
                card == "A0.1"
                and not has_external_sources
                and not any(f.endswith("uv.lock") for f in files)
            )
            if not a01_allowed:
                failures.append("lockfile changed but PR body's ## New dependencies is empty or 'none'")
    return failures


def check_line_patterns(file_path, added_lines_list):
    """Scan added lines of an implementation file for disallowed patterns (unsafe, asm, stubs, rng)."""
    failures = []
    f = file_path
    if not is_implementation_source(f):
        return failures

    for ln, text in added_lines_list:
        t = text.strip()
        if f.endswith(".rs") and re.search(r"\bunsafe\b", t) and not f.startswith(UNSAFE_OK) and not is_test_path(f):
            failures.append(f"{f}:{ln}: `unsafe` outside allowed crates")
        if re.search(r"\basm!\s*\(|__asm__|\basm\s+volatile", t) and not f.startswith(ASM_OK):
            failures.append(f"{f}:{ln}: inline asm outside r9v-kgen/src/leaf/")
        if not is_test_path(f):
            if re.search(r"\btodo!\(|\bunimplemented!\(", t) and not re.search(r"\b[A-D]\d+\.(?:S)?\d+[a-z]?\b", t):
                failures.append(f"{f}:{ln}: stub without a card id")
            if re.search(r"(//|#)\s*TODO\b", t) and not re.search(r"TODO\([A-D]\d+\.(?:S)?\d+[a-z]?\)", t):
                failures.append(f"{f}:{ln}: TODO without a card id")
        if not is_test_path(f) and re.search(r"thread_rng\(\)|\brandom\(\)|np\.random\.(?!default_rng|Generator)", t):
            failures.append(f"{f}:{ln}: unseeded randomness outside tests")
    return failures


def collect_decisions(files_with_added_lines):
    """Collect DECISION(<card>) comments only from implementation source files."""
    decisions = []
    for f, lines in files_with_added_lines:
        if not is_implementation_source(f):
            continue
        for ln, text in lines:
            m = re.search(r"DECISION\(([A-D]\d+\.(?:S)?\d+[a-z]?)\)", text)
            if m:
                decisions.append((f, ln, m.group(1)))
    return decisions


def added_lines(repo, base, path):
    out = git(repo, "diff", "-U0", f"{base}...HEAD", "--", path)
    lines = []
    lineno = 0
    for raw in out.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if m:
            lineno = int(m.group(1))
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            lines.append((lineno, raw[1:]))
            lineno += 1
        elif not raw.startswith("-"):
            lineno += 1
    return lines


def is_test_path(path):
    return "/tests/" in path or path.endswith("_test.rs") or "/benches/" in path or path.startswith("tests/")


def section(body, heading):
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", body, re.S | re.M)
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--base", default="main")
    ap.add_argument("--pr-body", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--api", action="store_true", help="retained for compatibility; untagged stubs are never allowed")
    args = ap.parse_args()

    if not CARD_RE.match(args.card):
        print(f"FAIL card id '{args.card}' does not look like a card id (e.g. A3.4, A0.S1)")
        return 1

    repo = str(Path(args.repo).resolve())
    body = Path(args.pr_body).read_text()
    files = changed_files(repo, args.base)
    failures = []

    # 1. specs are read-only (with narrow bootstrap exception for A0.1)
    spec_edits = [f for f in files if f.startswith("specs/")]
    if spec_edits:
        deliv = section(body, "Deliverables") or ""
        spec_statuses = git_spec_statuses(repo, args.base)
        failures.extend(check_spec_edits(args.card, spec_edits, spec_statuses, deliv))

    # 2. unsafe / asm placement, 7. stubs, 8. rng — scan added lines in implementation files
    for f in files:
        if not is_implementation_source(f):
            continue
        failures.extend(check_line_patterns(f, added_lines(repo, args.base, f)))

    # 3. generated artifacts
    gen_edits = [f for f in files if f.startswith(GENERATED)]
    if gen_edits:
        deliv = section(body, "Deliverables") or ""
        claimed = [f for f in gen_edits if any(p.rstrip("/") in deliv for p in GENERATED if f.startswith(p))]
        unclaimed = [f for f in gen_edits if f not in claimed]
        if unclaimed:
            failures.append("generated/measured artifacts edited without the card claiming them in Deliverables: "
                            + ", ".join(unclaimed))

    # 4. DECISION comments enumerated in implementation source files
    files_with_lines = [(f, added_lines(repo, args.base, f)) for f in files if is_implementation_source(f)]
    decisions = collect_decisions(files_with_lines)
    dec_section = section(body, "Decisions") or ""
    for f, ln, cid in decisions:
        if cid != args.card:
            failures.append(f"{f}:{ln}: DECISION tagged {cid} but this PR is card {args.card}")
        if f not in dec_section:
            failures.append(f"{f}:{ln}: DECISION not listed in the PR body's ## Decisions section")
    if not decisions and dec_section and dec_section.strip().lower() not in ("none", "- none"):
        failures.append("PR body lists decisions but the diff contains no DECISION comments")

    # 5. commits
    log = git(repo, "log", "--format=%H%x00%s%x00%b%x01", f"{args.base}..HEAD")
    for entry in filter(None, (e.strip() for e in log.split("\x01"))):
        sha, subject, msg_body = (entry.split("\x00") + ["", ""])[:3]
        if not subject.startswith(f"{args.card}:"):
            failures.append(f"commit {sha[:10]}: subject does not start with '{args.card}:' ({subject!r})")
        if "Signed-off-by:" not in msg_body:
            failures.append(f"commit {sha[:10]}: missing Signed-off-by")

    # 6. dependencies (with narrow bootstrap exception for A0.1)
    has_external = lockfile_has_external_sources(repo, "Cargo.lock")
    deps_text = section(body, "New dependencies") or ""
    failures.extend(check_dependencies(args.card, files, deps_text, has_external))

    # required sections present
    for h in ("Card", "Spec sections implemented", "Deliverables", "Done-when tests", "Decisions",
              "SPEC-ISSUES filed", "New dependencies", "Hardware", "Checklist", "Size check"):
        if section(body, h) is None:
            failures.append(f"PR body missing section '## {h}'")

    if failures:
        print(f"FAIL ({len(failures)})")
        for msg in failures:
            print("  -", msg)
        return 1
    print(f"PASS card {args.card}: {len(files)} files, {len(decisions)} decisions, specs untouched" if not spec_edits else f"PASS card {args.card}: {len(files)} files, {len(decisions)} decisions, specs seeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
