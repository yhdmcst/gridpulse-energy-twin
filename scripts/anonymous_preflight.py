from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    "outputs/report/gridpulse-technical-report-v0.1.pdf",
    "outputs/presentation/gridpulse-defense-v9.pptx",
    "outputs/presentation/gridpulse-defense-v9.pdf",
)
SENSITIVE_TEXT = re.compile(
    r"学校|学院|大学|指导教师|导师|队员|姓名|学号|手机号|微信|QQ|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+",
    re.IGNORECASE,
)
LOCAL_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/](?:Users|home)[\\/]|(?:OneDrive|Desktop)[\\/])",
    re.IGNORECASE,
)


def scan_text(text: str) -> list[str]:
    return sorted({match.group(0) for match in SENSITIVE_TEXT.finditer(text)} | {match.group(0) for match in LOCAL_PATH.finditer(text)})


def scan_pdf(path: Path) -> list[dict[str, object]]:
    reader = PdfReader(path)
    findings: list[dict[str, object]] = []
    metadata = "\n".join(str(value) for value in (reader.metadata or {}).values())
    if hits := scan_text(metadata):
        findings.append({"location": "metadata", "hits": hits})
    for page_number, page in enumerate(reader.pages, start=1):
        if hits := scan_text(page.extract_text() or ""):
            findings.append({"location": f"page:{page_number}", "hits": hits})
    return findings


def scan_pptx(path: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    with ZipFile(path) as archive:
        for entry in archive.infolist():
            if not entry.filename.endswith((".xml", ".rels")):
                continue
            text = archive.read(entry).decode("utf-8", errors="ignore")
            if hits := scan_text(text):
                findings.append({"location": entry.filename, "hits": hits})
    return findings


def scan_source_paths() -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for directory in (ROOT / "scripts", ROOT / "src"):
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".mjs", ".js"}:
                continue
            if hits := sorted({match.group(0) for match in LOCAL_PATH.finditer(path.read_text(encoding="utf-8"))}):
                findings.append({"location": str(path.relative_to(ROOT)), "hits": hits})
    return findings


def main() -> int:
    checks: list[dict[str, object]] = []
    for relative in ARTIFACTS:
        path = ROOT / relative
        if not path.is_file():
            checks.append({"artifact": relative, "passed": False, "findings": ["missing"]})
            continue
        findings = scan_pptx(path) if path.suffix == ".pptx" else scan_pdf(path)
        checks.append({"artifact": relative, "passed": not findings, "findings": findings})

    source_findings = scan_source_paths()
    checks.append({"artifact": "source-local-paths", "passed": not source_findings, "findings": source_findings})
    report = {
        "schemaVersion": "gridpulse-anonymous-preflight.v1",
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "manual_boundary": (
            "This scan cannot identify a person from visual-only content, Git history, "
            "browser tabs, or a shared-drive URL. Exclude .git and identity-bearing "
            "repository metadata from an anonymous submission package."
        ),
    }
    output = ROOT / ".codex-finalizer" / "gridpulse-anonymous-preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
