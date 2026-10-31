"""Publish the checked current report and synchronize competition-facing copies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

from competition_common import DEST, ROOT, WORKSPACE
from competition_report import PDF

sys.path.insert(0, str(ROOT / "scripts"))
from build_submission_package import write_manifest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    check = json.loads((DEST / "QA/submission_audit.json").read_text(encoding="utf-8"))
    build = json.loads((DEST / "QA/report_build.json").read_text(encoding="utf-8"))
    manifest = json.loads((DEST / "delivery_manifest.json").read_text(encoding="utf-8"))
    if not check["automated_passed"] or sha(PDF) != build["pdf_sha256"]:
        raise ValueError("Current report has not passed its build and audit checks.")
    for relative, expected in manifest["files_sha256"].items():
        path = (DEST / relative).resolve()
        if not path.is_relative_to(DEST.resolve()) or sha(path) != expected:
            raise ValueError(f"Report evidence changed after audit: {relative}")
    target = ROOT.parent / "01_技术报告/GridPulse-技术报告.pdf"
    upload = ROOT.parent / "06_平台上传文件"
    candidates = sorted(upload.glob("*-技术报告.pdf"))
    if len(candidates) != 1:
        raise ValueError("Exactly one platform report copy is required.")
    for destination in (target, candidates[0]):
        shutil.copy2(PDF, destination)
        if sha(destination) != sha(PDF):
            raise RuntimeError("Published PDF differs from checked report.")
    registry = WORKSPACE / "MAIN_VERSION.json"
    if registry.is_file():
        value = json.loads(registry.read_text(encoding="utf-8"))
        value["current_technical_report"]["sha256"] = sha(PDF)
        value["application_display_version"] = "fusion_v8"
        value["application_default_replay_policy"] = "envelope_tail"
        registry.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = ROOT / "outputs/review/report_publication.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "passed": True, "version": "fusion_v8", "sha256": sha(PDF),
        "pages": check["pages"], "bytes": PDF.stat().st_size,
        "published_copies": 2,
    }, indent=2), encoding="utf-8")
    write_manifest(ROOT.parent)
    print(receipt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
