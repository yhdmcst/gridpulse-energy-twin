"""Package the current V8 working tree and generate its content manifest."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ROOT = ROOT / "submission"
DEFAULT_TEAM_ID = "TEAM-ID-TO-RENAME"
DIRECTORIES = ("01_技术报告", "02_答辩材料", "03_源代码",
               "04_实验结果", "05_复现与合规", "06_平台上传文件")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(package: Path) -> None:
    entries = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package).as_posix()
        if path.is_file() and relative not in {"MANIFEST.json", "SHA256SUMS.txt"}:
            entries.append({"path": relative,
                            "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schemaVersion": "gridpulse-submission-package.v2",
        "generatedOn": date.today().isoformat(),
        "version": "fusion_v8",
        "contents": "Current V8 source, report, standardized data and frozen experiment evidence.",
        "files": entries,
    }
    (package / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (package / "SHA256SUMS.txt").write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries), encoding="utf-8"
    )


def build(team_id: str, replace: bool = False) -> Path:
    safe_team_id = team_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", safe_team_id):
        raise ValueError("Team ID must contain ASCII letters, numbers, hyphens or underscores.")
    SUBMISSION_ROOT.mkdir(exist_ok=True)
    package = (SUBMISSION_ROOT / f"{safe_team_id}-AIC-AI能源-GridPulse").resolve()
    if package.parent != SUBMISSION_ROOT.resolve():
        raise ValueError("Package target escaped submission/.")
    if package.exists():
        raise FileExistsError(f"Package exists: {package}. Choose a new output team label.")
    current = ROOT.parent
    report = current / "01_技术报告/GridPulse-技术报告.pdf"
    if not report.is_file() or not (ROOT / "outputs/fusion_v8/status.json").is_file():
        raise FileNotFoundError("The current report and completed V8 results are required.")
    if replace:
        raise ValueError("Use a fresh output label; existing packages are preserved.")
    package.mkdir()
    for name in DIRECTORIES:
        shutil.copytree(current / name, package / name, ignore=shutil.ignore_patterns(
            "submission", "__pycache__", ".pytest_cache", "*.pyc", ".venv"
        ))
    for name in ("README.md", "00_提交说明.md", "INTEGRATION_NOTES.md"):
        shutil.copy2(current / name, package / name)
    upload = package / "06_平台上传文件"
    for label, directory in (("技术报告", "01_技术报告"), ("答辩材料", "02_答辩材料")):
        original = package / directory / f"GridPulse-{label}.pdf"
        if not original.is_file():
            continue
        target = upload / f"{package.name}-{label}.pdf"
        for previous in sorted(upload.glob(f"*-{label}.pdf")):
            if previous != target:
                if target.exists():
                    raise FileExistsError(f"Duplicate upload copy: {target}")
                previous.rename(target)
        shutil.copy2(original, target)
    write_manifest(package)
    return package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the current V8 competition package.")
    parser.add_argument("--team-id", default=DEFAULT_TEAM_ID)
    args = parser.parse_args()
    print(build(args.team_id))
