from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import os
from pathlib import Path
from zipfile import ZipFile

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "00_提交说明.md",
    "01_技术报告/GridPulse-技术报告.pdf",
    "03_源代码/README.md",
    "03_源代码/app.py",
    "03_源代码/requirements.txt",
    "03_源代码/data/processed/net_load_hourly.csv",
    "03_源代码/outputs/models/hist_gradient_boosting_test_predictions.csv",
    "03_源代码/outputs/models/hist_gradient_boosting_validation_permutation_importance.csv",
    "03_源代码/outputs/alerts/test_review_queue.csv",
    "03_源代码/outputs/models/day3_model_metrics.json",
    "03_源代码/outputs/models/multihorizon_metrics.json",
    "03_源代码/outputs/models/reference_baseline_metrics.json",
    "03_源代码/outputs/alerts/alert_evaluation.json",
    "03_源代码/outputs/scenarios/reserve_ablation_metrics.json",
    "03_源代码/outputs/scenarios/daily_energy_balance_metrics.json",
    "03_源代码/outputs/scenarios/daily_fixed_open_timeseries.csv",
    "03_源代码/outputs/scenarios/daily_dynamic_open_timeseries.csv",
    "03_源代码/outputs/scenarios/daily_fixed_balanced_timeseries.csv",
    "03_源代码/outputs/scenarios/daily_dynamic_balanced_timeseries.csv",
    "03_源代码/outputs/scenarios/peak_response_metrics.json",
    "03_源代码/outputs/scenarios/peak_test_dynamic_peak_response_timeseries.csv",
    "03_源代码/outputs/scenarios/peak_test_fixed_peak_response_timeseries.csv",
    "03_源代码/data/processed/independent_de_2021_hourly.csv",
    "03_源代码/outputs/validation/independent_2021_protocol.json",
    "03_源代码/outputs/validation/independent_2021_metrics.json",
    "03_源代码/outputs/validation/independent_2021_data_quality.json",
    "03_源代码/outputs/validation/independent_2021_robustness.json",
    "03_源代码/outputs/validation/independent_2021_predictions.csv",
    "03_源代码/outputs/validation/independent_2021_dynamic_peak_response_timeseries.csv",
    "05_复现与合规/independent-validation.md",
    "05_复现与合规/global-optimality.md",
    "05_复现与合规/controller-development.md",
    "05_复现与合规/later-year-validation.md",
    "05_复现与合规/release-review.md",
    "05_复现与合规/project-walkthrough-zh.md",
    "03_源代码/outputs/validation/independent_2022_protocol.json",
    "03_源代码/outputs/validation/independent_2022_metrics.json",
    "03_源代码/outputs/validation/independent_2022_data_quality.json",
    "03_源代码/outputs/validation/independent_2022_predictions.csv",
    "03_源代码/data/processed/independent_de_2022_hourly.csv",
    "03_源代码/outputs/development/controller_metrics.json",
    "03_源代码/outputs/development/operating_stress_metrics.json",
    "03_源代码/outputs/development/interval_audit_metrics.json",
    "03_源代码/outputs/development/renewable_alignment_metrics.json",
    "03_源代码/outputs/development/clean_environment_verification.json",
    "03_源代码/outputs/optimality/certificates.json",
    "03_源代码/outputs/optimality/independent_2021_oracle.csv",
    "03_源代码/outputs/optimality/exploratory_test_oracle.csv",
    "03_源代码/outputs/scenarios/storage_scenario_metrics.json",
    "03_源代码/outputs/metrics/data-quality-report.json",
    "05_复现与合规/reproducibility.md",
    "MANIFEST.json",
    "SHA256SUMS.txt",
    "03_源代码/outputs/fusion_v8/metrics.json",
    "03_源代码/outputs/fusion_v8/selection.json",
    "03_源代码/outputs/fusion_v8/ARTIFACT_SHA256.json",
    "03_源代码/outputs/fusion_v8/status.json",
    "03_源代码/src/fusion_evidence.py",
}
FORBIDDEN_PARTS = {".git", ".github", ".codex-finalizer", "tmp", "node_modules", "references", "__pycache__"}
FORBIDDEN_NAMES = {"CITATION.cff", "package-lock.json"}
SENSITIVE = re.compile(
    r"学校|学院|大学|指导教师|导师|队员|姓名|学号|手机号|微信|QQ|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+",
    re.IGNORECASE,
)
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/](?:Users|home)[\\/]|(?:OneDrive|Desktop)[\\/])", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_text(text: str) -> list[str]:
    return sorted({match.group(0) for match in SENSITIVE.finditer(text)} | {match.group(0) for match in LOCAL_PATH.finditer(text)})


def scan_pdf(path: Path) -> list[str]:
    reader = PdfReader(path)
    text = "\n".join(str(value) for value in (reader.metadata or {}).values())
    text += "\n" + "\n".join(page.extract_text() or "" for page in reader.pages)
    return scan_text(text)


def scan_pptx(path: Path) -> list[str]:
    text: list[str] = []
    with ZipFile(path) as archive:
        for item in archive.infolist():
            if item.filename.endswith((".xml", ".rels")):
                text.append(archive.read(item).decode("utf-8", errors="ignore"))
    return scan_text("\n".join(text))


def is_excluded(path: Path, package: Path) -> bool:
    parts = path.relative_to(package).parts
    forbidden = set(parts) & FORBIDDEN_PARTS
    if parts[:2] == ("03_源代码", "outputs"):
        forbidden.discard("references")
    return bool(forbidden or path.name in FORBIDDEN_NAMES or path.suffix.lower() == ".mp4")


def validate(package: Path) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    files = [path for path in package.rglob("*") if path.is_file()]
    rel_paths = {path.relative_to(package).as_posix() for path in files}
    checks.append({"name": "expected-files", "passed": EXPECTED <= rel_paths, "missing": sorted(EXPECTED - rel_paths)})

    forbidden = [
        path.relative_to(package).as_posix()
        for path in files
        if is_excluded(path, package)
    ]
    checks.append({"name": "excluded-files", "passed": not forbidden, "findings": forbidden})

    report = package / "01_技术报告/GridPulse-技术报告.pdf"
    report_reader = PdfReader(report) if report.is_file() else None
    checks.append(
        {
            "name": "technical-report-pdf-spec",
            "passed": bool(report_reader) and report.stat().st_size <= 10 * 1024 * 1024,
            "bytes": report.stat().st_size if report.is_file() else None,
            "pages": len(report_reader.pages) if report_reader else None,
        }
    )
    platform_pairs = (
        (report, next(iter(sorted((package / "06_平台上传文件").glob("*-技术报告.pdf"))),
                      package / "06_平台上传文件" / f"{package.name}-技术报告.pdf")),
        (package / "02_答辩材料/GridPulse-答辩材料.pdf",
         next(iter(sorted((package / "06_平台上传文件").glob("*-答辩材料.pdf"))),
              package / "06_平台上传文件" / f"{package.name}-答辩材料.pdf")),
    )
    active_pairs = [pair for pair in platform_pairs if pair[0].is_file() or pair[0] == report]
    checks.append({"name": "platform-upload-pdf-copies",
                   "passed": all(original.is_file() and named.is_file()
                                 and sha256(original) == sha256(named)
                                 for original, named in active_pairs)})

    identity_findings: dict[str, list[str]] = {}
    for artifact in (
        report,
        package / "02_答辩材料/GridPulse-答辩材料.pdf",
        package / "02_答辩材料/GridPulse-答辩材料.pptx",
    ):
        if artifact.is_file():
            hits = scan_pptx(artifact) if artifact.suffix == ".pptx" else scan_pdf(artifact)
            if hits:
                identity_findings[artifact.relative_to(package).as_posix()] = hits
    for artifact in files:
        if artifact.suffix in {".py", ".mjs", ".js", ".json", ".md", ".txt"}:
            text = artifact.read_text(encoding="utf-8", errors="ignore")
            hits = sorted({match.group(0) for match in LOCAL_PATH.finditer(text)})
            # Policy documents legitimately discuss anonymity; scan email identities separately.
            hits += re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text) if artifact.suffix not in {".py", ".mjs", ".js"} else []
            if hits:
                identity_findings[artifact.relative_to(package).as_posix()] = hits
    checks.append({"name": "automated-anonymous-scan", "passed": not identity_findings, "findings": identity_findings})

    manifest_path = package / "MANIFEST.json"
    manifest_ok = False
    manifest_error = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = {entry["path"]: entry for entry in manifest.get("files", [])}
            actual_paths = rel_paths - {"MANIFEST.json", "SHA256SUMS.txt"}
            manifest_ok = set(entries) == actual_paths and all(
                entries[relative]["sha256"] == sha256(package / relative)
                and entries[relative]["bytes"] == (package / relative).stat().st_size
                for relative in actual_paths
            )
            sums = (package / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
            manifest_ok = manifest_ok and set(sums) == {f"{entry['sha256']}  {entry['path']}" for entry in entries.values()}
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            manifest_error = str(exc)
    checks.append({"name": "manifest-checksums", "passed": manifest_ok, "detail": manifest_error})
    source = package / "03_源代码"
    protocol_file = source / "outputs/validation/independent_2021_protocol.json"
    validation_file = source / "outputs/validation/independent_2021_metrics.json"
    lineage_ok = False
    if protocol_file.is_file() and validation_file.is_file():
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        evidence = json.loads(validation_file.read_text(encoding="utf-8"))
        lineage_ok = (all((source/name).is_file() and sha256(source/name) == expected
                          for name,expected in protocol["frozen_sha256"].items())
                      and sha256(protocol_file) == evidence["protocol_sha256"]
                      and sha256(source / "data/processed/independent_de_2021_hourly.csv") == evidence["dataset_sha256"])
    checks.append({"name": "packaged-independent-frozen-hashes", "passed": lineage_ok})
    cert_check = subprocess.run(
        [sys.executable,"-B",str(source / "scripts/certify_optimality.py"),"--verify-only"],
        cwd=source,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=90,
        env={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONDONTWRITEBYTECODE":"1"})
    checks.append({"name":"packaged-exact-and-primal-dual-optimality-certificates",
                   "passed":cert_check.returncode == 0,
                   "detail":cert_check.stdout[-1500:] if cert_check.returncode == 0 else cert_check.stderr[-1500:]})
    development_check = subprocess.run([sys.executable,"-B",str(source/"scripts/audit_development.py")],
        cwd=source,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120,
        env={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONDONTWRITEBYTECODE":"1"})
    checks.append({"name":"packaged-controller-operating-and-energy-audit","passed":development_check.returncode == 0,
        "detail":development_check.stdout[-1000:] if development_check.returncode == 0 else development_check.stderr[-1500:]})
    later_check = subprocess.run([sys.executable,"-B",str(source/"scripts/audit_later_year.py")],
        cwd=source,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120,
        env={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONDONTWRITEBYTECODE":"1"})
    checks.append({"name":"packaged-locked-2022-replication-audit","passed":later_check.returncode == 0,
        "detail":later_check.stdout[-1000:] if later_check.returncode == 0 else later_check.stderr[-1500:]})
    app_check = subprocess.run(
        [sys.executable, "-B", str(source / "scripts/check_app.py"), "--root", str(source)],
        cwd=source if source.is_dir() else package,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=150,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    checks.append({"name": "packaged-app-startup-and-date-interaction", "passed": app_check.returncode == 0,
                   "detail": "startup, full period and last day passed" if app_check.returncode == 0 else app_check.stdout[-2000:]})

    return {
        "schemaVersion": "gridpulse-submission-package-validation.v2",
        "version": "fusion_v8",
        "package": str(package),
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "submission_items": "Team identity, demonstration video, defense update and organizer upload.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a generated GridPulse submission package.")
    parser.add_argument("package", type=Path, help="Path to the generated submission folder.")
    args = parser.parse_args()
    package = args.package.resolve()
    report = validate(package)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
