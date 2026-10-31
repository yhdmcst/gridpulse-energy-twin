from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import subprocess
import sys
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from validate_submission_package import validate


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ROOT = ROOT / "submission"
DEFAULT_PACKAGE = SUBMISSION_ROOT / "TEAM-ID-TO-RENAME-AIC-AI能源-GridPulse"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive(package: Path, output: Path, replace: bool) -> Path:
    package = package.resolve()
    output = output.resolve()
    if package.parent != SUBMISSION_ROOT.resolve() or not package.is_dir():
        raise ValueError("Package must be an existing direct child of submission/.")
    if output.parent != SUBMISSION_ROOT.resolve() or output.suffix.lower() != ".zip":
        raise ValueError("ZIP output must be a .zip file directly under submission/.")
    report = validate(package)
    if not report["passed"]:
        raise RuntimeError("Package validation failed; refusing to archive incomplete material.")
    if output.exists() and not replace:
        raise FileExistsError(f"ZIP already exists: {output}. Use --replace to overwrite it.")

    root_name = package.name
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive_file:
        for path in sorted(item for item in package.rglob("*") if item.is_file()):
            archive_file.write(path, arcname=(Path(root_name) / path.relative_to(package)).as_posix())
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    return output


def verify_archive(archive_path: Path, package: Path) -> None:
    expected_prefix = package.name + "/"
    with ZipFile(archive_path) as archive_file:
        names = archive_file.namelist()
        if not names or any(not name.startswith(expected_prefix) for name in names):
            raise RuntimeError("ZIP does not contain exactly one submission-folder root.")
        if f"{expected_prefix}MANIFEST.json" not in names or f"{expected_prefix}SHA256SUMS.txt" not in names:
            raise RuntimeError("ZIP is missing package integrity files.")
        bad = [name for name in names if any(part in {".git", ".github", "tmp", "node_modules"} for part in Path(name).parts)]
        if bad:
            raise RuntimeError(f"ZIP contains excluded paths: {bad}")
        if archive_file.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed.")
        manifest = json.loads(archive_file.read(expected_prefix + "MANIFEST.json"))
        for entry in manifest["files"]:
            if hashlib.sha256(archive_file.read(expected_prefix + entry["path"])).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Archived content checksum mismatch: {entry['path']}")
        with tempfile.TemporaryDirectory(prefix="gridpulse-zip-check-") as temporary:
            archive_file.extractall(temporary)
            source = Path(temporary) / package.name / "03_源代码"
            result = subprocess.run(
                [sys.executable, "-B", "scripts/check_app.py"], cwd=source,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=150,
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if result.returncode:
                raise RuntimeError("Extracted application check failed: " + result.stdout[-2000:])


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a validated ZIP archive of a GridPulse submission package.")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE, help="Submission folder to archive.")
    parser.add_argument("--output", type=Path, help="ZIP output path. Defaults to submission/<package-name>.zip.")
    parser.add_argument("--replace", action="store_true", help="Replace an existing ZIP under submission/.")
    args = parser.parse_args()
    package = args.package if args.package.is_absolute() else ROOT / args.package
    output = args.output if args.output else SUBMISSION_ROOT / f"{package.name}.zip"
    output = output if output.is_absolute() else ROOT / output
    archive_path = archive(package, output, args.replace)
    verify_archive(archive_path, package.resolve())
    print(f"ZIP archive: {archive_path}")
    print(f"SHA-256: {sha256(archive_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
