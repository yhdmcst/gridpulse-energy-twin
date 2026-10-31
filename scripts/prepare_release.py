"""Restore byte-identical research evidence from the bundled release archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_target(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        not relative or "\\" in relative or ":" in relative or path.is_absolute()
        or ".." in path.parts or relative != path.as_posix()
        or not (path.parts[0] == "outputs" or path.parts[:2] == ("data", "processed"))
    ):
        raise ValueError(f"Unsafe evidence path: {relative}")
    target = root.joinpath(*path.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Evidence path escapes the repository: {relative}")
    return target


def prepare(root: Path = ROOT, *, verify_only: bool = False, repair: bool = False) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "artifacts/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("bundles"):
        raise ValueError("Unsupported or empty evidence manifest.")
    seen = set()
    pending = []
    for bundle in manifest["bundles"]:
        relative = PurePosixPath(bundle["path"])
        if (
            relative.parts[:1] != ("artifacts",) or ".." in relative.parts
            or "\\" in bundle["path"] or ":" in bundle["path"]
            or relative.is_absolute() or relative.suffix != ".zip"
        ):
            raise ValueError("Unsafe archive path.")
        archive_path = (root / relative).resolve()
        if not archive_path.is_relative_to(root):
            raise ValueError("Archive path escapes the repository.")
        raw = archive_path.read_bytes()
        if len(raw) != bundle["bytes"] or digest(raw) != bundle["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {relative}")
        records = {record["path"]: record for record in bundle["files"]}
        if len(records) != len(bundle["files"]):
            raise ValueError("Duplicate manifest paths.")
        for name in records:
            safe_target(root, name)
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(records):
                raise ValueError(f"Archive inventory mismatch: {relative}")
            for info in archive.infolist():
                name = info.filename
                record = records[name]
                target = safe_target(root, name)
                if name in seen or info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Duplicate or non-regular evidence entry: {name}")
                seen.add(name)
                if info.file_size != record["bytes"]:
                    raise ValueError(f"Evidence size mismatch: {name}")
                data = archive.read(info)
                if digest(data) != record["sha256"]:
                    raise ValueError(f"Evidence checksum mismatch: {name}")
                if target.exists():
                    if not target.is_file():
                        raise ValueError(f"Evidence destination is not a file: {name}")
                    if digest(target.read_bytes()) == record["sha256"]:
                        continue
                    if not repair or verify_only:
                        raise ValueError(
                            f"Local evidence differs: {name}. "
                            "Preserve your changes before explicitly using --repair."
                        )
                if verify_only:
                    raise FileNotFoundError(f"Evidence has not been prepared: {name}")
                pending.append((archive_path, name, target))

    # Validate every bundle and destination before changing any local file.
    for archive_path, name, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive_path) as archive:
            data = archive.read(name)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(data)
            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    result = {
        "passed": True,
        "experiment": manifest["experiment"],
        "archives_verified": len(manifest["bundles"]),
        "files_verified": len(seen),
        "files_restored": len(pending),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify without restoring files.")
    parser.add_argument("--repair", action="store_true", help="Explicitly restore changed evidence files.")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(verify_only=args.verify_only, repair=args.repair), indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Evidence preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
