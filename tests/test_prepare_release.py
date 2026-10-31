import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.prepare_release import prepare


def fixture_release(root: Path, name: str = "outputs/fusion_v8/example.csv") -> Path:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    data = b"timestamp,value\n2022-01-01,42\n"
    archive_path = artifacts / "evidence-01.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(name, data)
    manifest = {
        "schema_version": 1,
        "experiment": "fusion_v8",
        "bundles": [{
            "path": "artifacts/evidence-01.zip",
            "bytes": archive_path.stat().st_size,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "files": [{
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }],
        }],
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root / name


def test_prepare_is_idempotent_and_verifiable(tmp_path):
    target = fixture_release(tmp_path)
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path, verify_only=True)
    assert prepare(tmp_path)["files_restored"] == 1
    assert target.read_bytes().endswith(b",42\n")
    assert prepare(tmp_path)["files_restored"] == 0
    assert prepare(tmp_path, verify_only=True)["files_verified"] == 1


def test_changed_local_evidence_is_preserved_unless_repair_is_explicit(tmp_path):
    target = fixture_release(tmp_path)
    prepare(tmp_path)
    target.write_bytes(b"local research changes")
    with pytest.raises(ValueError, match="Local evidence differs"):
        prepare(tmp_path)
    assert target.read_bytes() == b"local research changes"
    assert prepare(tmp_path, repair=True)["files_restored"] == 1


def test_corrupted_archive_is_rejected_before_extraction(tmp_path):
    target = fixture_release(tmp_path)
    with (tmp_path / "artifacts/evidence-01.zip").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Archive checksum"):
        prepare(tmp_path)
    assert not target.exists()


@pytest.mark.parametrize("name", [
    "../outside.csv", "outputs/../../outside.csv", "/outputs/absolute.csv",
    "src/replaced.py", "outputs\\windows.csv", "outputs/C:/drive.csv",
])
def test_unsafe_archive_entries_are_rejected(tmp_path, name):
    fixture_release(tmp_path, name)
    with pytest.raises(ValueError, match="Unsafe evidence path"):
        prepare(tmp_path)


def test_manifest_hash_mismatch_is_rejected(tmp_path):
    fixture_release(tmp_path)
    path = tmp_path / "artifacts/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["bundles"][0]["files"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Evidence checksum"):
        prepare(tmp_path)
