import importlib.util
import json
from pathlib import Path

import pytest


def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_submission_package.py"
    spec = importlib.util.spec_from_file_location("submission_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_uses_current_pdf_and_includes_v8(tmp_path, monkeypatch):
    module = builder()
    main = tmp_path / "current"
    for name in module.DIRECTORIES:
        (main / name).mkdir(parents=True)
    source = main / "03_源代码"
    (source / "outputs/fusion_v8").mkdir(parents=True)
    (source / "outputs/fusion_v8/status.json").write_text('{"status":"complete"}')
    (main / "01_技术报告/GridPulse-技术报告.pdf").write_bytes(b"latest-v8-pdf")
    (main / "06_平台上传文件/PLACEHOLDER-技术报告.pdf").write_bytes(b"latest-v8-pdf")
    for name in ("README.md", "00_提交说明.md", "INTEGRATION_NOTES.md"):
        (main / name).write_text("current v8", encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", source)
    monkeypatch.setattr(module, "SUBMISSION_ROOT", source / "submission")
    package = module.build("TEST-TEAM")
    assert (package / "01_技术报告/GridPulse-技术报告.pdf").read_bytes() == b"latest-v8-pdf"
    assert (package / f"06_平台上传文件/{package.name}-技术报告.pdf").read_bytes() == b"latest-v8-pdf"
    assert (package / "03_源代码/outputs/fusion_v8/status.json").is_file()
    assert not (package / "03_源代码/submission").exists()
    manifest = json.loads((package / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "fusion_v8"
    with pytest.raises(FileExistsError):
        module.build("TEST-TEAM")


@pytest.mark.parametrize("label", ["../escape", "bad/name", "bad:name", ""])
def test_builder_rejects_unsafe_team_labels(label):
    with pytest.raises(ValueError):
        builder().build(label)


def test_experiment_reference_trajectories_are_packaged_but_external_repositories_are_excluded(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/validate_submission_package.py"
    spec = importlib.util.spec_from_file_location("submission_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.is_excluded(
        tmp_path / "03_源代码/outputs/fusion_v8/references/2022_fresh6_healthy.csv", tmp_path)
    assert not module.is_excluded(
        tmp_path / "03_源代码/outputs/fusion_v7/references/2021_fresh6_healthy.csv", tmp_path)
    assert module.is_excluded(tmp_path / "03_源代码/references/external_repo.py", tmp_path)
    assert module.is_excluded(tmp_path / "03_源代码/outputs/__pycache__/module.pyc", tmp_path)
