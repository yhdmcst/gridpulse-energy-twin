import csv
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def test_showcase_snapshot_preserves_frozen_values_and_source_hashes():
    text = (ROOT / "docs/showcase/trace-data.js").read_text(encoding="utf-8")
    snapshot = json.loads(text.removeprefix("window.GRIDPULSE_TRACE = ").rstrip(";\n"))
    manifest = json.loads((ROOT / "artifacts/manifest.json").read_text(encoding="utf-8"))
    originals = {}
    for source, expected in snapshot["source_sha256"].items():
        for bundle in manifest["bundles"]:
            if any(item["path"] == source for item in bundle["files"]):
                with ZipFile(ROOT / bundle["path"]) as archive:
                    raw = archive.read(source)
                assert hashlib.sha256(raw).hexdigest() == expected
                originals[source] = {
                    row["timestamp"]: row
                    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
                }
                break
    assert len(originals) == 3
    trace = next(rows for name, rows in originals.items() if "fusion_v8/evaluation" in name)
    fresh = next(rows for name, rows in originals.items() if name.endswith("fresh6.csv"))
    stale = next(rows for name, rows in originals.items() if name.endswith("delayed3.csv"))
    assert len(snapshot["points"]) == 24
    for point in snapshot["points"]:
        original = trace[point["timestamp"]]
        assert point["timestamp"].startswith(snapshot["date"])
        for key, value in point.items():
            if isinstance(value, float):
                assert value == float(original[key])
        assert point["gate_reason"] == original["gate_reason"]
        assert point["optimizer_called"] == (original["primary_solver_attempted"].lower() == "true")
        panel = (stale if point["gate_reason"] == "forecast_expired" else fresh)[point["timestamp"]]
        for horizon, values in enumerate(point["forecast"], start=1):
            for field, value in values.items():
                assert value == float(panel[f"{field}_h{horizon}_mw"])


def test_showcase_runtime_resources_are_local_and_present():
    class Resources(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "script" and "src" in attrs:
                self.paths.append(attrs["src"])
            if tag == "link" and attrs.get("rel") == "stylesheet":
                self.paths.append(attrs["href"])

    directory = ROOT / "docs/showcase"
    parser = Resources()
    parser.feed((directory / "index.html").read_text(encoding="utf-8"))
    assert len(parser.paths) == 4
    for relative in parser.paths:
        assert not urlparse(relative).scheme
        assert (directory / relative).is_file()
    assert (directory / "LUCIDE-LICENSE.txt").is_file()


def test_readme_language_editions_have_matching_structure_and_theme_fallbacks():
    chinese = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README_EN.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^## ", chinese, re.MULTILINE)) == 8
    assert len(re.findall(r"^## ", english, re.MULTILINE)) == 8
    assert chinese.count("<details>") == english.count("<details>") == 6
    for text in (chinese, english):
        assert "prefers-color-scheme: dark" in text
        assert "prefers-reduced-motion: reduce" in text
        assert "demo-preview.gif" in text
        assert "demo-poster.png" in text
        assert "30 / 30" in text and "11 / 30" in text


def test_preview_deliverables_are_inside_the_upload_folder_and_self_contained():
    directory = ROOT / "preview"
    expected = [
        "index.html", "GridPulse-SVG演示.html", "README预览.html", "README_EN预览.html",
        "README预览.png", "README预览-dark.png", "README_EN预览.png",
        "README_EN预览-dark.png", "README体验升级说明.md",
    ]
    assert all((directory / name).is_file() for name in expected)

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.base = ""
            self.targets = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "base":
                self.base = attrs.get("href", "")
                return
            for name in ("src", "srcset", "href"):
                if name in attrs:
                    self.targets.append(attrs[name])

    for path in directory.glob("*.html"):
        parser = Links()
        parser.feed(path.read_text(encoding="utf-8"))
        base = (path.parent / parser.base).resolve()
        assert base.is_relative_to(ROOT.resolve())
        for relative in parser.targets:
            parts = urlparse(relative)
            if parts.scheme or not parts.path:
                continue
            target = (base / unquote(parts.path)).resolve()
            assert target.is_relative_to(ROOT.resolve()), (path.name, relative)
            assert target.exists(), (path.name, relative)

    standalone = (directory / "GridPulse-SVG演示.html").read_text(encoding="utf-8")
    assert "window.GRIDPULSE_TRACE =" in standalone
    assert "window.GridPulsePlayer =" in standalone
    assert not re.search(r'<script[^>]+src=|<link[^>]+rel="stylesheet"', standalone)
