"""Validated atomic downloads: incomplete bytes never become a cache hit."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen


def atomic_download(url,destination,*,validator=None,expected_sha256=None,timeout=45,attempts=3,opener=urlopen):
    destination = Path(destination)
    if attempts < 1 or not 0 < timeout <= 60:
        raise ValueError("Positive attempt count and timeout <=60 seconds required.")
    destination.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(attempts):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent,prefix=destination.name+".",suffix=".part",delete=False) as stream:
                temp_path = Path(stream.name)
                digest = hashlib.sha256()
                with opener(url,timeout=timeout) as response:
                    expected_length = response.headers.get("Content-Length")
                    size = 0
                    while block := response.read(1024*1024):
                        stream.write(block)
                        digest.update(block)
                        size += len(block)
                if not size or (expected_length is not None and size != int(expected_length)):
                    raise ValueError("Empty or truncated download.")
                stream.flush()
                os.fsync(stream.fileno())
            if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                raise ValueError("Downloaded SHA-256 differs from expected snapshot.")
            if validator is not None:
                validator(temp_path)
            os.replace(temp_path,destination)
            return destination
        except (OSError,ValueError) as exc:
            if attempt == attempts-1:
                raise RuntimeError(f"Download failed after {attempts} attempts; previous cache preserved: {exc}") from exc
            time.sleep(min(2**attempt,4))
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
