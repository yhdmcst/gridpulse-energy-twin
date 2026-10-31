import io

import pytest

from src.downloads import atomic_download


class Response(io.BytesIO):
    def __init__(self,data,length=None):
        super().__init__(data)
        self.headers = {} if length is None else {"Content-Length":str(length)}


def test_partial_download_preserves_existing_cache_and_removes_temp(tmp_path):
    path = tmp_path/"data.csv"
    path.write_bytes(b"original")
    with pytest.raises(RuntimeError,match="preserved"):
        atomic_download("test",path,attempts=1,opener=lambda *a,**k:Response(b"partial",100))
    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.part"))


def test_rejects_wrong_digest_and_validation_before_replacing(tmp_path):
    path = tmp_path/"data.csv"
    with pytest.raises(RuntimeError,match="SHA-256"):
        atomic_download("test",path,attempts=1,expected_sha256="wrong",opener=lambda *a,**k:Response(b"ok"))
    assert not path.exists()
    def reject(_):
        raise ValueError("bad format")
    with pytest.raises(RuntimeError,match="bad format"):
        atomic_download("test",path,attempts=1,validator=reject,opener=lambda *a,**k:Response(b"ok"))
    assert not path.exists()


def test_successful_validated_download_replaces_atomically(tmp_path):
    path = tmp_path/"data.csv"
    def validate(p):
        assert p.read_bytes() == b"complete"
    atomic_download("test",path,attempts=1,validator=validate,opener=lambda *a,**k:Response(b"complete",8))
    assert path.read_bytes() == b"complete"
