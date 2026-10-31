import asyncio
import gzip

import pytest

from streamlit.web.server.starlette.starlette_gzip_middleware import SelectiveGZipMiddleware


@pytest.mark.parametrize("accept_encoding", [b"gzip", b"identity"])
def test_dashboard_media_sends_response_headers_before_image_bytes(accept_encoding):
    messages = []
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"gridpulse-image" * 100

    async def application(scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"image/png"),
                (b"content-length", str(len(image_bytes)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": image_bytes})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    middleware = SelectiveGZipMiddleware(application)
    asyncio.run(middleware({
        "type": "http", "path": "/media/example.png",
        "headers": [(b"accept-encoding", accept_encoding)],
    }, receive, send))
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 200
    headers = dict(messages[0]["headers"])
    assert headers[b"content-type"] == b"image/png"
    body = b"".join(message.get("body", b"") for message in messages[1:])
    if headers.get(b"content-encoding") == b"gzip":
        body = gzip.decompress(body)
    assert body == image_bytes
