"""`python -m qwen_api` - run the service with uvicorn."""
from __future__ import annotations

import os

import uvicorn

from .config import settings


def main() -> int:
    uvicorn.run(
        "qwen_api.app:app",
        host=settings.host,
        port=settings.port,
        log_level=os.environ.get("API_LOG_LEVEL", "info"),
        access_log=os.environ.get("API_ACCESS_LOG", "0") not in {"0", "false", ""},
        # Under ~100s so uvicorn's own keep-alive never collides with the
        # Cloudflare proxy read timeout on long-running requests.
        timeout_keep_alive=75,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
