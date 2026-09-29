"""Device registry: where is each machine's public link right now?

A quick tunnel gets a new random `*.trycloudflare.com` hostname on every
restart, so "the URL" is not a stable identifier — the device is. This module
keeps one document per device in MongoDB and overwrites it whenever that device
comes back:

    {"link": "https://<random>.trycloudflare.com/v1/edit",
     "device": HOST_NAME,
     "updated_at": <BSON UTC datetime>}

Rules this file follows:

* **Never fatal.** A registry outage must not take down a working endpoint, so
  every failure becomes `RegistryError` and the caller decides what to do.
* **Never log the URI.** It carries a password; errors and log lines only ever
  name the host and database.
* **One row per device, enforced by the database**, not by hope: a unique index
  on `device` plus an idempotent upsert.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

try:  # pymongo lives in the API venv; importing qwen_api must not require it
    from pymongo import AsyncMongoClient
    from pymongo.errors import DuplicateKeyError, PyMongoError

    PYMONGO_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without api/requirements
    AsyncMongoClient = None  # type: ignore[assignment]
    PYMONGO_AVAILABLE = False

    class PyMongoError(Exception):  # type: ignore[no-redef]
        pass

    class DuplicateKeyError(PyMongoError):  # type: ignore[no-redef]
        pass


PUBLISH_TIMEOUT = 10.0
LINK_MAX_LENGTH = 512
CREDENTIALS = re.compile(r"(?<=://)[^/@]*@")


class RegistryError(RuntimeError):
    """The device could not be published. The endpoint is unaffected."""


def redact(uri: str) -> str:
    """`mongodb+srv://user:secret@host/db` -> `mongodb+srv://***@host/db`."""
    return CREDENTIALS.sub("***@", uri)


def validate_link(link: str) -> str:
    """Accept only an absolute https URL. Returns it trimmed."""
    link = (link or "").strip()
    if not link:
        raise ValueError("link is empty")
    if len(link) > LINK_MAX_LENGTH:
        raise ValueError(f"link is longer than {LINK_MAX_LENGTH} characters")
    parts = urlsplit(link)
    if parts.scheme != "https":
        raise ValueError("link must be https")
    if not parts.netloc:
        raise ValueError("link has no host")
    if any(character in link for character in "\r\n\t"):
        raise ValueError("link contains control characters")
    return link


class Registry:
    """Upserts this device's public link. Cheap to construct, lazy to connect."""

    def __init__(self, uri: str, collection: str, device: str) -> None:
        self.uri = uri
        self.collection_name = collection
        self.device = device
        self._client: Any = None
        self._collection: Any = None
        self._index_ready = False

    # ------------------------------------------------------------- plumbing

    @property
    def configured(self) -> bool:
        return bool(self.uri)

    @property
    def target(self) -> str:
        """A log-safe description: host and database, never credentials."""
        if not self.uri:
            return "not configured"
        return redact(self.uri).split("?", 1)[0] or "mongodb://<configured>"

    async def _collection_ref(self) -> Any:
        if not self.configured:
            raise RegistryError("MONGODB is not set; the device registry is disabled")
        if not PYMONGO_AVAILABLE:
            raise RegistryError(
                "pymongo is not installed. Run: qwen21 install "
                "(it installs api/requirements.txt into the API venv)"
            )
        if self._collection is None:
            self._client = AsyncMongoClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                appname="qwen21",
            )
            self._collection = self._client[self.device_database][self.collection_name]
        return self._collection

    @property
    def device_database(self) -> str:
        """The database is the URI path; an empty path means MongoDB's default."""
        path = urlsplit(self.uri).path.strip("/")
        return path or "test"

    async def _ensure_index(self, collection: Any) -> None:
        if self._index_ready:
            return
        try:
            await collection.create_index("device", unique=True, name="device_unique")
        except PyMongoError as exc:  # an existing index with other options is fine
            if "already exists" not in str(exc).lower():
                raise
        self._index_ready = True

    # ------------------------------------------------------------- writing

    async def publish(self, link: str) -> dict[str, Any]:
        """Point `device` at `link`, creating the document if it is new.

        A crashed or restarted device simply publishes again, which overwrites
        the previous row rather than adding a second one.
        """
        link = validate_link(link)
        try:
            collection = await self._collection_ref()
            await self._ensure_index(collection)
            document = {
                "link": link,
                "device": self.device,
                "updated_at": datetime.now(timezone.utc),
            }
            try:
                await collection.update_one(
                    {"device": self.device}, {"$set": document}, upsert=True
                )
            except DuplicateKeyError:
                # Lost a race against another process creating the same device
                # row between our index check and the upsert: set it, don't add.
                await collection.update_one({"device": self.device}, {"$set": document})
            return self._public(document)
        except RegistryError:
            raise
        except (PyMongoError, OSError, asyncio.TimeoutError) as exc:
            raise RegistryError(f"{self.target}: {type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------- reading

    async def current(self) -> dict[str, Any] | None:
        """What is stored for this device, or None. Never raises."""
        if not self.configured or not PYMONGO_AVAILABLE:
            return None
        try:
            collection = await self._collection_ref()
            document = await asyncio.wait_for(
                collection.find_one({"device": self.device}, {"_id": 0}), PUBLISH_TIMEOUT
            )
            return self._public(document) if document else None
        except Exception:  # a read-only convenience must never break a request
            return None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._collection = None

    @staticmethod
    def _public(document: dict[str, Any]) -> dict[str, Any]:
        updated = document.get("updated_at")
        if isinstance(updated, datetime):
            updated = updated.isoformat()
        return {"link": document.get("link"), "device": document.get("device"),
                "updated_at": updated}
