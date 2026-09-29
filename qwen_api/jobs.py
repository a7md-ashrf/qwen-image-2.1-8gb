"""Job bookkeeping: in-memory state plus a small SQLite mirror.

ComfyUI serialises execution on its own queue, so this module does not run a
worker pool - it hands a prompt to ComfyUI and watches the history. Its jobs are
what turns a long generation into a pollable, restart-survivable id.

**Rendered images live in RAM, never on disk.** Each job keeps its image bytes in
a bounded LRU cache (`IMAGE_CACHE_MB`, 64 MB by default) so that the async 202
path can still hand them over when the client polls. The SQLite mirror keeps
metadata only - after a restart the bytes are gone by design and the job reports
`retained: false`.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL = {SUCCEEDED, FAILED, CANCELLED}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    comfy_prompt_id TEXT,
    seed INTEGER,
    error TEXT,
    duration REAL,
    images TEXT NOT NULL DEFAULT '[]',
    request TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs (created_at);
"""


@dataclass
class Job:
    id: str
    kind: str
    status: str = QUEUED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    comfy_prompt_id: str | None = None
    seed: int | None = None
    error: str | None = None
    duration: float | None = None
    images: list[dict[str, Any]] = field(default_factory=list)
    request: dict[str, Any] = field(default_factory=dict)
    task: asyncio.Task | None = field(default=None, repr=False, compare=False)
    done: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)
    # {index: bytes} for this job only. Never persisted, never serialised.
    blobs: dict[int, bytes] = field(default_factory=dict, repr=False, compare=False)

    @property
    def elapsed(self) -> float:
        return (self.updated_at or time.time()) - self.created_at

    def image(self, index: int) -> dict[str, Any] | None:
        for position, entry in enumerate(self.images):
            if entry.get("index", position) == index:
                return entry
        return None

    def blob(self, index: int) -> bytes | None:
        """The image bytes, or None once evicted or after a restart."""
        if not self.blobs:
            return None
        return self.blobs.get(index)


class JobStore:
    """Jobs plus a bounded in-RAM image cache. Writes no image files.

    `db_path` is the only thing that ever touches the filesystem.
    """

    def __init__(
        self,
        db_path: Path,
        ttl_hours: float = 24.0,
        image_cache_mb: float = 64.0,
    ) -> None:
        self.db_path = db_path
        self.ttl = ttl_hours * 3600
        self.cache_budget = max(0, int(image_cache_mb * 1024 * 1024))
        self._cached_bytes = 0
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._restore()

    # ------------------------------------------------------------- storage

    def _restore(self) -> None:
        rows = self._db.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        for row in rows:
            (jid, kind, status, created, updated, prompt_id, seed, error, duration, images,
             request) = row
            if time.time() - created > self.ttl:
                continue
            job = Job(
                id=jid,
                kind=kind,
                status=status,
                created_at=created,
                updated_at=updated,
                comfy_prompt_id=prompt_id,
                seed=seed,
                error=error,
                duration=duration,
                images=json.loads(images),
                request=json.loads(request),
            )
            # Nothing from the old process can be retained: the bytes died with
            # it, so say so up front rather than 404-ing at poll time.
            for entry in job.images:
                entry.pop("path", None)
                entry["retained"] = False
            if status in TERMINAL:
                job.done.set()
            else:
                # Anything still "running" died with the previous process.
                job.status = FAILED
                job.error = "service restarted while this job was in flight"
                job.updated_at = time.time()
                job.done.set()
            self._jobs[jid] = job

    def _persist(self, job: Job) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO jobs "
                "(id, kind, status, created_at, updated_at, comfy_prompt_id, seed, error, "
                " duration, images, request) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job.id,
                    job.kind,
                    job.status,
                    job.created_at,
                    job.updated_at,
                    job.comfy_prompt_id,
                    job.seed,
                    job.error,
                    job.duration,
                    json.dumps(job.images),
                    json.dumps(job.request),
                ),
            )
            self._db.commit()

    # ---------------------------------------------------------------- api

    def create(self, kind: str, request: dict[str, Any] | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:24], kind=kind, request=request or {})
        self._jobs[job.id] = job
        self._persist(job)
        return job

    # ------------------------------------------------------------- RAM cache

    @property
    def cached_bytes(self) -> int:
        return self._cached_bytes

    def cache_images(self, job: Job, blobs: dict[int, bytes]) -> list[int]:
        """Try to retain `blobs` for later polling.

        Never raises and never fails the request: an image that does not fit is
        simply not retained, and the response that is about to be built still
        has the bytes. Returns the indices that were kept.
        """
        kept: list[int] = []
        for index, data in blobs.items():
            if len(data) > self.cache_budget or self.cache_budget == 0:
                self._mark_unretained(job, index)
                continue
            while self._cached_bytes + len(data) > self.cache_budget:
                if not self._evict_oldest(protect=job.id):
                    break
            if self._cached_bytes + len(data) > self.cache_budget:
                self._mark_unretained(job, index)
                continue
            job.blobs[index] = data
            self._cached_bytes += len(data)
            kept.append(index)
        self._mark_retained(job, kept)
        return kept

    def _evict_oldest(self, protect: str | None = None) -> bool:
        """Drop the least recently updated job's images. True if something went."""
        candidates = [
            job for job in self._jobs.values()
            if job.blobs and job.id != protect and job.status in TERMINAL
        ]
        if not candidates:
            candidates = [job for job in self._jobs.values()
                          if job.blobs and job.id != protect]
        if not candidates:
            return False
        victim = min(candidates, key=lambda j: j.updated_at)
        self._forget_blobs(victim)
        return True

    def _forget_blobs(self, job: Job) -> None:
        for index in list(job.blobs):
            self._cached_bytes -= len(job.blobs.pop(index))
        self._mark_unretained(job, [i.get("index", n) for n, i in enumerate(job.images)])

    @staticmethod
    def _mark_retained(job: Job, indices: list[int]) -> None:
        for entry in job.images:
            entry["retained"] = entry.get("index", 0) in indices

    @staticmethod
    def _mark_unretained(job: Job, index: int) -> None:
        entry = job.image(index)
        if entry is not None:
            entry["retained"] = False

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def update(self, job: Job, **fields: Any) -> Job:
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = time.time()
        if job.status in TERMINAL:
            job.done.set()
        self._persist(job)
        return job

    def counts(self) -> tuple[int, int]:
        running = sum(1 for j in self._jobs.values() if j.status == RUNNING)
        queued = sum(1 for j in self._jobs.values() if j.status == QUEUED)
        return running, queued

    def queue_length(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in {QUEUED, RUNNING})

    # ------------------------------------------------------------- control

    async def run(self, job: Job, work: Callable[[Job], Awaitable[None]]) -> Job:
        async def wrapper() -> None:
            self.update(job, status=RUNNING)
            try:
                await work(job)
            except asyncio.CancelledError:
                self.update(job, status=CANCELLED, error="cancelled")
                raise
            except Exception as exc:  # surfaced to the client, never swallowed
                self.update(job, status=FAILED, error=str(exc)[:2000])

        job.task = asyncio.create_task(wrapper())
        return job

    async def cancel(self, job: Job) -> bool:
        if job.status in TERMINAL:
            return False
        if job.task:
            job.task.cancel()
        self.update(job, status=CANCELLED, error="cancelled")
        return True

    async def wait(self, job: Job, timeout: float) -> bool:
        try:
            await asyncio.wait_for(job.done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return job.status in TERMINAL

    def sweep(self) -> int:
        """Forget expired jobs and their cached images. Returns how many."""
        cutoff = time.time() - self.ttl
        stale = [j for j in self._jobs.values() if j.updated_at < cutoff]
        for job in stale:
            self._forget_blobs(job)
            self._jobs.pop(job.id, None)
        if stale:
            placeholders = ",".join("?" * len(stale))
            with self._lock:
                self._db.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})",
                                 [j.id for j in stale])
                self._db.commit()
        return len(stale)

    def close(self) -> None:
        self._db.close()
