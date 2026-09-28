"""Thin async client for the ComfyUI server API.

Only the five endpoints the service actually needs: upload an input image,
submit a prompt, poll history, read the queue, fetch the rendered file.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx


class ComfyError(RuntimeError):
    """ComfyUI refused a request, or answered something unexpected."""


def decode_json(response: httpx.Response) -> Any:
    """ComfyUI answers 400s with a JSON *string* holding JSON; unwrap that."""
    try:
        payload = response.json()
    except ValueError:
        return {"raw": response.text[:400]}
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except ValueError:
            return {"raw": payload[:400]}
    return payload


def format_node_errors(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)[:300]
    errors = payload.get("node_errors") or {}
    lines: list[str] = []
    for node_id, detail in errors.items():
        message = (detail or {}).get("errors") or []
        for entry in message:
            if isinstance(entry, dict):
                lines.append(f"node {node_id}: {entry.get('message') or entry.get('type')}")
            else:
                lines.append(f"node {node_id}: {entry}")
    return "; ".join(lines) or "no detail"


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"User-Agent": "qwen-api/2.0"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, **params: Any) -> Any:
        try:
            response = await self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise ComfyError(f"ComfyUI unreachable at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            detail = format_node_errors(decode_json(response))
            raise ComfyError(f"GET {path} -> {response.status_code}: {detail or response.text[:200]}")
        return response.json()

    # ------------------------------------------------------------- read-only

    async def system_stats(self) -> dict[str, Any]:
        return await self._get("/system_stats")

    async def object_info(self, class_type: str) -> dict[str, Any] | None:
        try:
            response = await self._client.get(f"/object_info/{class_type}")
        except httpx.HTTPError:
            return None
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            return None
        return response.json().get(class_type)

    async def models(self, folder: str) -> list[str]:
        payload = await self._get(f"/models/{folder}")
        return list(payload or [])

    async def queue_state(self) -> dict[str, Any]:
        return await self._get("/queue")

    async def history(self, prompt_id: str) -> dict[str, Any] | None:
        payload = await self._get(f"/history/{prompt_id}")
        return (payload or {}).get(prompt_id)

    # ---------------------------------------------------------------- writes

    async def upload_image(self, data: bytes, filename: str, overwrite: bool = True) -> str:
        files = {"image": (filename, data, "application/octet-stream")}
        data_fields = {"type": "input", "overwrite": str(overwrite).lower()}
        try:
            response = await self._client.post("/upload/image", files=files, data=data_fields)
        except httpx.HTTPError as exc:
            raise ComfyError(f"upload failed: {exc}") from exc
        if response.status_code >= 400:
            raise ComfyError(f"upload failed: {response.status_code} {format_node_errors(decode_json(response))}")
        payload = response.json()
        name = payload.get("name")
        if not name:
            raise ComfyError(f"upload returned no name: {response.text[:200]}")
        return name

    async def queue_prompt(self, prompt: dict[str, Any], client_id: str) -> str:
        payload = {"prompt": prompt, "client_id": client_id}
        try:
            response = await self._client.post("/prompt", json=payload)
        except httpx.HTTPError as exc:
            raise ComfyError(f"queue_prompt failed: {exc}") from exc
        if response.status_code >= 400:
            detail = format_node_errors(decode_json(response))
            raise ComfyError(f"ComfyUI rejected the workflow: {detail}")
        prompt_id = response.json().get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"no prompt_id in response: {response.text[:200]}")
        return prompt_id

    async def view(self, filename: str, subfolder: str = "", type_: str = "output") -> bytes:
        try:
            response = await self._client.get(
                "/view", params={"filename": filename, "subfolder": subfolder, "type": type_}
            )
        except httpx.HTTPError as exc:
            raise ComfyError(f"view failed: {exc}") from exc
        if response.status_code >= 400:
            raise ComfyError(f"view {filename} -> {response.status_code}")
        return response.content

    async def interrupt(self) -> None:
        try:
            await self._client.post("/interrupt")
        except httpx.HTTPError:
            pass

    # ------------------------------------------------------------- composed

    async def wait_for_result(
        self, prompt_id: str, timeout: float, poll: float = 1.5
    ) -> dict[str, Any]:
        """Block until the prompt leaves ComfyUI's history, or time out."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            record = await self.history(prompt_id)
            if record:
                return record
            await asyncio.sleep(poll)
        raise ComfyError(f"timed out after {timeout:.0f}s waiting for {prompt_id}")

    async def fetch_outputs(self, record: dict[str, Any]) -> list[dict[str, str]]:
        """Every image SaveImage produced, in node order."""
        images: list[dict[str, str]] = []
        for node_output in (record.get("outputs") or {}).values():
            for entry in node_output.get("images", []) or []:
                if entry.get("type") in (None, "output", "temp"):
                    images.append(
                        {
                            "filename": entry.get("filename", ""),
                            "subfolder": entry.get("subfolder", ""),
                            "type": entry.get("type", "output"),
                        }
                    )
        return images


def history_error(record: dict[str, Any]) -> str | None:
    status = (record.get("status") or {}).get("status_str")
    if status == "success":
        return None
    messages = (record.get("status") or {}).get("messages") or []
    for entry in messages:
        if isinstance(entry, list) and len(entry) > 1 and entry[0] == "execution_error":
            detail = entry[1]
            return f"{detail.get('node_type', 'node')} {detail.get('node_id', '?')}: " \
                   f"{detail.get('exception_message', 'execution error')}"
    return status or "execution failed"
