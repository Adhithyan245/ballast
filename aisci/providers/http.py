"""Shared HTTP plumbing for the API providers. Keys travel only in headers and
are never included in logged errors."""
from __future__ import annotations

import os

import httpx

from .base import FatalError, ProviderUnavailable, QuotaExhausted, RetryableError


def require_key(var: str) -> str:
    key = os.environ.get(var, "").strip()
    if not key:
        raise ProviderUnavailable(f"{var} is not set (see .env.example)")
    return key


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None


def post_json(client: httpx.Client, url: str, headers: dict, body: dict, timeout: float,
              is_daily_quota, retry_after=None) -> tuple[dict, httpx.Headers]:
    """POST and classify failures. `is_daily_quota(resp)` decides whether a 429
    means the day's quota is gone (stop) or a per-minute limit (retry);
    `retry_after(resp)` optionally reads a provider-specific wait hint."""
    retry_after = retry_after or _retry_after
    try:
        resp = client.post(url, headers=headers, json=body, timeout=timeout)
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise RetryableError(f"transport: {type(e).__name__}") from None
    if resp.status_code == 200:
        return resp.json(), resp.headers
    detail = resp.text[:500]
    if resp.status_code == 429:
        if is_daily_quota(resp):
            raise QuotaExhausted(f"429 daily quota: {detail}")
        raise RetryableError(f"429: {detail}", retry_after=retry_after(resp))
    if resp.status_code in (401, 403):
        raise ProviderUnavailable(f"{resp.status_code}: {detail}")
    if resp.status_code == 402:  # payment required / credits used up
        raise QuotaExhausted(f"402: {detail}")
    if resp.status_code >= 500 or resp.status_code == 408:
        raise RetryableError(f"{resp.status_code}: {detail}", retry_after=retry_after(resp))
    raise FatalError(f"{resp.status_code}: {detail}")


def get_json(url: str, headers: dict, timeout: float = 30.0) -> tuple[dict, httpx.Headers]:
    resp = httpx.get(url, headers=headers, timeout=timeout)
    if resp.status_code in (401, 403):
        raise ProviderUnavailable(f"{resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    return resp.json(), resp.headers
