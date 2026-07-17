from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


class RateLimiter:
    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self.starts: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        while True:
            with self.lock:
                now = time.monotonic()
                while self.starts and now - self.starts[0] >= 60.0:
                    self.starts.popleft()
                if len(self.starts) < self.rpm:
                    self.starts.append(now)
                    return
                wait_s = 60.0 - (now - self.starts[0])
            time.sleep(max(0.02, min(wait_s, 0.5)))


def call_api(
    index: int,
    base_url: str,
    api_key: str,
    model: str,
    group: str,
    timeout: int,
    max_retries: int,
    limiter: RateLimiter,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "temperature": 0,
        "max_tokens": 4,
    }
    if group:
        payload["group"] = group
    body = json.dumps(payload).encode("utf-8")
    last_error = ""
    started = time.time()
    for attempt in range(max_retries + 1):
        try:
            limiter.acquire()
            req_started = time.time()
            req = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0.0.0 Safari/537.36"
                    ),
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
            elapsed = time.time() - req_started
            obj = json.loads(raw)
            content = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "index": index,
                "status": "ok",
                "attempts": attempt + 1,
                "latency_s": round(elapsed, 3),
                "wall_s": round(time.time() - started, 3),
                "content": content,
                "usage_json": json.dumps(obj.get("usage", {}), ensure_ascii=False),
                "error": "",
            }
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code}: {text}"
            if exc.code in {400, 401, 403, 404}:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries:
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    return {
        "index": index,
        "status": "failed",
        "attempts": max_retries + 1,
        "latency_s": "",
        "wall_s": round(time.time() - started, 3),
        "content": "",
        "usage_json": "",
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic API concurrency probe.")
    parser.add_argument("--base_url", default=os.environ.get("BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key_env", default="API_KEY")
    parser.add_argument("--model", default=os.environ.get("MODEL", "gpt-4o-mini"))
    parser.add_argument("--group", default=os.environ.get("GROUP", ""))
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--rpm", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("revision_analysis/api_concurrency_probe.csv"))
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(args.rpm)
    rows: list[dict[str, Any]] = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                call_api,
                i,
                args.base_url,
                api_key,
                args.model,
                args.group,
                args.timeout,
                args.max_retries,
                limiter,
            )
            for i in range(1, args.requests + 1)
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "index",
                        "status",
                        "attempts",
                        "latency_s",
                        "wall_s",
                        "content",
                        "usage_json",
                        "error",
                    ],
                )
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda r: int(r["index"])))

    elapsed = time.time() - start
    ok = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] != "ok"]
    latencies = sorted(float(row["latency_s"]) for row in ok if row["latency_s"] != "")
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0
    print(f"requests={args.requests} workers={args.workers} rpm={args.rpm}")
    print(f"ok={len(ok)} failed={len(failed)} elapsed_s={elapsed:.2f}")
    print(f"effective_rpm={len(ok) / elapsed * 60:.1f}" if elapsed else "effective_rpm=0")
    print(f"latency_p50_s={p50:.2f} latency_p95_s={p95:.2f}")
    if failed:
        print("first_error=" + str(failed[0].get("error", ""))[:300])
    print(args.out)


if __name__ == "__main__":
    main()
