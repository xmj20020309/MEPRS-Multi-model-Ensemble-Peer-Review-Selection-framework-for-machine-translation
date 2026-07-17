from __future__ import annotations

import argparse
import csv
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def load_items(input_path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = read_csv(input_path)
    items = [
        row
        for row in rows
        if truthy(row.get("sent_to_human")) and not truthy(row.get("exact_identical_pair"))
    ]
    if limit is not None:
        items = items[:limit]
    return items


def load_completed(output_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not output_path.exists():
        return [], set()
    rows = read_csv(output_path)
    ok_by_item: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "ok":
            ok_by_item[row["item_id"]] = row
    return list(ok_by_item.values()), set(ok_by_item)


class RateLimiter:
    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self._starts: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._starts and now - self._starts[0] >= 60.0:
                    self._starts.popleft()
                if len(self._starts) < self.rpm:
                    self._starts.append(now)
                    return
                wait_s = 60.0 - (now - self._starts[0])
            time.sleep(max(0.05, min(wait_s, 1.0)))


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, timeout: int, limiter: RateLimiter) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.limiter = limiter

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.limiter.acquire()
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
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
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))


def build_prompt(item: dict[str, str]) -> list[dict[str, str]]:
    user = (
        "Evaluate two candidate machine translations. Use the source sentence and "
        "the reference translation as context. Return JSON only with these keys: "
        "`preference` (A, B, or Tie), `adequacy_A_1_5`, `fluency_A_1_5`, "
        "`adequacy_B_1_5`, `fluency_B_1_5`, and `brief_reason`. Prefer Tie when "
        "the two candidates are equivalent or the difference is not meaningful.\n\n"
        f"Language pair: {item.get('pair', '')}\n"
        f"Prompting strategy: {item.get('strategy', '')}\n\n"
        f"Source:\n{item.get('source_text', '')}\n\n"
        f"Reference:\n{item.get('reference_translation', '')}\n\n"
        f"Candidate A:\n{item.get('candidate_A_text', '')}\n\n"
        f"Candidate B:\n{item.get('candidate_B_text', '')}"
    )
    return [
        {
            "role": "system",
            "content": "You are a careful bilingual machine-translation evaluator.",
        },
        {"role": "user", "content": user},
    ]


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start : end + 1])


def normalize_preference(value: Any) -> str:
    pref = str(value or "").strip().upper()
    if pref.startswith("A"):
        return "A"
    if pref.startswith("B"):
        return "B"
    if "TIE" in pref or pref in {"EQUAL", "NONE"}:
        return "Tie"
    return pref


def score_item(
    item: dict[str, str],
    client: OpenAICompatibleClient,
    model: str,
    group: str,
    temperature: float,
    max_retries: int,
    backoff_base: float,
    backoff_cap: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": build_prompt(item),
        "temperature": temperature,
        "max_tokens": 450,
        "response_format": {"type": "json_object"},
    }
    if group:
        payload["group"] = group
    last_error = ""
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        try:
            response = client.chat(payload)
            content = response["choices"][0]["message"]["content"]
            parsed = parse_json_object(content)
            preference = normalize_preference(parsed.get("preference"))
            winner_role = ""
            winner_model = ""
            winner_model_name = ""
            if preference == "A":
                winner_role = item.get("candidate_A_role", "")
                winner_model = item.get("candidate_A_model", "")
                winner_model_name = item.get("candidate_A_model_name", "")
            elif preference == "B":
                winner_role = item.get("candidate_B_role", "")
                winner_model = item.get("candidate_B_model", "")
                winner_model_name = item.get("candidate_B_model_name", "")
            elif preference == "Tie":
                winner_role = "Tie"
            return {
                "item_id": item.get("item_id", ""),
                "row_no": item.get("row_no", ""),
                "pair": item.get("pair", ""),
                "strategy": item.get("strategy", ""),
                "sentence_index_0based": item.get("sentence_index_0based", ""),
                "sample_id": item.get("sample_id", ""),
                "status": "ok",
                "model": model,
                "attempts": attempts,
                "candidate_A_role": item.get("candidate_A_role", ""),
                "candidate_A_model": item.get("candidate_A_model", ""),
                "candidate_B_role": item.get("candidate_B_role", ""),
                "candidate_B_model": item.get("candidate_B_model", ""),
                "llm_preference": preference,
                "llm_winner_role": winner_role,
                "llm_winner_model": winner_model,
                "llm_winner_model_name": winner_model_name,
                "adequacy_A_1_5": parsed.get("adequacy_A_1_5", ""),
                "fluency_A_1_5": parsed.get("fluency_A_1_5", ""),
                "adequacy_B_1_5": parsed.get("adequacy_B_1_5", ""),
                "fluency_B_1_5": parsed.get("fluency_B_1_5", ""),
                "brief_reason": parsed.get("brief_reason", ""),
                "response_json": content,
                "usage_json": json.dumps(response.get("usage", {}), ensure_ascii=False),
                "error": "",
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code in {400, 401, 403, 404}:
                break
        except Exception as exc:  # noqa: BLE001 - keep exact API/network error.
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries:
            delay = min(backoff_cap, backoff_base * (2**attempt))
            delay += random.uniform(0.0, min(1.0, delay * 0.25))
            time.sleep(delay)
    return {
        "item_id": item.get("item_id", ""),
        "row_no": item.get("row_no", ""),
        "pair": item.get("pair", ""),
        "strategy": item.get("strategy", ""),
        "sentence_index_0based": item.get("sentence_index_0based", ""),
        "sample_id": item.get("sample_id", ""),
        "status": "failed",
        "model": model,
        "attempts": attempts,
        "candidate_A_role": item.get("candidate_A_role", ""),
        "candidate_A_model": item.get("candidate_A_model", ""),
        "candidate_B_role": item.get("candidate_B_role", ""),
        "candidate_B_model": item.get("candidate_B_model", ""),
        "llm_preference": "",
        "llm_winner_role": "",
        "llm_winner_model": "",
        "llm_winner_model_name": "",
        "adequacy_A_1_5": "",
        "fluency_A_1_5": "",
        "adequacy_B_1_5": "",
        "fluency_B_1_5": "",
        "brief_reason": "",
        "response_json": "",
        "usage_json": "",
        "error": last_error,
    }


FIELDNAMES = [
    "item_id",
    "row_no",
    "pair",
    "strategy",
    "sentence_index_0based",
    "sample_id",
    "status",
    "model",
    "attempts",
    "candidate_A_role",
    "candidate_A_model",
    "candidate_B_role",
    "candidate_B_model",
    "llm_preference",
    "llm_winner_role",
    "llm_winner_model",
    "llm_winner_model_name",
    "adequacy_A_1_5",
    "fluency_A_1_5",
    "adequacy_B_1_5",
    "fluency_B_1_5",
    "brief_reason",
    "response_json",
    "usage_json",
    "error",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast concurrent A/B MT judge runner.")
    parser.add_argument("--input", type=Path, default=Path("final_result_v2.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("revision_analysis/latest_model_ab_528"))
    parser.add_argument("--base_url", default=os.environ.get("BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key_env", default="API_KEY")
    parser.add_argument("--model", default=os.environ.get("MODEL", "gpt-4o-mini"))
    parser.add_argument("--group", default=os.environ.get("GROUP", ""))
    parser.add_argument("--rpm", type=int, default=300)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--backoff_base", type=float, default=1.0)
    parser.add_argument("--backoff_cap", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all eligible rows.")
    parser.add_argument("--run", action="store_true", help="Actually call the API.")
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    items = load_items(args.input, limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "llm_ab_judge_results.csv"
    plan_path = args.out_dir / "llm_ab_judge_plan.csv"

    if not args.run:
        plan_rows = [
            {
                "item_id": item.get("item_id", ""),
                "row_no": item.get("row_no", ""),
                "pair": item.get("pair", ""),
                "strategy": item.get("strategy", ""),
                "candidate_A_role": item.get("candidate_A_role", ""),
                "candidate_B_role": item.get("candidate_B_role", ""),
                "status": "planned",
            }
            for item in items
        ]
        write_rows(
            plan_path,
            plan_rows,
            [
                "item_id",
                "row_no",
                "pair",
                "strategy",
                "candidate_A_role",
                "candidate_B_role",
                "status",
            ],
        )
        print(plan_path)
        print(f"planned_items={len(plan_rows)}")
        return

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    existing_rows, completed = load_completed(output_path)
    todo = [item for item in items if item.get("item_id", "") not in completed]
    print(
        f"items={len(items)} completed={len(completed)} todo={len(todo)} "
        f"workers={args.workers} rpm={args.rpm}"
    )
    if not todo:
        print(output_path)
        return

    rows = existing_rows[:]
    rows_lock = threading.Lock()
    limiter = RateLimiter(args.rpm)
    client = OpenAICompatibleClient(api_key, args.base_url, args.timeout, limiter)
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                score_item,
                item,
                client,
                args.model,
                args.group,
                args.temperature,
                args.max_retries,
                args.backoff_base,
                args.backoff_cap,
            )
            for item in todo
        ]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with rows_lock:
                rows.append(result)
                write_rows(output_path, rows, FIELDNAMES)
            if index == 1 or index % 25 == 0 or index == len(futures):
                elapsed = time.time() - start
                ok = sum(1 for row in rows if row.get("status") == "ok")
                failed = sum(1 for row in rows if row.get("status") == "failed")
                print(
                    f"done={index}/{len(futures)} total_ok={ok} "
                    f"total_failed={failed} elapsed_s={elapsed:.1f}"
                )
    print(output_path)


if __name__ == "__main__":
    main()
