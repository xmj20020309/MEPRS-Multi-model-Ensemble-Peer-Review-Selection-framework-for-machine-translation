from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL_CODES = ["G35", "G4o", "C3", "C35", "GP"]
MODEL_NAMES = {
    "G35": "GPT-3.5-Turbo",
    "G4o": "GPT-4o",
    "C3": "Claude-3-Opus",
    "C35": "Claude-3.5-Sonnet",
    "GP": "Gemini-Pro",
}


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def read_scores(path: Path) -> list[float]:
    return [float(line.split()[0]) for line in read_lines(path) if line.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, timeout: int, max_retries: int) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        last_error = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                self.base_url + path,
                data=body,
                headers=headers,
                method=method,
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                text = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code}: {text[:1000]}") from exc
            except Exception as exc:  # noqa: BLE001 - preserve network details in result files.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"{type(last_error).__name__}: {last_error}") from last_error

    def models(self) -> dict:
        return self.request("GET", "/models")

    def chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> dict:
        return self.request(
            "POST",
            "/chat/completions",
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            },
        )


def build_overall_prompt(source: str, reference: str, candidates: list[tuple[str, str]]) -> list[dict[str, str]]:
    candidate_text = "\n".join(f"{label}. {text}" for label, text in candidates)
    user = (
        "Evaluate the candidate machine translations. Return JSON only with keys "
        "`best_label`, `scores`, and `brief_reason`. `scores` should map each label "
        "to an overall quality score from 0 to 100.\n\n"
        f"Source:\n{source}\n\n"
        f"Reference:\n{reference}\n\n"
        f"Candidates:\n{candidate_text}"
    )
    return [
        {"role": "system", "content": "You are a careful bilingual machine-translation evaluator."},
        {"role": "user", "content": user},
    ]


def prepare_micro_items(root: Path, n_items: int, seed: int) -> list[dict[str, object]]:
    cases = read_csv(root / "revision_analysis" / "remaining_gap_audit_outputs" / "qualitative_error_cases_top_aligned.csv")
    rng = random.Random(seed)
    selected = cases[:]
    rng.shuffle(selected)
    selected = selected[:n_items]
    items = []
    for item in selected:
        pair_dir = root / "dataset" / f"{item['pair']}-new"
        strategy = item["strategy"]
        sent = int(item["sentence_index_0based"])
        candidates = [(model, read_lines(pair_dir / f"{model}_{strategy}.txt")[sent]) for model in MODEL_CODES]
        rng.shuffle(candidates)
        labels = [chr(ord("A") + i) for i in range(len(candidates))]
        labeled = list(zip(labels, [model for model, _ in candidates], [text for _, text in candidates]))
        items.append(
            {
                "pair": item["pair"],
                "strategy": strategy,
                "sentence_index_0based": sent,
                "source_text": item["source_text"],
                "reference_translation": item["reference_translation"],
                "candidate_labels": ";".join(f"{label}={model}" for label, model, _ in labeled),
                "prompt_candidates": [(label, text) for label, _, text in labeled],
            }
        )
    return items


def run_probe(args: argparse.Namespace) -> None:
    key = os.environ.get(args.api_key_env, "")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict[str, object]] = [
        {
            "check": "api_key_present",
            "status": bool(key),
            "detail": f"env={args.api_key_env}; length={len(key) if key else 0}",
        },
        {"check": "base_url", "status": True, "detail": args.base_url},
    ]
    if not args.run:
        status_rows.append({"check": "dry_run", "status": True, "detail": "No API request sent. Add --run to call."})
        write_csv(out_dir / "api_probe_status.csv", status_rows)
        print(out_dir / "api_probe_status.csv")
        return
    if not key:
        status_rows.append({"check": "models", "status": False, "detail": "missing api key"})
        write_csv(out_dir / "api_probe_status.csv", status_rows)
        return

    client = OpenAICompatibleClient(key, args.base_url, args.timeout, args.max_retries)
    try:
        models = client.models()
        ids = [row.get("id", "") for row in models.get("data", [])]
        status_rows.append({"check": "models", "status": True, "detail": ";".join(ids[:20])})
    except Exception as exc:  # noqa: BLE001
        status_rows.append({"check": "models", "status": False, "detail": str(exc)})
    write_csv(out_dir / "api_probe_status.csv", status_rows)
    print(out_dir / "api_probe_status.csv")


def run_micro_eval(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    key = os.environ.get(args.api_key_env, "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.run:
        items = prepare_micro_items(root, args.n_items, args.seed)
        rows = [
            {
                "item_id": i + 1,
                "pair": item["pair"],
                "strategy": item["strategy"],
                "sentence_index_0based": item["sentence_index_0based"],
                "candidate_labels": item["candidate_labels"],
                "status": "dry_run",
            }
            for i, item in enumerate(items)
        ]
        write_csv(args.out_dir / "micro_eval_plan.csv", rows)
        print(args.out_dir / "micro_eval_plan.csv")
        return
    if not key:
        write_csv(args.out_dir / "micro_eval_results.csv", [{"status": "failed", "error": "missing API key"}])
        return

    client = OpenAICompatibleClient(key, args.base_url, args.timeout, args.max_retries)
    items = prepare_micro_items(root, args.n_items, args.seed)
    existing = read_csv(args.out_dir / "micro_eval_results.csv")
    done = {row["item_id"] for row in existing if row.get("status") == "ok"}
    rows: list[dict[str, object]] = existing[:]
    for item_id, item in enumerate(items, 1):
        if str(item_id) in done:
            continue
        messages = build_overall_prompt(
            str(item["source_text"]),
            str(item["reference_translation"]),
            item["prompt_candidates"],  # type: ignore[arg-type]
        )
        try:
            response = client.chat(args.model, messages, temperature=args.temperature)
            content = response["choices"][0]["message"]["content"]
            rows.append(
                {
                    "item_id": item_id,
                    "pair": item["pair"],
                    "strategy": item["strategy"],
                    "sentence_index_0based": item["sentence_index_0based"],
                    "candidate_labels": item["candidate_labels"],
                    "status": "ok",
                    "model": args.model,
                    "response_json": content,
                    "usage_json": json.dumps(response.get("usage", {}), ensure_ascii=False),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "item_id": item_id,
                    "pair": item["pair"],
                    "strategy": item["strategy"],
                    "sentence_index_0based": item["sentence_index_0based"],
                    "candidate_labels": item["candidate_labels"],
                    "status": "failed",
                    "model": args.model,
                    "response_json": "",
                    "usage_json": "",
                    "error": str(exc),
                }
            )
        write_csv(args.out_dir / "micro_eval_results.csv", rows)
        time.sleep(args.sleep)
    print(args.out_dir / "micro_eval_results.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe OpenAI-compatible API and optionally run a tiny MT evaluation.")
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key_env", default="API_KEY")
    parser.add_argument("--out_dir", type=Path, default=Path("revision_analysis/api_micro_eval_outputs"))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="Actually send API requests. Without this, only writes a plan.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("probe")

    micro = subparsers.add_parser("micro_eval")
    micro.add_argument("--model", default="gpt-4o-mini")
    micro.add_argument("--n_items", type=int, default=5)
    micro.add_argument("--seed", type=int, default=20260624)
    micro.add_argument("--temperature", type=float, default=0.0)
    micro.add_argument("--sleep", type=float, default=1.0)

    args = parser.parse_args()
    if args.command == "probe":
        run_probe(args)
    elif args.command == "micro_eval":
        run_micro_eval(args)


if __name__ == "__main__":
    main()
