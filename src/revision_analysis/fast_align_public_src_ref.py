from __future__ import annotations

import csv
import io
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen


RAW_BASE = "https://raw.githubusercontent.com/fzp0424/self_correct_mt/main"
EXTERNAL_SOURCES = {
    "zh-en": {
        "year": "wmt23",
        "src_ref_cache": "external_wmt23_zh-en_src_ref.csv",
        "output_cache": "external_wmt23_zh-en_output.csv",
        "output_csv": "dataset/mqm/wmt23/zh-en/output.csv",
        "filtered_csv": "dataset/mqm/wmt23/zh-en/zh-en_final_file_filtered.csv",
        "sample_index": "dataset/mqm/wmt23/zh-en/sample.index",
    },
    "en-de": {
        "year": "wmt23",
        "src_ref_cache": "external_wmt23_en-de_src_ref.csv",
        "output_cache": "external_wmt23_en-de_output.csv",
        "output_csv": "dataset/mqm/wmt23/en-de/output.csv",
        "filtered_csv": "dataset/mqm/wmt23/en-de/en-de_final_file_filtered.csv",
        "sample_index": "dataset/mqm/wmt23/en-de/sample.index",
    },
    "en-ru": {
        "year": "wmt22",
        "src_ref_cache": "external_wmt22_en-ru_src_ref.csv",
        "output_cache": "external_wmt22_en-ru_output.csv",
        "output_csv": "dataset/mqm/wmt22/en-ru/output.csv",
        "filtered_csv": "dataset/mqm/wmt22/en-ru/en-ru_final_file_filtered.csv",
        "sample_index": "dataset/mqm/wmt22/en-ru/sample.index",
    },
    "he-en": {
        "year": "wmt23",
        "src_ref_cache": "external_wmt23_he-en_src_ref.csv",
        "output_cache": "external_wmt23_he-en_output.csv",
        "output_csv": "dataset/mqm/wmt23/he-en/output.csv",
        "filtered_csv": "dataset/mqm/wmt23/he-en/he-en_final_file_filtered.csv",
        "sample_index": "dataset/mqm/wmt23/he-en/sample.index",
    },
}
DIRECT_LOCAL_PAIRS = {
    "zh-en-new": "zh-en",
    "en-de-new": "en-de",
    "en-ru-new": "en-ru",
    "he-en-new": "he-en",
}
REVERSE_LOCAL_PAIRS = {
    "zh-en-new": "en-zh-new",
    "en-de-new": "de-en-new",
    "en-ru-new": "ru-en-new",
    "he-en-new": "en-he-new",
}
MODEL_CODES = ["G35", "G4o", "C3", "C35", "GP"]
STRATEGIES = ["it", "tear"]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"your case:\s*source:\s*", "", text)
    text = re.sub(r"your case:\s*", "", text)
    text = re.sub(r"source:\s*", "", text)
    text = re.sub(r"target:\s*", "", text)
    text = re.sub(r"##\s*final [^:]+:\s*", "", text)
    text = re.sub(r"##\s*[^:]+:\s*", "", text)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'“”‘’")


def token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize(text), flags=re.UNICODE))


def fetch_text(relative_path: str, timeout: int = 120, retries: int = 3) -> str:
    url = f"{RAW_BASE}/{relative_path}"
    request = Request(url, headers={"User-Agent": "MEPRS-revision-fast-align"})
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8-sig")
        except Exception as exc:
            last_error = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def read_csv_file(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_sample_index(text: str) -> dict[str, dict[str, str]]:
    rows = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            rows[parts[0]] = {"sample_domain": parts[1], "sample_id": parts[2]}
    return rows


def collect_src_ref(filtered_rows: list[dict[str, str]], sample_rows: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    out = {}
    for row in filtered_rows:
        target = row.get("Target_Index", "")
        if not target or target in out:
            continue
        sample = sample_rows.get(target, {})
        out[target] = {
            "external_sample_domain": sample.get("sample_domain", ""),
            "external_sample_id": sample.get("sample_id", ""),
            "external_src": row.get("src", ""),
            "external_ref": row.get("ref", ""),
        }
    return out


def external_output_texts(output_rows: list[dict[str, str]], src_ref: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in output_rows:
        target = row.get("Target_Index", "")
        for key, value in row.items():
            if key in {"id", "Target_Index"} or not value:
                continue
            out.append(
                {
                    "target_index": target,
                    "external_column": key,
                    "external_text": value,
                    "external_norm": normalize(value),
                    "external_tokens": token_set(value),
                    **src_ref.get(target, {}),
                }
            )
    return out


def build_pool_indexes(pool: list[dict[str, object]]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    exact: dict[str, list[int]] = defaultdict(list)
    inverted: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(pool):
        norm = str(item["external_norm"])
        exact[norm].append(idx)
        tokens = item["external_tokens"]
        if isinstance(tokens, set):
            for token in tokens:
                if len(token) >= 2:
                    inverted[token].append(idx)
    return dict(exact), dict(inverted)


def load_local_examples(pair_dir: Path) -> list[list[tuple[str, str, str]]]:
    by_sentence: list[list[tuple[str, str, str]]] = [[] for _ in range(200)]
    for strategy in STRATEGIES:
        for model in MODEL_CODES:
            path = pair_dir / f"{model}_{strategy}.txt"
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            for idx, text in enumerate(lines[:200]):
                if text.strip():
                    by_sentence[idx].append((model, strategy, text))
    return by_sentence


def score_match(local_text: str, external: dict[str, object]) -> tuple[float, str]:
    local_norm = normalize(local_text)
    ext_norm = str(external["external_norm"])
    if not local_norm or not ext_norm:
        return 0.0, "empty"
    if local_norm == ext_norm:
        return 1.0, "exact"
    if len(local_norm) >= 20 and (local_norm in ext_norm or ext_norm in local_norm):
        ratio = min(len(local_norm), len(ext_norm)) / max(len(local_norm), len(ext_norm))
        return max(0.94, ratio), "substring"
    local_tokens = token_set(local_text)
    ext_tokens = external["external_tokens"]
    if not local_tokens or not ext_tokens:
        return 0.0, "empty_tokens"
    overlap = len(local_tokens & ext_tokens)
    precision = overlap / len(local_tokens)
    recall = overlap / len(ext_tokens)
    if precision == 0 or recall == 0:
        return 0.0, "token"
    f1 = 2 * precision * recall / (precision + recall)
    return f1, "token_f1"


def candidate_external_indices(local_text: str, exact_index: dict[str, list[int]], inverted: dict[str, list[int]], pool_size: int) -> list[int]:
    local_norm = normalize(local_text)
    if local_norm in exact_index:
        return exact_index[local_norm]
    counts: Counter[int] = Counter()
    for token in token_set(local_text):
        if len(token) >= 2:
            counts.update(inverted.get(token, []))
    if not counts:
        return list(range(pool_size))
    top = [idx for idx, _ in counts.most_common(80)]
    return top


def best_match_for_sentence(
    examples: list[tuple[str, str, str]],
    external_text_pool: list[dict[str, object]],
    exact_index: dict[str, list[int]],
    inverted: dict[str, list[int]],
) -> dict[str, object]:
    best: dict[str, object] = {
        "best_similarity": 0.0,
        "match_method": "",
        "local_model": "",
        "local_strategy": "",
        "local_matched_text": "",
        "external_matched_text": "",
        "best_target_index": "",
        "external_column": "",
        "external_sample_domain": "",
        "external_sample_id": "",
        "external_src": "",
        "external_ref": "",
    }
    for model, strategy, local_text in examples:
        for ext_idx in candidate_external_indices(local_text, exact_index, inverted, len(external_text_pool)):
            external = external_text_pool[ext_idx]
            score, method = score_match(local_text, external)
            if score > float(best["best_similarity"]):
                best = {
                    "best_similarity": round(score, 4),
                    "match_method": method,
                    "local_model": model,
                    "local_strategy": strategy,
                    "local_matched_text": local_text,
                    "external_matched_text": external["external_text"],
                    "best_target_index": external["target_index"],
                    "external_column": external["external_column"],
                    "external_sample_domain": external.get("external_sample_domain", ""),
                    "external_sample_id": external.get("external_sample_id", ""),
                    "external_src": external.get("external_src", ""),
                    "external_ref": external.get("external_ref", ""),
                }
                if score == 1.0:
                    return best
    return best


def run(dataset_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pools = {}
    inventories = []
    for direction, meta in EXTERNAL_SOURCES.items():
        output_cache = output_dir / meta["output_cache"]
        if output_cache.exists():
            output_rows = read_csv_file(output_cache)
        else:
            output_rows = read_csv_text(fetch_text(meta["output_csv"]))
            write_csv(output_cache, output_rows)

        src_ref_cache = output_dir / meta["src_ref_cache"]
        if src_ref_cache.exists():
            src_ref_rows = read_csv_file(src_ref_cache)
            src_ref = {
                row["target_index"]: {
                    "external_sample_domain": row.get("sample_domain", ""),
                    "external_sample_id": row.get("sample_id", ""),
                    "external_src": row.get("src", ""),
                    "external_ref": row.get("ref", ""),
                }
                for row in src_ref_rows
            }
        else:
            filtered_rows = read_csv_text(fetch_text(meta["filtered_csv"]))
            sample_rows = parse_sample_index(fetch_text(meta["sample_index"]))
            src_ref = collect_src_ref(filtered_rows, sample_rows)
        pools[direction] = external_output_texts(output_rows, src_ref)
        exact_index, inverted = build_pool_indexes(pools[direction])
        pools[f"{direction}__exact"] = exact_index
        pools[f"{direction}__inverted"] = inverted
        inventories.append(
            {
                "external_direction": direction,
                "year": meta["year"],
                "output_rows": len(output_rows),
                "src_ref_rows": len(src_ref),
                "external_text_candidates": len(pools[direction]),
            }
        )

    alignment_rows = []
    coverage_rows = []
    for local_pair, direction in DIRECT_LOCAL_PAIRS.items():
        pair_dir = dataset_dir / local_pair
        examples_by_sentence = load_local_examples(pair_dir)
        high = medium = 0
        for idx, examples in enumerate(examples_by_sentence):
            best = best_match_for_sentence(
                examples,
                pools[direction],
                pools[f"{direction}__exact"],
                pools[f"{direction}__inverted"],
            )
            sim = float(best["best_similarity"])
            high += sim >= 0.92
            medium += sim >= 0.75
            row = {
                "local_pair": local_pair,
                "local_sentence_index_0based": idx,
                "external_direction": direction,
                **best,
            }
            alignment_rows.append(row)
            if sim >= 0.92:
                reverse = REVERSE_LOCAL_PAIRS[local_pair]
                reverse_row = dict(row)
                reverse_row["local_pair"] = reverse
                reverse_row["external_src"], reverse_row["external_ref"] = row["external_ref"], row["external_src"]
                alignment_rows.append(reverse_row)
        coverage_rows.append(
            {
                "local_pair_checked": local_pair,
                "reverse_pair_added": REVERSE_LOCAL_PAIRS[local_pair],
                "external_direction": direction,
                "local_sentence_count": 200,
                "high_confidence_matches_ge_0_92": high,
                "medium_confidence_matches_ge_0_75": medium,
            }
        )

    write_csv(output_dir / "fast_external_data_source_inventory.csv", inventories)
    write_csv(output_dir / "fast_external_alignment_candidates.csv", alignment_rows)
    write_csv(output_dir / "fast_external_alignment_coverage_summary.csv", coverage_rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run(root / "dataset", root / "revision_analysis" / "outputs")


if __name__ == "__main__":
    main()
