"""Locate public source/reference data related to the MEPRS revision.

The released MEPRS package contains model outputs and scores, but not the
source sentences, references, or WMT sample IDs. This helper pulls the public
TEaR/WMT MQM files that appear to match the experiment language pairs and writes
small, inspectable CSV files for revision planning. It does not modify the
released MEPRS dataset and does not call any LLM APIs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import Request, urlopen


RAW_BASE = "https://raw.githubusercontent.com/fzp0424/self_correct_mt/main"

EXTERNAL_SOURCES = {
    "zh-en": {
        "year": "wmt23",
        "output_csv": "dataset/mqm/wmt23/zh-en/output.csv",
        "filtered_csv": "dataset/mqm/wmt23/zh-en/zh-en_final_file_filtered.csv",
        "sample_index": "dataset/mqm/wmt23/zh-en/sample.index",
        "covered_local_pairs": "zh-en-new; en-zh-new by reversing src/ref",
    },
    "en-de": {
        "year": "wmt23",
        "output_csv": "dataset/mqm/wmt23/en-de/output.csv",
        "filtered_csv": "dataset/mqm/wmt23/en-de/en-de_final_file_filtered.csv",
        "sample_index": "dataset/mqm/wmt23/en-de/sample.index",
        "covered_local_pairs": "en-de-new; de-en-new by reversing src/ref",
    },
    "en-ru": {
        "year": "wmt22",
        "output_csv": "dataset/mqm/wmt22/en-ru/output.csv",
        "filtered_csv": "dataset/mqm/wmt22/en-ru/en-ru_final_file_filtered.csv",
        "sample_index": "dataset/mqm/wmt22/en-ru/sample.index",
        "covered_local_pairs": "en-ru-new; ru-en-new by reversing src/ref",
    },
    "he-en": {
        "year": "wmt23",
        "output_csv": "dataset/mqm/wmt23/he-en/output.csv",
        "filtered_csv": "dataset/mqm/wmt23/he-en/he-en_final_file_filtered.csv",
        "sample_index": "dataset/mqm/wmt23/he-en/sample.index",
        "covered_local_pairs": "he-en-new; en-he-new by reversing src/ref",
    },
}

DIRECT_LOCAL_PAIRS = {
    "zh-en-new": "zh-en",
    "en-de-new": "en-de",
    "en-ru-new": "en-ru",
    "he-en-new": "he-en",
}

MODEL_CODES = ["G35", "G4o", "C3", "C35", "GP"]
STRATEGIES = ["it", "tear"]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[#\s]+|[#\s]+$", "", text)
    return text.strip()


def fetch_text(relative_path: str, timeout: int = 180, retries: int = 3) -> str:
    url = f"{RAW_BASE}/{relative_path}"
    request = Request(url, headers={"User-Agent": "MEPRS-revision-data-locator"})
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            chunks = []
            with urlopen(request, timeout=timeout) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
            return b"".join(chunks).decode("utf-8-sig")
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def parse_sample_index(text: str) -> dict[str, dict[str, str]]:
    rows = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            rows[parts[0]] = {
                "sample_domain": parts[1],
                "sample_id": parts[2],
            }
    return rows


def collect_unique_src_ref(filtered_rows: list[dict[str, str]], sample_rows: dict[str, dict[str, str]]) -> list[dict]:
    by_target: dict[str, dict] = {}
    for row in filtered_rows:
        target = row.get("Target_Index", "")
        if not target or target in by_target:
            continue
        sample = sample_rows.get(target, {})
        by_target[target] = {
            "target_index": target,
            "sample_domain": sample.get("sample_domain", ""),
            "sample_id": sample.get("sample_id", ""),
            "src": row.get("src", ""),
            "ref": row.get("ref", ""),
        }
    return [by_target[k] for k in sorted(by_target, key=lambda value: int(value))]


def collect_match_pool(output_rows: list[dict[str, str]], src_ref_rows: list[dict]) -> dict[str, list[str]]:
    pool: dict[str, list[str]] = {}
    src_ref_by_target = {row["target_index"]: row for row in src_ref_rows}
    for row in output_rows:
        target = row.get("Target_Index", "")
        values = []
        for key, value in row.items():
            if key not in {"id", "Target_Index"} and value:
                values.append(value)
        if target in src_ref_by_target:
            values.extend([src_ref_by_target[target]["src"], src_ref_by_target[target]["ref"]])
        pool[target] = values
    return pool


def read_local_candidate_lines(pair_dir: Path) -> list[list[str]]:
    all_files = []
    for model in MODEL_CODES:
        for strategy in STRATEGIES:
            path = pair_dir / f"{model}_{strategy}.txt"
            if path.exists():
                all_files.append(path)
    if not all_files:
        return []
    by_sentence: list[list[str]] = []
    file_lines = [path.read_text(encoding="utf-8").splitlines() for path in all_files]
    max_len = max(len(lines) for lines in file_lines)
    for idx in range(max_len):
        examples = []
        for lines in file_lines:
            if idx < len(lines) and lines[idx].strip():
                examples.append(lines[idx].strip())
        by_sentence.append(examples)
    return by_sentence


def best_external_match(
    local_examples: list[str],
    external_pool: dict[str, list[str]],
    exact_index: dict[str, list[tuple[str, str]]],
    flattened_pool: list[tuple[str, str, str]],
) -> tuple[str, float, str, str]:
    best_target = ""
    best_score = 0.0
    best_local = ""
    best_external = ""
    normalized_examples = [(text, normalize(text)) for text in local_examples]
    for local_text, local_norm in normalized_examples:
        if local_norm in exact_index:
            target, external_text = exact_index[local_norm][0]
            return target, 1.0, local_text, external_text

    for local_text, local_norm in normalized_examples:
        if not local_norm:
            continue
        for target, external_text, external_norm in flattened_pool:
            if not external_norm:
                continue
            if len(local_norm) >= 20 and (local_norm in external_norm or external_norm in local_norm):
                score = min(len(local_norm), len(external_norm)) / max(len(local_norm), len(external_norm))
                score = max(score, 0.92)
                if score > best_score:
                    best_target = target
                    best_score = score
                    best_local = local_text
                    best_external = external_text
    if best_score >= 0.92:
        return best_target, best_score, best_local, best_external

    long_examples = [(text, norm) for text, norm in normalized_examples if len(norm) >= 20]
    for local_text, local_norm in long_examples[:2]:
        for target, external_text, external_norm in flattened_pool:
            score = SequenceMatcher(None, local_norm, external_norm).ratio()
            if score > best_score:
                best_target = target
                best_score = score
                best_local = local_text
                best_external = external_text
    return best_target, best_score, best_local, best_external


def build_match_indexes(external_pool: dict[str, list[str]]) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    exact_index: dict[str, list[tuple[str, str]]] = {}
    flattened_pool = []
    for target, candidates in external_pool.items():
        for external_text in candidates:
            external_norm = normalize(external_text)
            if not external_norm:
                continue
            exact_index.setdefault(external_norm, []).append((target, external_text))
            flattened_pool.append((target, external_text, external_norm))
    return exact_index, flattened_pool


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(data_dir: Path, output_dir: Path, skip_alignment: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = []
    extracted_rows_by_direction = {}
    match_pool_by_direction = {}

    for direction, meta in EXTERNAL_SOURCES.items():
        filtered_text = fetch_text(meta["filtered_csv"])
        output_text = fetch_text(meta["output_csv"])
        sample_text = fetch_text(meta["sample_index"])

        sample_rows = parse_sample_index(sample_text)
        src_ref_rows = collect_unique_src_ref(read_csv_text(filtered_text), sample_rows)
        output_rows = read_csv_text(output_text)
        extracted_rows_by_direction[direction] = src_ref_rows
        if not skip_alignment:
            match_pool_by_direction[direction] = collect_match_pool(output_rows, src_ref_rows)

        write_csv(output_dir / f"external_{meta['year']}_{direction}_src_ref.csv", src_ref_rows)

        source_rows.append(
            {
                "external_direction": direction,
                "year": meta["year"],
                "unique_source_reference_rows": len(src_ref_rows),
                "sample_index_rows": len(sample_rows),
                "output_rows": len(output_rows),
                "covered_local_pairs": meta["covered_local_pairs"],
                "filtered_csv_url": f"{RAW_BASE}/{meta['filtered_csv']}",
                "output_csv_url": f"{RAW_BASE}/{meta['output_csv']}",
                "sample_index_url": f"{RAW_BASE}/{meta['sample_index']}",
            }
        )

    alignment_rows = []
    coverage_rows = []
    if skip_alignment:
        write_csv(output_dir / "external_data_source_inventory.csv", source_rows)
        metadata = {
            "raw_base": RAW_BASE,
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "external_sources": EXTERNAL_SOURCES,
            "matching_note": "Alignment was skipped. Run without --skip_alignment to create candidate matches.",
        }
        (output_dir / "external_data_source_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    for local_pair, direction in DIRECT_LOCAL_PAIRS.items():
        pair_dir = data_dir / local_pair
        local_by_sentence = read_local_candidate_lines(pair_dir)
        external_pool = match_pool_by_direction[direction]
        exact_index, flattened_pool = build_match_indexes(external_pool)
        src_ref_by_target = {
            row["target_index"]: row for row in extracted_rows_by_direction[direction]
        }
        high_confidence = 0
        medium_confidence = 0
        for idx, examples in enumerate(local_by_sentence):
            target, score, local_text, external_text = best_external_match(
                examples, external_pool, exact_index, flattened_pool
            )
            if score >= 0.85:
                high_confidence += 1
            if score >= 0.72:
                medium_confidence += 1
            src_ref = src_ref_by_target.get(target, {})
            alignment_rows.append(
                {
                    "local_pair": local_pair,
                    "local_sentence_index_0based": idx,
                    "external_direction": direction,
                    "best_target_index": target,
                    "best_similarity": round(score, 4),
                    "external_sample_domain": src_ref.get("sample_domain", ""),
                    "external_sample_id": src_ref.get("sample_id", ""),
                    "external_src": src_ref.get("src", ""),
                    "external_ref": src_ref.get("ref", ""),
                    "local_matched_text": local_text,
                    "external_matched_text": external_text,
                }
            )
        coverage_rows.append(
            {
                "local_pair_checked": local_pair,
                "external_direction": direction,
                "local_sentence_count": len(local_by_sentence),
                "external_unique_rows": len(extracted_rows_by_direction[direction]),
                "high_confidence_matches_ge_0_85": high_confidence,
                "medium_confidence_matches_ge_0_72": medium_confidence,
                "reverse_pair_can_reuse_indices": {
                    "zh-en-new": "en-zh-new",
                    "en-de-new": "de-en-new",
                    "en-ru-new": "ru-en-new",
                    "he-en-new": "en-he-new",
                }[local_pair],
                "note": "This is a conservative text-similarity locator, not a final audited alignment.",
            }
        )

    write_csv(output_dir / "external_data_source_inventory.csv", source_rows)
    write_csv(output_dir / "external_alignment_candidates.csv", alignment_rows)
    write_csv(output_dir / "external_alignment_coverage_summary.csv", coverage_rows)

    metadata = {
        "raw_base": RAW_BASE,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "external_sources": EXTERNAL_SOURCES,
        "matching_note": (
            "Direct directions were matched by text similarity against public WMT/TEaR MQM outputs. "
            "Reverse directions should reuse the same sentence indices after the direct alignment is audited."
        ),
    }
    (output_dir / "external_data_source_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
        help="MEPRS dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Output directory for locator CSV files",
    )
    parser.add_argument(
        "--skip_alignment",
        action="store_true",
        help="Only download/extract public source-reference rows; skip local text matching.",
    )
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.skip_alignment)


if __name__ == "__main__":
    main()
