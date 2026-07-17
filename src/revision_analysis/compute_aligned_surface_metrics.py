from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


MODEL_CODES = ["G35", "G4o", "C3", "C35", "GP"]
STRATEGIES = ["it", "tear"]
REVERSE_PAIRS = {
    "zh-en-new": "en-zh-new",
    "en-de-new": "de-en-new",
    "en-ru-new": "ru-en-new",
    "he-en-new": "en-he-new",
}


def read_csv(path: Path) -> list[dict[str, str]]:
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


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def read_scores(path: Path) -> np.ndarray:
    return np.asarray([float(line.split()[0]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
    return tokens or list(text.strip())


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def sentence_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    cand = tokenize(candidate)
    ref = tokenize(reference)
    if not cand or not ref:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        cand_ng = ngrams(cand, n)
        ref_ng = ngrams(ref, n)
        if not cand_ng:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_ng[gram]) for gram, count in cand_ng.items())
        precisions.append((overlap + 1.0) / (sum(cand_ng.values()) + 1.0))
    geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(cand) > len(ref) else math.exp(1.0 - len(ref) / max(1, len(cand)))
    return 100.0 * bp * geo


def char_ngrams(text: str, n: int) -> Counter[str]:
    compact = re.sub(r"\s+", " ", text.strip().lower())
    if len(compact) < n:
        return Counter([compact]) if compact else Counter()
    return Counter(compact[i : i + n] for i in range(len(compact) - n + 1))


def sentence_chrf(candidate: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    scores = []
    for n in range(1, max_n + 1):
        cand_ng = char_ngrams(candidate, n)
        ref_ng = char_ngrams(reference, n)
        if not cand_ng or not ref_ng:
            scores.append(0.0)
            continue
        overlap = sum(min(count, ref_ng[gram]) for gram, count in cand_ng.items())
        precision = overlap / max(1, sum(cand_ng.values()))
        recall = overlap / max(1, sum(ref_ng.values()))
        if precision == 0 and recall == 0:
            scores.append(0.0)
        else:
            beta2 = beta * beta
            scores.append((1 + beta2) * precision * recall / (beta2 * precision + recall))
    return 100.0 * sum(scores) / len(scores)


def load_alignment(output_dir: Path, min_similarity: float) -> dict[str, dict[int, dict[str, str]]]:
    fast_path = output_dir / "fast_external_alignment_candidates.csv"
    path = fast_path if fast_path.exists() else output_dir / "external_alignment_candidates.csv"
    rows = read_csv(path)
    by_pair: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        if float(row["best_similarity"]) < min_similarity:
            continue
        pair = row["local_pair"]
        idx = int(row["local_sentence_index_0based"])
        by_pair.setdefault(pair, {})[idx] = row
    if any(pair in by_pair for pair in REVERSE_PAIRS.values()):
        return by_pair
    for direct, reverse in REVERSE_PAIRS.items():
        if direct in by_pair:
            rev_rows = {}
            for idx, row in by_pair[direct].items():
                copied = dict(row)
                copied["local_pair"] = reverse
                copied["external_src"], copied["external_ref"] = row["external_ref"], row["external_src"]
                rev_rows[idx] = copied
            by_pair[reverse] = rev_rows
    return by_pair


def selected_model_by_meprs(pair_dir: Path, strategy: str, sent_idx: int) -> str:
    totals = {}
    for model in MODEL_CODES:
        total = 0.0
        for reviewer in MODEL_CODES:
            scores = read_scores(pair_dir / f"{model}_{strategy}_{reviewer}.score")
            total += float(scores[sent_idx])
        totals[model] = total
    return max(totals, key=totals.get)


def run(dataset_dir: Path, alignment_dir: Path, output_dir: Path, min_similarity: float) -> None:
    alignment = load_alignment(alignment_dir, min_similarity)
    detail_rows = []
    summary_rows = []
    for pair_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        pair = pair_dir.name
        pair_alignment = alignment.get(pair, {})
        if not pair_alignment:
            continue
        for strategy in STRATEGIES:
            metric_by_model = {model: read_scores(pair_dir / f"{model}_{strategy}.bleurt") for model in MODEL_CODES}
            candidate_by_model = {model: read_lines(pair_dir / f"{model}_{strategy}.txt") for model in MODEL_CODES}
            best_baseline = max(MODEL_CODES, key=lambda m: float(metric_by_model[m].mean()))
            per_system = {model: {"bleu": [], "chrf": []} for model in [*MODEL_CODES, "MEPRS"]}
            for idx, row in sorted(pair_alignment.items()):
                ref = row["external_ref"]
                if not ref:
                    continue
                meprs_model = selected_model_by_meprs(pair_dir, strategy, idx)
                for system in [*MODEL_CODES, "MEPRS"]:
                    model = meprs_model if system == "MEPRS" else system
                    hyp = candidate_by_model[model][idx]
                    bleu = sentence_bleu(hyp, ref)
                    chrf = sentence_chrf(hyp, ref)
                    per_system[system]["bleu"].append(bleu)
                    per_system[system]["chrf"].append(chrf)
                    detail_rows.append(
                        {
                            "pair": pair.replace("-new", ""),
                            "strategy": strategy,
                            "sentence_index_0based": idx,
                            "system": system,
                            "underlying_model": model,
                            "alignment_similarity": row["best_similarity"],
                            "sample_id": row["external_sample_id"],
                            "bleu": bleu,
                            "chrf": chrf,
                        }
                    )
            for system, values in per_system.items():
                if not values["bleu"]:
                    continue
                summary_rows.append(
                    {
                        "pair": pair.replace("-new", ""),
                        "strategy": strategy,
                        "system": system,
                        "is_best_bleurt_baseline": system == best_baseline,
                        "aligned_sentence_count": len(values["bleu"]),
                        "bleu_mean": float(np.mean(values["bleu"])),
                        "chrf_mean": float(np.mean(values["chrf"])),
                    }
                )
    write_csv(output_dir / "aligned_surface_metric_details.csv", detail_rows)
    write_csv(output_dir / "aligned_surface_metric_summary.csv", summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset_dir", type=Path, default=root / "dataset")
    parser.add_argument("--alignment_dir", type=Path, default=root / "revision_analysis" / "outputs")
    parser.add_argument("--output_dir", type=Path, default=root / "revision_analysis" / "surface_metric_outputs")
    parser.add_argument("--min_similarity", type=float, default=0.92)
    args = parser.parse_args()
    run(args.dataset_dir, args.alignment_dir, args.output_dir, args.min_similarity)


if __name__ == "__main__":
    main()
