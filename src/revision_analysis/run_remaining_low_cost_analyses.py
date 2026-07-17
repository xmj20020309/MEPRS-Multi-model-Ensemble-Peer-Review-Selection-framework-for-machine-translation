"""Run remaining no-API reviewer analyses from released MEPRS files.

This script adds three low-cost analyses that do not require source/reference
sentences, GPUs, or LLM calls:

1. Candidate diversity and its relationship to MEPRS gains.
2. Exhaustive model-subset robustness for the peer-review selector.
3. Single-dimension-only peer-review selection.
"""

from __future__ import annotations

import csv
import itertools
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


MODEL_CODES = ["G35", "G4o", "C3", "C35", "GP"]
MODEL_NAMES = {
    "G35": "GPT-3.5-Turbo",
    "G4o": "GPT-4o",
    "C3": "Claude-3-Opus",
    "C35": "Claude-3.5-Sonnet",
    "GP": "Gemini-Pro",
}
STRATEGIES = ["it", "tear"]
DIMENSIONS = ["accuracy", "fluency", "style", "terminology"]


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def read_scores(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(float(line.split()[0]))
    return np.asarray(values, dtype=float)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def char_ngrams(text: str, n: int) -> Counter[str]:
    clean = re.sub(r"\s+", " ", text.lower()).strip()
    if not clean:
        return Counter()
    padded = " " + clean + " "
    if len(padded) < n:
        return Counter([padded])
    return Counter(padded[i : i + n] for i in range(len(padded) - n + 1))


def f_score(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def multiset_overlap(a: Counter, b: Counter) -> int:
    return sum((a & b).values())


def chrf_like(a: str, b: str) -> float:
    scores = []
    for n in range(1, 7):
        ca = char_ngrams(a, n)
        cb = char_ngrams(b, n)
        if not ca or not cb:
            scores.append(0.0)
            continue
        overlap = multiset_overlap(ca, cb)
        precision = overlap / sum(ca.values())
        recall = overlap / sum(cb.values())
        scores.append(f_score(precision, recall, beta=2.0))
    return float(np.mean(scores)) if scores else 0.0


def token_f1(a: str, b: str) -> float:
    ca = Counter(tokenize(a))
    cb = Counter(tokenize(b))
    if not ca or not cb:
        return 0.0
    overlap = multiset_overlap(ca, cb)
    precision = overlap / sum(ca.values())
    recall = overlap / sum(cb.values())
    return f_score(precision, recall, beta=1.0)


def load_metric(pair_dir: Path, strategy: str, models: list[str] = MODEL_CODES) -> np.ndarray:
    return np.vstack([read_scores(pair_dir / f"{model}_{strategy}.bleurt") for model in models])


def load_reviewer_scores(
    pair_dir: Path,
    strategy: str,
    models_predict: list[str],
    models_eval: list[str],
    suffix: str = "score",
) -> np.ndarray:
    arr = np.zeros((len(models_eval), len(models_predict), 0), dtype=float)
    loaded = []
    for reviewer in models_eval:
        rows = []
        for candidate in models_predict:
            rows.append(read_scores(pair_dir / f"{candidate}_{strategy}_{reviewer}.{suffix}"))
        loaded.append(rows)
    return np.asarray(loaded, dtype=float)


def select(metric: np.ndarray, selection_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.argmax(selection_scores, axis=0)
    scores = metric[selected, np.arange(metric.shape[1])]
    return selected, scores


def pearson(x: list[float], y: list[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(rankdata_average(np.asarray(x, dtype=float)), rankdata_average(np.asarray(y, dtype=float)))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "dataset"
    out_dir = root / "revision_analysis" / "remaining_low_cost_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    diversity_rows: list[dict] = []
    subset_rows: list[dict] = []
    subset_summary_rows: list[dict] = []
    dimension_rows: list[dict] = []
    dimension_summary_rows: list[dict] = []

    for pair_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.endswith("-new")):
        pair = pair_dir.name.replace("-new", "")
        for strategy in STRATEGIES:
            metric = load_metric(pair_dir, strategy)
            reviewer_scores = load_reviewer_scores(pair_dir, strategy, MODEL_CODES, MODEL_CODES, "score")
            meprs_selection_scores = reviewer_scores.sum(axis=0)
            meprs_selected, meprs_scores = select(metric, meprs_selection_scores)
            model_means = metric.mean(axis=1)
            best_i = int(np.argmax(model_means))
            best_scores = metric[best_i]
            gain = float((meprs_scores.mean() - best_scores.mean()) * 100)

            texts = [read_lines(pair_dir / f"{model}_{strategy}.txt") for model in MODEL_CODES]
            n_items = len(texts[0])
            pairwise_chrf = []
            pairwise_token_f1 = []
            pairwise_bleurt_range = []
            oracle_margin = []
            for sent in range(n_items):
                chrf_vals = []
                tok_vals = []
                for i, j in itertools.combinations(range(len(MODEL_CODES)), 2):
                    chrf_vals.append(chrf_like(texts[i][sent], texts[j][sent]))
                    tok_vals.append(token_f1(texts[i][sent], texts[j][sent]))
                pairwise_chrf.append(float(np.mean(chrf_vals)))
                pairwise_token_f1.append(float(np.mean(tok_vals)))
                pairwise_bleurt_range.append(float(metric[:, sent].max() - metric[:, sent].min()))
                sorted_metric = np.sort(metric[:, sent])[::-1]
                oracle_margin.append(float(sorted_metric[0] - sorted_metric[1]))

            diversity_rows.append(
                {
                    "pair": pair,
                    "strategy": strategy,
                    "mean_pairwise_chrf_similarity": float(np.mean(pairwise_chrf)),
                    "mean_pairwise_token_f1_similarity": float(np.mean(pairwise_token_f1)),
                    "mean_bleurt_candidate_range": float(np.mean(pairwise_bleurt_range) * 100),
                    "mean_oracle_margin": float(np.mean(oracle_margin) * 100),
                    "meprs_gain_vs_best_baseline": gain,
                    "best_baseline_model": MODEL_CODES[best_i],
                }
            )

            # Exhaustive predictor/reviewer subset analysis.
            for size_pred in range(1, len(MODEL_CODES) + 1):
                for pred_idx in itertools.combinations(range(len(MODEL_CODES)), size_pred):
                    pred_models = [MODEL_CODES[i] for i in pred_idx]
                    pred_metric = metric[list(pred_idx), :]
                    pred_model_means = pred_metric.mean(axis=1)
                    pred_best = float(pred_model_means.max() * 100)
                    pred_oracle = float(pred_metric.max(axis=0).mean() * 100)
                    for size_eval in range(1, len(MODEL_CODES) + 1):
                        for eval_idx in itertools.combinations(range(len(MODEL_CODES)), size_eval):
                            eval_models = [MODEL_CODES[i] for i in eval_idx]
                            sub_scores = reviewer_scores[np.ix_(eval_idx, pred_idx, range(metric.shape[1]))].sum(axis=0)
                            _, selected_scores = select(pred_metric, sub_scores)
                            subset_rows.append(
                                {
                                    "pair": pair,
                                    "strategy": strategy,
                                    "predictor_subset_size": size_pred,
                                    "reviewer_subset_size": size_eval,
                                    "predictor_models": ";".join(pred_models),
                                    "reviewer_models": ";".join(eval_models),
                                    "selected_bleurt": float(selected_scores.mean() * 100),
                                    "best_single_within_predictors": pred_best,
                                    "oracle_within_predictors": pred_oracle,
                                    "gain_vs_best_single_within_predictors": float(selected_scores.mean() * 100 - pred_best),
                                }
                            )

            # Single-dimension-only selection.
            for dim in DIMENSIONS:
                dim_scores = load_reviewer_scores(pair_dir, strategy, MODEL_CODES, MODEL_CODES, dim).sum(axis=0)
                dim_selected, dim_selected_scores = select(metric, dim_scores)
                dimension_rows.append(
                    {
                        "pair": pair,
                        "strategy": strategy,
                        "dimension": dim,
                        "selected_bleurt": float(dim_selected_scores.mean() * 100),
                        "meprs_bleurt": float(meprs_scores.mean() * 100),
                        "best_baseline_bleurt": float(model_means[best_i] * 100),
                        "dimension_minus_meprs": float((dim_selected_scores.mean() - meprs_scores.mean()) * 100),
                        "dimension_minus_best_baseline": float(dim_selected_scores.mean() * 100 - model_means[best_i] * 100),
                        "selection_agreement_with_meprs": float(np.mean(dim_selected == meprs_selected)),
                    }
                )

    for size_pred in range(1, len(MODEL_CODES) + 1):
        for size_eval in range(1, len(MODEL_CODES) + 1):
            rows = [
                r
                for r in subset_rows
                if r["predictor_subset_size"] == size_pred and r["reviewer_subset_size"] == size_eval
            ]
            if rows:
                vals = [r["selected_bleurt"] for r in rows]
                gains = [r["gain_vs_best_single_within_predictors"] for r in rows]
                subset_summary_rows.append(
                    {
                        "predictor_subset_size": size_pred,
                        "reviewer_subset_size": size_eval,
                        "num_subsets_across_conditions": len(rows),
                        "mean_selected_bleurt": float(np.mean(vals)),
                        "sd_selected_bleurt": float(np.std(vals, ddof=1)),
                        "mean_gain_vs_best_single_within_predictors": float(np.mean(gains)),
                        "positive_gain_rate": float(np.mean(np.asarray(gains) > 0)),
                    }
                )

    for dim in DIMENSIONS:
        rows = [r for r in dimension_rows if r["dimension"] == dim]
        dimension_summary_rows.append(
            {
                "dimension": dim,
                "mean_selected_bleurt": float(np.mean([r["selected_bleurt"] for r in rows])),
                "mean_minus_meprs": float(np.mean([r["dimension_minus_meprs"] for r in rows])),
                "mean_minus_best_baseline": float(np.mean([r["dimension_minus_best_baseline"] for r in rows])),
                "beats_meprs_conditions": int(sum(r["dimension_minus_meprs"] > 0 for r in rows)),
                "beats_best_baseline_conditions": int(sum(r["dimension_minus_best_baseline"] > 0 for r in rows)),
                "mean_selection_agreement_with_meprs": float(np.mean([r["selection_agreement_with_meprs"] for r in rows])),
            }
        )

    diversity_summary = [
        {
            "num_conditions": len(diversity_rows),
            "mean_pairwise_chrf_similarity": float(np.mean([r["mean_pairwise_chrf_similarity"] for r in diversity_rows])),
            "mean_pairwise_token_f1_similarity": float(
                np.mean([r["mean_pairwise_token_f1_similarity"] for r in diversity_rows])
            ),
            "mean_bleurt_candidate_range": float(np.mean([r["mean_bleurt_candidate_range"] for r in diversity_rows])),
            "pearson_chrf_similarity_vs_gain": pearson(
                [r["mean_pairwise_chrf_similarity"] for r in diversity_rows],
                [r["meprs_gain_vs_best_baseline"] for r in diversity_rows],
            ),
            "spearman_chrf_similarity_vs_gain": spearman(
                [r["mean_pairwise_chrf_similarity"] for r in diversity_rows],
                [r["meprs_gain_vs_best_baseline"] for r in diversity_rows],
            ),
            "pearson_bleurt_range_vs_gain": pearson(
                [r["mean_bleurt_candidate_range"] for r in diversity_rows],
                [r["meprs_gain_vs_best_baseline"] for r in diversity_rows],
            ),
            "spearman_bleurt_range_vs_gain": spearman(
                [r["mean_bleurt_candidate_range"] for r in diversity_rows],
                [r["meprs_gain_vs_best_baseline"] for r in diversity_rows],
            ),
        }
    ]

    write_csv(out_dir / "candidate_diversity_by_condition.csv", diversity_rows)
    write_csv(out_dir / "candidate_diversity_summary.csv", diversity_summary)
    write_csv(out_dir / "model_subset_exhaustive_details.csv", subset_rows)
    write_csv(out_dir / "model_subset_exhaustive_summary.csv", subset_summary_rows)
    write_csv(out_dir / "single_dimension_only_details.csv", dimension_rows)
    write_csv(out_dir / "single_dimension_only_summary.csv", dimension_summary_rows)

    print(f"Wrote outputs to {out_dir}")
    print("Diversity summary:", diversity_summary[0])
    print("Dimension summary:")
    for row in dimension_summary_rows:
        print(row)
    print("Subset summary rows:", len(subset_summary_rows), "detail rows:", len(subset_rows))


if __name__ == "__main__":
    main()
