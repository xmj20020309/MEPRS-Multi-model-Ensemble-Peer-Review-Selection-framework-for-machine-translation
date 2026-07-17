"""Run revision-oriented analyses on existing MEPRS outputs.

This script does not call any LLM APIs. It uses the released candidate-level
BLEURT scores and LLM reviewer scores to compute the low-cost analyses requested
by reviewers: MEPRS-NoSelf, paired bootstrap significance tests, top-k selection,
inter-reviewer agreement, self-preference diagnostics, and aggregation variants.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def read_scores(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(float(line.split()[0]))
    return np.asarray(values, dtype=float)


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


def select_by_scores(candidate_metric: np.ndarray, selection_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.argmax(selection_scores, axis=0)
    scores = candidate_metric[selected, np.arange(candidate_metric.shape[1])]
    return selected, scores


def kendalls_w(scores_by_reviewer_candidate: np.ndarray) -> float:
    ranks = np.vstack([rankdata_average(row) for row in scores_by_reviewer_candidate])
    m, n = ranks.shape
    expected = m * (n + 1) / 2.0
    s = float(((ranks.sum(axis=0) - expected) ** 2).sum())
    denom = m * m * (n**3 - n)
    return 0.0 if denom == 0 else 12.0 * s / denom


def paired_bootstrap(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    p_two = min(1.0, 2.0 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return p_two, float(ci_low), float(ci_high)


def holm_adjust(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        adjusted[idx] = running
    return adjusted


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_pair_strategy(data_dir: Path, pair: str, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    pair_dir = data_dir / pair
    metric = np.vstack([read_scores(pair_dir / f"{model}_{strategy}.bleurt") for model in MODEL_CODES])
    reviewer_scores = np.zeros((len(MODEL_CODES), len(MODEL_CODES), metric.shape[1]), dtype=float)
    for reviewer_i, reviewer in enumerate(MODEL_CODES):
        for candidate_i, candidate in enumerate(MODEL_CODES):
            reviewer_scores[reviewer_i, candidate_i, :] = read_scores(
                pair_dir / f"{candidate}_{strategy}_{reviewer}.score"
            )
    return metric, reviewer_scores


def aggregate_variants(reviewer_scores: np.ndarray) -> dict[str, np.ndarray]:
    mean_scores = reviewer_scores.sum(axis=0)
    median_scores = np.median(reviewer_scores, axis=0)
    z_scores = np.zeros_like(mean_scores)
    rank_scores = np.zeros_like(mean_scores)
    for sent in range(reviewer_scores.shape[2]):
        for reviewer_i in range(reviewer_scores.shape[0]):
            vals = reviewer_scores[reviewer_i, :, sent]
            std = vals.std()
            if std > 0:
                z_scores[:, sent] += (vals - vals.mean()) / std
            rank_scores[:, sent] += rankdata_average(vals)
    return {
        "mean": mean_scores,
        "median": median_scores,
        "zscore": z_scores,
        "rank": rank_scores,
    }


def run(data_dir: Path, output_dir: Path, n_boot: int, seed: int) -> None:
    pairs = sorted(p.name for p in data_dir.iterdir() if p.is_dir() and p.name.endswith("-new"))
    inventory_rows = []
    summary_rows = []
    self_bias_rows = []
    dimension_rows = []
    missing_rows = [
        {
            "item": "source sentences",
            "status": "not present in current dataset directory",
            "why_needed": "COMET/CometKIWI inputs, exact prompt audit, human evaluation sheets",
        },
        {
            "item": "reference translations",
            "status": "not present in current dataset directory",
            "why_needed": "COMET-22, chrF++, BLEU, MBR-COMET, MBR-BLEURT, MetricX",
        },
        {
            "item": "WMT sample IDs",
            "status": "not present in current dataset directory",
            "why_needed": "rebuilding exact 200-sentence subsets from WMT22/WMT23",
        },
        {
            "item": "API token/latency logs",
            "status": "not present in current dataset directory",
            "why_needed": "actual USD and latency cost table",
        },
        {
            "item": "raw prompts and raw LLM responses",
            "status": "not present in current dataset directory",
            "why_needed": "auditability of parsed reviewer scores and zero-score cases",
        },
    ]

    for pair in pairs:
        pair_dir = data_dir / pair
        for strategy in STRATEGIES:
            files = list(pair_dir.glob(f"*_{strategy}*"))
            inventory_rows.append(
                {
                    "pair": pair,
                    "strategy": strategy,
                    "candidate_txt": len(list(pair_dir.glob(f"*_{strategy}.txt"))),
                    "candidate_bleurt": len(list(pair_dir.glob(f"*_{strategy}.bleurt"))),
                    "iterative_txt": len(list(pair_dir.glob(f"*_{strategy}_[0-4].txt"))),
                    "iterative_bleurt": len(list(pair_dir.glob(f"*_{strategy}_[0-4].bleurt"))),
                    "review_score_files": len(list(pair_dir.glob(f"*_{strategy}_*.score"))),
                    "dimension_files": sum(len(list(pair_dir.glob(f"*_{strategy}_*.{dim}"))) for dim in DIMENSIONS),
                    "total_strategy_files": len(files),
                }
            )

            metric, reviewer_scores = load_pair_strategy(data_dir, pair, strategy)
            model_means = metric.mean(axis=1)
            best_i = int(np.argmax(model_means))
            best_scores = metric[best_i]
            variants = aggregate_variants(reviewer_scores)
            meprs_selected, meprs_scores = select_by_scores(metric, variants["mean"])

            no_self_scores_for_selection = np.zeros_like(variants["mean"])
            for candidate_i in range(len(MODEL_CODES)):
                keep_reviewers = [i for i in range(len(MODEL_CODES)) if i != candidate_i]
                no_self_scores_for_selection[candidate_i, :] = reviewer_scores[
                    keep_reviewers, candidate_i, :
                ].sum(axis=0)
            no_self_selected, no_self_scores = select_by_scores(metric, no_self_scores_for_selection)

            agg_scores = {}
            for name, selection_scores in variants.items():
                _, selected_scores = select_by_scores(metric, selection_scores)
                agg_scores[name] = selected_scores.mean() * 100

            diff = meprs_scores - best_scores
            p_boot, ci_low, ci_high = paired_bootstrap(diff, n_boot=n_boot, seed=seed)

            ranks = []
            for sent in range(metric.shape[1]):
                selected_metric = metric[meprs_selected[sent], sent]
                ranks.append(1 + int(np.sum(metric[:, sent] > selected_metric)))
            ranks = np.asarray(ranks)

            w_values = np.asarray([kendalls_w(reviewer_scores[:, :, sent]) for sent in range(metric.shape[1])])

            row = {
                "pair": pair.replace("-new", ""),
                "strategy": strategy,
                "best_baseline_model": MODEL_CODES[best_i],
                "best_baseline_model_name": MODEL_NAMES[MODEL_CODES[best_i]],
                "best_baseline_bleurt": model_means[best_i] * 100,
                "meprs_bleurt": meprs_scores.mean() * 100,
                "meprs_gain": diff.mean() * 100,
                "bootstrap_p": p_boot,
                "ci95_low_gain": ci_low * 100,
                "ci95_high_gain": ci_high * 100,
                "noself_bleurt": no_self_scores.mean() * 100,
                "noself_minus_meprs": (no_self_scores.mean() - meprs_scores.mean()) * 100,
                "noself_changed_selection_rate": float(np.mean(no_self_selected != meprs_selected)),
                "median_agg_bleurt": agg_scores["median"],
                "zscore_agg_bleurt": agg_scores["zscore"],
                "rank_agg_bleurt": agg_scores["rank"],
                "oracle_bleurt": metric.max(axis=0).mean() * 100,
                "random_expected_bleurt": metric.mean(axis=0).mean() * 100,
                "top1_rate": float(np.mean(ranks <= 1)),
                "top2_rate": float(np.mean(ranks <= 2)),
                "top3_rate": float(np.mean(ranks <= 3)),
                "kendall_w_mean": float(w_values.mean()),
                "kendall_w_sd": float(w_values.std(ddof=1)),
            }
            for i, model in enumerate(MODEL_CODES):
                row[f"{model}_bleurt"] = model_means[i] * 100
            summary_rows.append(row)

            for reviewer_i, reviewer in enumerate(MODEL_CODES):
                own = reviewer_scores[reviewer_i, reviewer_i, :]
                other = np.delete(reviewer_scores[reviewer_i, :, :], reviewer_i, axis=0)
                strict_top = []
                tied_top = []
                for sent in range(metric.shape[1]):
                    vals = reviewer_scores[reviewer_i, :, sent]
                    strict_top.append(vals[reviewer_i] > np.max(np.delete(vals, reviewer_i)))
                    tied_top.append(vals[reviewer_i] == np.max(vals))
                self_bias_rows.append(
                    {
                        "pair": pair.replace("-new", ""),
                        "strategy": strategy,
                        "reviewer": reviewer,
                        "reviewer_name": MODEL_NAMES[reviewer],
                        "own_mean_score": float(own.mean()),
                        "other_mean_score": float(other.mean()),
                        "own_minus_other": float(own.mean() - other.mean()),
                        "strict_self_top_rate": float(np.mean(strict_top)),
                        "self_top_with_ties_rate": float(np.mean(tied_top)),
                    }
                )

            for dim in DIMENSIONS + ["score"]:
                dim_scores = np.zeros_like(variants["mean"])
                for reviewer_i, reviewer in enumerate(MODEL_CODES):
                    for candidate_i, candidate in enumerate(MODEL_CODES):
                        suffix = "score" if dim == "score" else dim
                        dim_scores[candidate_i, :] += read_scores(pair_dir / f"{candidate}_{strategy}_{reviewer}.{suffix}")
                _, selected_scores = select_by_scores(metric, dim_scores)
                dimension_rows.append(
                    {
                        "pair": pair.replace("-new", ""),
                        "strategy": strategy,
                        "selection_dimension": dim,
                        "selected_bleurt": selected_scores.mean() * 100,
                    }
                )

    adjusted = holm_adjust([float(row["bootstrap_p"]) for row in summary_rows])
    for row, adj in zip(summary_rows, adjusted):
        row["holm_adjusted_p"] = adj

    write_csv(output_dir / "data_inventory_summary.csv", inventory_rows)
    write_csv(output_dir / "bleurt_revision_analysis.csv", summary_rows)
    write_csv(output_dir / "self_bias_summary.csv", self_bias_rows)
    write_csv(output_dir / "dimension_selection_summary.csv", dimension_rows)
    write_csv(output_dir / "missing_items_for_additional_metrics.csv", missing_rows)

    metadata = {
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "models": MODEL_NAMES,
        "strategies": STRATEGIES,
        "bootstrap_iterations": n_boot,
        "random_seed": seed,
        "note": "Scores are BLEURT-based and use existing released candidate and reviewer score files only.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MEPRS revision analyses on existing released data.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output_dir", type=Path, default=Path("revision_analysis") / "outputs")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
