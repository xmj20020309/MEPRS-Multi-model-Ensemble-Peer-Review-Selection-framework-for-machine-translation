from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU, CHRF


MODEL_CODES = ["G35", "G4o", "C3", "C35", "GP"]
MODEL_NAMES = {
    "G35": "GPT-3.5-Turbo",
    "G4o": "GPT-4o",
    "C3": "Claude-3-Opus",
    "C35": "Claude-3.5-Sonnet",
    "GP": "Gemini-Pro",
}
STRATEGIES = ["it", "tear"]


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def read_scores(path: Path) -> np.ndarray:
    return np.asarray([float(line.split()[0]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def meprs_selection(pair_dir: Path, strategy: str, n_sent: int) -> np.ndarray:
    selected = []
    for idx in range(n_sent):
        totals = []
        for model in MODEL_CODES:
            total = 0.0
            for reviewer in MODEL_CODES:
                total += float(read_scores(pair_dir / f"{model}_{strategy}_{reviewer}.score")[idx])
            totals.append(total)
        selected.append(int(np.argmax(totals)))
    return np.asarray(selected, dtype=int)


def mbr_selection(candidates: dict[str, list[str]], metric_name: str) -> np.ndarray:
    metric = BLEU(effective_order=True) if metric_name == "sacrebleu" else CHRF(word_order=2)
    n_sent = len(next(iter(candidates.values())))
    selected = []
    for idx in range(n_sent):
        utilities = []
        for model in MODEL_CODES:
            hyp = candidates[model][idx]
            refs = [candidates[other][idx] for other in MODEL_CODES if other != model]
            scores = [metric.sentence_score(hyp, [ref]).score for ref in refs]
            utilities.append(float(np.mean(scores)))
        selected.append(int(np.argmax(utilities)))
    return np.asarray(selected, dtype=int)


def run(root: Path) -> None:
    dataset_dir = root / "dataset"
    out_dir = root / "revision_analysis" / "surface_mbr_outputs"
    rows = []
    detail_rows = []
    for pair_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        pair = pair_dir.name
        for strategy in STRATEGIES:
            candidates = {model: read_lines(pair_dir / f"{model}_{strategy}.txt") for model in MODEL_CODES}
            bleurt = np.vstack([read_scores(pair_dir / f"{model}_{strategy}.bleurt") for model in MODEL_CODES])
            n_sent = bleurt.shape[1]
            model_means = bleurt.mean(axis=1)
            best_idx = int(np.argmax(model_means))
            meprs_selected = meprs_selection(pair_dir, strategy, n_sent)
            meprs_bleurt = float(bleurt[meprs_selected, np.arange(n_sent)].mean() * 100.0)
            oracle_bleurt = float(bleurt.max(axis=0).mean() * 100.0)
            random_bleurt = float(bleurt.mean(axis=0).mean() * 100.0)
            for metric_name in ["sacrebleu", "chrf_pp"]:
                selected = mbr_selection(candidates, metric_name)
                selected_bleurt = float(bleurt[selected, np.arange(n_sent)].mean() * 100.0)
                rows.append(
                    {
                        "pair": pair.replace("-new", ""),
                        "strategy": strategy,
                        "mbr_utility": metric_name,
                        "selected_bleurt": selected_bleurt,
                        "meprs_bleurt": meprs_bleurt,
                        "best_single_bleurt": float(model_means[best_idx] * 100.0),
                        "best_single_model": MODEL_CODES[best_idx],
                        "oracle_bleurt": oracle_bleurt,
                        "random_expected_bleurt": random_bleurt,
                        "mbr_minus_meprs": selected_bleurt - meprs_bleurt,
                        "mbr_minus_best_single": selected_bleurt - float(model_means[best_idx] * 100.0),
                        "selection_agreement_with_meprs": float(np.mean(selected == meprs_selected)),
                    }
                )
                for idx, sel in enumerate(selected):
                    detail_rows.append(
                        {
                            "pair": pair.replace("-new", ""),
                            "strategy": strategy,
                            "mbr_utility": metric_name,
                            "sentence_index_0based": idx,
                            "selected_model": MODEL_CODES[int(sel)],
                            "selected_model_name": MODEL_NAMES[MODEL_CODES[int(sel)]],
                            "selected_bleurt": float(bleurt[int(sel), idx] * 100.0),
                            "meprs_model": MODEL_CODES[int(meprs_selected[idx])],
                            "meprs_bleurt": float(bleurt[int(meprs_selected[idx]), idx] * 100.0),
                        }
                    )
    write_csv(out_dir / "surface_mbr_bleurt_summary.csv", rows)
    write_csv(out_dir / "surface_mbr_sentence_details.csv", detail_rows)


def main() -> None:
    run(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
