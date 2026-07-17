from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU, CHRF


MODEL_CODES = ["G35", "G4o", "C3", "C35", "GP"]
STRATEGIES = ["it", "tear"]


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


def selected_model_by_meprs(pair_dir: Path, strategy: str, sent_idx: int) -> str:
    totals = {}
    for model in MODEL_CODES:
        total = 0.0
        for reviewer in MODEL_CODES:
            scores = read_scores(pair_dir / f"{model}_{strategy}_{reviewer}.score")
            total += float(scores[sent_idx])
        totals[model] = total
    return max(totals, key=totals.get)


def run(root: Path, min_similarity: float = 0.92) -> None:
    dataset_dir = root / "dataset"
    align_path = root / "revision_analysis" / "outputs" / "fast_external_alignment_candidates.csv"
    out_dir = root / "revision_analysis" / "sacrebleu_overlap_outputs"
    align_rows = read_csv(align_path)
    align_by_pair = defaultdict(dict)
    for row in align_rows:
        if float(row["best_similarity"]) >= min_similarity and row["external_ref"]:
            align_by_pair[row["local_pair"]][int(row["local_sentence_index_0based"])] = row

    bleu = BLEU(effective_order=True)
    chrfpp = CHRF(word_order=2)
    summary = []
    for pair_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        pair = pair_dir.name
        if pair not in align_by_pair:
            continue
        for strategy in STRATEGIES:
            metric_by_model = {model: read_scores(pair_dir / f"{model}_{strategy}.bleurt") for model in MODEL_CODES}
            candidate_by_model = {model: read_lines(pair_dir / f"{model}_{strategy}.txt") for model in MODEL_CODES}
            best_baseline = max(MODEL_CODES, key=lambda m: float(metric_by_model[m].mean()))
            systems = [*MODEL_CODES, "MEPRS"]
            hyps = {system: [] for system in systems}
            refs = []
            for idx, align in sorted(align_by_pair[pair].items()):
                refs.append(align["external_ref"])
                meprs_model = selected_model_by_meprs(pair_dir, strategy, idx)
                for system in systems:
                    model = meprs_model if system == "MEPRS" else system
                    hyps[system].append(candidate_by_model[model][idx])
            for system in systems:
                if not refs:
                    continue
                summary.append(
                    {
                        "pair": pair.replace("-new", ""),
                        "strategy": strategy,
                        "system": system,
                        "is_best_bleurt_baseline": system == best_baseline,
                        "aligned_sentence_count": len(refs),
                        "sacrebleu": bleu.corpus_score(hyps[system], [refs]).score,
                        "chrf_pp": chrfpp.corpus_score(hyps[system], [refs]).score,
                    }
                )
    write_csv(out_dir / "aligned_sacrebleu_chrfpp_summary.csv", summary)


def main() -> None:
    run(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
