from __future__ import annotations

import csv
import random
from pathlib import Path


MODEL_CODES = ["G35", "G4o", "C3", "C35", "GP"]
MODEL_NAMES = {
    "G35": "GPT-3.5-Turbo",
    "G4o": "GPT-4o",
    "C3": "Claude-3-Opus",
    "C35": "Claude-3.5-Sonnet",
    "GP": "Gemini-Pro",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def read_scores(path: Path) -> list[float]:
    return [float(line.split()[0]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "dataset"
    out_dir = root / "revision_analysis" / "human_eval_materials"
    align_rows = read_csv(root / "revision_analysis" / "outputs" / "fast_external_alignment_candidates.csv")
    pair = "en-he-new"
    strategy = "tear"
    seed = 20260624
    n_items = 60
    rng = random.Random(seed)

    usable = [
        row
        for row in align_rows
        if row["local_pair"] == pair
        and float(row["best_similarity"]) >= 0.92
        and row["external_src"]
        and row["external_ref"]
    ]
    usable = sorted(usable, key=lambda r: int(r["local_sentence_index_0based"]))
    if len(usable) < n_items:
        selected = usable
    else:
        selected = sorted(rng.sample(usable, n_items), key=lambda r: int(r["local_sentence_index_0based"]))

    pair_dir = dataset / pair
    candidates = {model: read_lines(pair_dir / f"{model}_{strategy}.txt") for model in MODEL_CODES}
    bleurt = {model: read_scores(pair_dir / f"{model}_{strategy}.bleurt") for model in MODEL_CODES}
    reviewer_scores = {
        model: [
            sum(read_scores(pair_dir / f"{model}_{strategy}_{reviewer}.score")[i] for reviewer in MODEL_CODES)
            for i in range(len(candidates[model]))
        ]
        for model in MODEL_CODES
    }
    best_baseline_model = max(MODEL_CODES, key=lambda m: sum(bleurt[m]) / len(bleurt[m]))

    sheet_rows = []
    key_rows = []
    for item_no, align in enumerate(selected, 1):
        sent_id = int(align["local_sentence_index_0based"])
        meprs_model = max(MODEL_CODES, key=lambda m: reviewer_scores[m][sent_id])
        options = [
            ("A", "MEPRS", meprs_model, candidates[meprs_model][sent_id]),
            ("B", "BestSingleBaseline", best_baseline_model, candidates[best_baseline_model][sent_id]),
        ]
        rng.shuffle(options)
        for label, role, model, text in options:
            sheet_rows.append(
                {
                    "item_id": item_no,
                    "language_pair": pair.replace("-new", ""),
                    "strategy": strategy,
                    "source_text": align["external_src"],
                    "reference_translation": align["external_ref"],
                    "candidate_label": label,
                    "candidate_translation": text,
                    "adequacy_1_5": "",
                    "fluency_1_5": "",
                    "overall_preference_A_B_Tie": "",
                    "comments": "",
                }
            )
            key_rows.append(
                {
                    "item_id": item_no,
                    "candidate_label": label,
                    "system_role": role,
                    "model_code": model,
                    "model_name": MODEL_NAMES[model],
                    "sentence_index_0based": sent_id,
                    "alignment_similarity": align["best_similarity"],
                    "sample_id": align["external_sample_id"],
                    "bleurt": bleurt[model][sent_id],
                }
            )

    write_csv(out_dir / "human_eval_blind_sheet_en_he_tear_aligned_60.csv", sheet_rows)
    write_csv(out_dir / "human_eval_answer_key_en_he_tear_aligned_60.csv", key_rows)
    write_csv(
        out_dir / "human_eval_aligned_60_summary.csv",
        [
            {"item": "usable_aligned_items", "value": len(usable)},
            {"item": "selected_items", "value": len(selected)},
            {"item": "candidate_rows", "value": len(sheet_rows)},
        ],
    )
    print(out_dir / "human_eval_blind_sheet_en_he_tear_aligned_60.csv")


if __name__ == "__main__":
    main()
