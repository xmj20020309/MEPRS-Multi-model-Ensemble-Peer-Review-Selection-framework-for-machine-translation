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


def read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def read_scores(path: Path) -> list[float]:
    return [float(line.split()[0]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "dataset"
    out_dir = root / "revision_analysis" / "human_eval_materials"
    pair = "en-he-new"
    strategy = "tear"
    n_items = 60
    seed = 20260623
    rng = random.Random(seed)

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

    n_sent = len(next(iter(candidates.values())))
    selected_ids = []
    for i in range(n_sent):
        best_baseline_model = max(MODEL_CODES, key=lambda m: sum(bleurt[m]) / len(bleurt[m]))
        meprs_model = max(MODEL_CODES, key=lambda m: reviewer_scores[m][i])
        if meprs_model != best_baseline_model:
            selected_ids.append(i)
    if len(selected_ids) < n_items:
        selected_ids = list(range(n_sent))
    selected_ids = sorted(rng.sample(selected_ids, min(n_items, len(selected_ids))))

    sheet_rows = []
    key_rows = []
    for item_no, sent_id in enumerate(selected_ids, 1):
        best_baseline_model = max(MODEL_CODES, key=lambda m: sum(bleurt[m]) / len(bleurt[m]))
        meprs_model = max(MODEL_CODES, key=lambda m: reviewer_scores[m][sent_id])
        options = [
            ("A", "MEPRS", meprs_model, candidates[meprs_model][sent_id]),
            ("B", "BestSingleBaseline", best_baseline_model, candidates[best_baseline_model][sent_id]),
        ]
        rng.shuffle(options)
        for label, system_role, model, text in options:
            sheet_rows.append(
                {
                    "item_id": item_no,
                    "language_pair": pair.replace("-new", ""),
                    "strategy": strategy,
                    "source_text": "",
                    "reference_translation": "",
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
                    "system_role": system_role,
                    "model_code": model,
                    "model_name": MODEL_NAMES[model],
                    "sentence_index_0based": sent_id,
                    "bleurt": bleurt[model][sent_id],
                }
            )

    write_csv(out_dir / "human_eval_blind_sheet_en_he_tear_60.csv", sheet_rows)
    write_csv(out_dir / "human_eval_answer_key_en_he_tear_60.csv", key_rows)
    (out_dir / "README_zh.md").write_text(
        "\n".join(
            [
                "# 人工评价材料说明",
                "",
                "- `human_eval_blind_sheet_en_he_tear_60.csv` 是给标注者使用的盲评表。",
                "- `human_eval_answer_key_en_he_tear_60.csv` 是作者内部使用的答案键，不应发给标注者。",
                "- 当前表格已填入候选译文，但 `source_text` 和 `reference_translation` 为空，因为当前 GitHub zip 中缺原始 WMT source/reference/sample ID。",
                "- 建议优先补齐 source/reference 后再发给标注者；若时间特别紧，可以做 target-side fluency + preference 的临时评价，但说服力低于完整 source-conditioned adequacy 评价。",
                "- 抽样策略：优先抽 MEPRS 与最佳单模型不同的 EN->HE TEaR 句子，目标 60 个 item。",
            ]
        ),
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
