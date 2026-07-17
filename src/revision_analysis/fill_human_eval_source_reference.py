from __future__ import annotations

import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    human_dir = root / "revision_analysis" / "human_eval_materials"
    sheet_path = human_dir / "human_eval_blind_sheet_en_he_tear_60.csv"
    key_path = human_dir / "human_eval_answer_key_en_he_tear_60.csv"
    align_path = root / "revision_analysis" / "outputs" / "fast_external_alignment_candidates.csv"

    sheet = read_csv(sheet_path)
    key = read_csv(key_path)
    align = read_csv(align_path)

    idx_by_item = {}
    for row in key:
        idx_by_item.setdefault(row["item_id"], row["sentence_index_0based"])

    align_by_idx = {}
    for row in align:
        if row["local_pair"] != "en-he-new":
            continue
        if float(row["best_similarity"]) < 0.92:
            continue
        align_by_idx[row["local_sentence_index_0based"]] = row

    filled = 0
    for row in sheet:
        idx = idx_by_item.get(row["item_id"])
        hit = align_by_idx.get(idx or "")
        if hit:
            row["source_text"] = hit["external_src"]
            row["reference_translation"] = hit["external_ref"]
            filled += 1

    out_path = human_dir / "human_eval_blind_sheet_en_he_tear_60_with_src_ref.csv"
    write_csv(out_path, sheet)

    summary = [
        {"item": "rows_total", "value": len(sheet)},
        {"item": "candidate_rows_with_src_ref", "value": filled},
        {"item": "unique_items_total", "value": len(set(row["item_id"] for row in sheet))},
        {"item": "unique_items_with_src_ref", "value": len(set(row["item_id"] for row in sheet if row["source_text"]))},
    ]
    write_csv(human_dir / "human_eval_src_ref_fill_summary.csv", summary)
    print(out_path)


if __name__ == "__main__":
    main()
