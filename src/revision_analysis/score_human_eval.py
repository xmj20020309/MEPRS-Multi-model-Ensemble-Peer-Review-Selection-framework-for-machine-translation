from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


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


def parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_pref(value: str) -> str:
    value = (value or "").strip().upper()
    if value in {"A", "B", "TIE", "T"}:
        return "Tie" if value in {"TIE", "T"} else value
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Score blind human-evaluation sheets against an answer key.")
    parser.add_argument("--sheet", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    sheet_rows = read_csv(args.sheet)
    key_rows = read_csv(args.key)
    key_by_item_label = {(row["item_id"], row["candidate_label"]): row for row in key_rows}

    joined = []
    for row in sheet_rows:
        key = key_by_item_label.get((row["item_id"], row["candidate_label"]), {})
        joined.append({**row, **{f"key_{k}": v for k, v in key.items()}})

    item_rows = []
    by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in joined:
        by_item[row["item_id"]].append(row)

    role_pref_counts: Counter[str] = Counter()
    role_score_sums: dict[str, Counter[str]] = defaultdict(Counter)
    role_score_counts: dict[str, Counter[str]] = defaultdict(Counter)
    error_counts: Counter[str] = Counter()
    completed_items = 0

    for item_id, rows in sorted(by_item.items(), key=lambda x: int(x[0])):
        pref = normalize_pref(rows[0].get("overall_preference_A_B_Tie", ""))
        label_to_role = {row["candidate_label"]: row.get("key_system_role", "") for row in rows}
        preferred_role = "Unfilled"
        if pref in {"A", "B"}:
            preferred_role = label_to_role.get(pref, "Unknown")
        elif pref == "Tie":
            preferred_role = "Tie"
        if preferred_role != "Unfilled":
            completed_items += 1
        role_pref_counts[preferred_role] += 1

        for row in rows:
            role = row.get("key_system_role", "Unknown")
            for field in ["adequacy_1_5", "fluency_1_5"]:
                score = parse_float(row.get(field, ""))
                if score is not None:
                    role_score_sums[role][field] += score
                    role_score_counts[role][field] += 1
            if row.get("error_type", "").strip():
                error_counts[row["error_type"].strip()] += 1

        item_rows.append(
            {
                "item_id": item_id,
                "preference": pref,
                "preferred_role": preferred_role,
                "roles": ";".join(f"{row['candidate_label']}={row.get('key_system_role','')}" for row in rows),
            }
        )

    summary_rows: list[dict[str, object]] = [
        {"metric": "candidate_rows", "value": len(sheet_rows)},
        {"metric": "items", "value": len(by_item)},
        {"metric": "items_with_preference", "value": completed_items},
    ]
    for role, count in sorted(role_pref_counts.items()):
        summary_rows.append({"metric": f"preference_count_{role}", "value": count})
    for role in sorted(role_score_sums):
        for field in ["adequacy_1_5", "fluency_1_5"]:
            count = role_score_counts[role][field]
            mean = role_score_sums[role][field] / count if count else ""
            summary_rows.append({"metric": f"mean_{field}_{role}", "value": mean})
    for error_type, count in sorted(error_counts.items()):
        summary_rows.append({"metric": f"error_type_{error_type}", "value": count})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "human_eval_joined_rows.csv", joined)
    write_csv(args.out_dir / "human_eval_item_preferences.csv", item_rows)
    write_csv(args.out_dir / "human_eval_summary.csv", summary_rows)
    print(args.out_dir / "human_eval_summary.csv")


if __name__ == "__main__":
    main()
