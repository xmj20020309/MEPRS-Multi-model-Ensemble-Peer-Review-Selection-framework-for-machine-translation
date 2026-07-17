from __future__ import annotations

import csv
import random
import re
from pathlib import Path


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


def read_scores(path: Path) -> list[float]:
    return [float(line.split()[0]) for line in read_lines(path) if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def build_alignment_map(root: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows = read_csv(root / "revision_analysis" / "outputs" / "fast_external_alignment_candidates.csv")
    best: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        try:
            sim = float(row.get("best_similarity", "0") or 0)
            sent = int(row["local_sentence_index_0based"])
        except (KeyError, ValueError):
            continue
        if sim < 0.92:
            continue
        key = (row["local_pair"], sent)
        old = best.get(key)
        if old is None or sim > float(old.get("best_similarity", "0") or 0):
            best[key] = row
    return best


def reviewer_sum(pair_dir: Path, model: str, strategy: str, sent: int) -> float:
    total = 0.0
    for reviewer in MODEL_CODES:
        scores = read_scores(pair_dir / f"{model}_{strategy}_{reviewer}.score")
        total += scores[sent]
    return total


def collect_qualitative_cases(root: Path, out_dir: Path) -> None:
    dataset = root / "dataset"
    align_map = build_alignment_map(root)
    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for pair_dir in sorted(p for p in dataset.iterdir() if p.is_dir() and p.name.endswith("-new")):
        pair = pair_dir.name
        for strategy in STRATEGIES:
            candidates = {model: read_lines(pair_dir / f"{model}_{strategy}.txt") for model in MODEL_CODES}
            bleurt = {model: read_scores(pair_dir / f"{model}_{strategy}.bleurt") for model in MODEL_CODES}
            n_items = len(next(iter(candidates.values())))
            model_means = {model: sum(bleurt[model]) / len(bleurt[model]) for model in MODEL_CODES}
            best_single_model = max(MODEL_CODES, key=lambda model: model_means[model])
            condition_rows: list[dict[str, object]] = []

            for sent in range(n_items):
                meprs_model = max(MODEL_CODES, key=lambda model: reviewer_sum(pair_dir, model, strategy, sent))
                bleurt_best_model = max(MODEL_CODES, key=lambda model: bleurt[model][sent])
                if meprs_model == bleurt_best_model:
                    continue
                meprs_text = candidates[meprs_model][sent]
                best_text = candidates[bleurt_best_model][sent]
                if normalize_text(meprs_text) == normalize_text(best_text):
                    continue
                align = align_map.get((pair, sent), {})
                gap = (bleurt[bleurt_best_model][sent] - bleurt[meprs_model][sent]) * 100
                best_single_gap = (bleurt[best_single_model][sent] - bleurt[meprs_model][sent]) * 100
                row = {
                    "pair": pair.replace("-new", ""),
                    "strategy": strategy,
                    "sentence_index_0based": sent,
                    "has_public_source_reference": bool(align),
                    "alignment_similarity": align.get("best_similarity", ""),
                    "sample_id": align.get("external_sample_id", ""),
                    "source_text": align.get("external_src", ""),
                    "reference_translation": align.get("external_ref", ""),
                    "meprs_model": meprs_model,
                    "meprs_model_name": MODEL_NAMES[meprs_model],
                    "bleurt_best_model": bleurt_best_model,
                    "bleurt_best_model_name": MODEL_NAMES[bleurt_best_model],
                    "best_single_model_for_condition": best_single_model,
                    "best_single_model_name": MODEL_NAMES[best_single_model],
                    "meprs_bleurt": bleurt[meprs_model][sent] * 100,
                    "bleurt_best_score": bleurt[bleurt_best_model][sent] * 100,
                    "gap_to_bleurt_best": gap,
                    "gap_to_best_single_model_at_sentence": best_single_gap,
                    "meprs_reviewer_sum": reviewer_sum(pair_dir, meprs_model, strategy, sent),
                    "bleurt_best_reviewer_sum": reviewer_sum(pair_dir, bleurt_best_model, strategy, sent),
                    "meprs_translation": meprs_text,
                    "bleurt_best_translation": best_text,
                    "reviewer_note_zh": "",
                }
                all_rows.append(row)
                condition_rows.append(row)

            if condition_rows:
                gaps = [float(row["gap_to_bleurt_best"]) for row in condition_rows]
                summary_rows.append(
                    {
                        "pair": pair.replace("-new", ""),
                        "strategy": strategy,
                        "num_meprs_not_bleurt_best_text_different": len(condition_rows),
                        "num_with_public_source_reference": sum(
                            1 for row in condition_rows if row["has_public_source_reference"]
                        ),
                        "mean_gap_to_bleurt_best": sum(gaps) / len(gaps),
                        "max_gap_to_bleurt_best": max(gaps),
                    }
                )
            else:
                summary_rows.append(
                    {
                        "pair": pair.replace("-new", ""),
                        "strategy": strategy,
                        "num_meprs_not_bleurt_best_text_different": 0,
                        "num_with_public_source_reference": 0,
                        "mean_gap_to_bleurt_best": 0.0,
                        "max_gap_to_bleurt_best": 0.0,
                    }
                )

    all_rows = sorted(all_rows, key=lambda row: float(row["gap_to_bleurt_best"]), reverse=True)
    write_csv(out_dir / "qualitative_error_cases_all.csv", all_rows)
    write_csv(out_dir / "qualitative_error_cases_summary.csv", summary_rows)

    aligned_rows = [row for row in all_rows if row["has_public_source_reference"]]
    write_csv(out_dir / "qualitative_error_cases_top_aligned.csv", aligned_rows[:80])

    rng = random.Random(20260624)
    sheet_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for item_id, row in enumerate(aligned_rows[:60], 1):
        options = [
            ("MEPRS", row["meprs_model"], row["meprs_model_name"], row["meprs_bleurt"], row["meprs_translation"]),
            (
                "BLEURTBest",
                row["bleurt_best_model"],
                row["bleurt_best_model_name"],
                row["bleurt_best_score"],
                row["bleurt_best_translation"],
            ),
        ]
        rng.shuffle(options)
        for label, (role, model_code, model_name, bleurt_score, translation) in zip(["A", "B"], options):
            sheet_rows.append(
                {
                    "item_id": item_id,
                    "pair": row["pair"],
                    "strategy": row["strategy"],
                    "source_text": row["source_text"],
                    "reference_translation": row["reference_translation"],
                    "candidate_label": label,
                    "candidate_translation": translation,
                    "adequacy_1_5": "",
                    "fluency_1_5": "",
                    "overall_preference_A_B_Tie": "",
                    "error_type": "",
                    "comments": "",
                }
            )
            key_rows.append(
                {
                    "item_id": item_id,
                    "candidate_label": label,
                    "system_role": role,
                    "model_code": model_code,
                    "model_name": model_name,
                    "bleurt": bleurt_score,
                    "sentence_index_0based": row["sentence_index_0based"],
                    "sample_id": row["sample_id"],
                    "gap_to_bleurt_best": row["gap_to_bleurt_best"],
                }
            )

    if sheet_rows:
        write_csv(out_dir / "qualitative_error_case_blind_sheet.csv", sheet_rows)
        write_csv(out_dir / "qualitative_error_case_answer_key.csv", key_rows)


def collect_gap_evidence(root: Path, out_dir: Path) -> None:
    dataset = root / "dataset"
    revision = root / "revision_analysis"
    source_like = list(dataset.rglob("*source*")) + list(dataset.rglob("*src*"))
    ref_like = list(dataset.rglob("*reference*")) + list(dataset.rglob("*ref*"))
    prompt_like = [p for p in root.rglob("*prompt*") if p.is_file()]
    token_like = [p for p in root.rglob("*token*") if p.is_file()]
    latency_like = [p for p in root.rglob("*latency*") if p.is_file()]
    api_like = [p for p in root.rglob("*api*") if p.is_file()]
    response_like = [
        p
        for p in root.rglob("*response*")
        if p.is_file() and p.suffix.lower() not in {".md"}
    ]
    coverage_rows = read_csv(revision / "outputs" / "fast_external_alignment_coverage_summary.csv")
    human_sheet = read_csv(revision / "human_eval_materials" / "human_eval_blind_sheet_en_he_tear_aligned_60.csv")
    label_fields = ["adequacy_1_5", "fluency_1_5", "overall_preference_A_B_Tie"]
    label_cells = [row.get(field, "") for row in human_sheet for field in label_fields]
    filled_label_cells = sum(1 for value in label_cells if value.strip())

    google_dirs = [
        revision / "google_mqm_charcnn_outputs_extended",
        revision / "google_mqm_charcnn_outputs_hyp_only_extended",
        revision / "google_mqm_charcnn_outputs_ref_hyp_extended",
        revision / "google_mqm_charcnn_outputs_wide_full",
        revision / "google_mqm_charcnn_outputs_more_seeds_full",
    ]
    checkpoint_count = sum(len(list((path / "checkpoints").glob("*.pt"))) for path in google_dirs if path.exists())

    rows = [
        {
            "item": "full-set source sentences",
            "status": "missing for the full 16 x 200 MEPRS test conditions",
            "local_evidence": f"dataset source/src-like files: {len(source_like)}; public-overlap source/reference coverage rows: {len(coverage_rows)}",
            "action_needed": "recover original WMT source sentences and exact MEPRS sample IDs, or state the limitation",
            "needs_gpu": "no",
            "needs_api": "no",
            "needs_human": "no",
        },
        {
            "item": "full-set reference translations",
            "status": "missing for the full 16 x 200 MEPRS test conditions",
            "local_evidence": f"dataset reference/ref-like files: {len(ref_like)}; public-overlap has 48-69 high-confidence matches per available direction",
            "action_needed": "recover original references before full COMET/BLEU/chrF/MBR metric recomputation",
            "needs_gpu": "no for BLEU/chrF; yes or recommended for COMET/CometKIWI",
            "needs_api": "no",
            "needs_human": "no",
        },
        {
            "item": "raw prompts and raw LLM responses",
            "status": "not found as raw artifacts in the current package",
            "local_evidence": f"prompt-like files: {len(prompt_like)}; non-md response-like files: {len(response_like)}",
            "action_needed": "ask authors to provide raw prompts/responses if available; otherwise describe parsed-score release only",
            "needs_gpu": "no",
            "needs_api": "no",
            "needs_human": "no",
        },
        {
            "item": "API token, latency, and cost logs",
            "status": "not found in the current package",
            "local_evidence": f"token-like files: {len(token_like)}; latency-like files: {len(latency_like)}; api-like files: {len(api_like)}",
            "action_needed": "provide actual API logs or keep formula-based call-count/cost discussion",
            "needs_gpu": "no",
            "needs_api": "no",
            "needs_human": "no",
        },
        {
            "item": "human evaluation labels",
            "status": "materials prepared but labels not collected",
            "local_evidence": f"blind-sheet rows: {len(human_sheet)}; filled rating/preference cells: {filled_label_cells}",
            "action_needed": "collect human adequacy/fluency/preference labels or report as uncollected",
            "needs_gpu": "no",
            "needs_api": "no",
            "needs_human": "yes",
        },
        {
            "item": "Google MQM GPU training outputs",
            "status": "complete for the planned five local runs",
            "local_evidence": f"completed configured output dirs: {sum(1 for path in google_dirs if path.exists())}/5; checkpoint count: {checkpoint_count}",
            "action_needed": "no more GPU needed unless adding COMET/CometKIWI or new experiments",
            "needs_gpu": "no",
            "needs_api": "no",
            "needs_human": "no",
        },
        {
            "item": "qualitative error-case materials",
            "status": "prepared from existing outputs",
            "local_evidence": "qualitative_error_cases_all.csv, top_aligned.csv, blind_sheet.csv, and answer_key.csv generated",
            "action_needed": "optional human review can fill error_type/comments fields",
            "needs_gpu": "no",
            "needs_api": "no",
            "needs_human": "optional",
        },
    ]
    write_csv(out_dir / "remaining_gap_evidence_matrix.csv", rows)


def write_api_call_scenarios(out_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    scenarios = [
        ("full_MEPRS_5_generators_5_reviewers_per_condition", 5, 5, 200, 1),
        ("full_MEPRS_5x5_all_16_conditions", 5, 5, 200, 16),
        ("low_cost_5_generators_3_reviewers_per_condition", 5, 3, 200, 1),
        ("low_cost_3_generators_5_reviewers_per_condition", 3, 5, 200, 1),
        ("low_cost_3_generators_3_reviewers_per_condition", 3, 3, 200, 1),
        ("low_cost_2_generators_5_reviewers_per_condition", 2, 5, 200, 1),
    ]
    for name, n_generators, n_reviewers, n_sentences, n_conditions in scenarios:
        generation_calls = n_generators * n_sentences * n_conditions
        review_calls = n_generators * n_reviewers * n_sentences * n_conditions
        rows.append(
            {
                "scenario": name,
                "n_generators": n_generators,
                "n_reviewers": n_reviewers,
                "sentences_per_condition": n_sentences,
                "conditions": n_conditions,
                "generation_calls": generation_calls,
                "review_calls": review_calls,
                "total_calls": generation_calls + review_calls,
                "cost_note": "Dollar cost requires model-specific token logs and current API rates; not inferable from released score files.",
            }
        )
    write_csv(out_dir / "api_call_count_scenarios.csv", rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "revision_analysis" / "remaining_gap_audit_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    collect_qualitative_cases(root, out_dir)
    collect_gap_evidence(root, out_dir)
    write_api_call_scenarios(out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
