"""Prepare Google WMT MQM TSV files for revision experiments.

The Google MQM files are error-span/rater-level annotations. This helper
aggregates them to segment-level rows compatible with the external MQM
char-CNN script: Target_Index, system, src, ref, hyps, mqm_score.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


SEVERITY_WEIGHTS = {
    "no-error": 0.0,
    "no error": 0.0,
    "minor": 1.0,
    "major": 5.0,
    "critical": 25.0,
}


def infer_language_pair(path: Path, frame: pd.DataFrame) -> str:
    name = path.name.lower()
    for pair in ["ende", "enru", "enzh", "zhen", "heen", "enes", "jazh"]:
        if pair in name:
            return f"{pair[:2]}-{pair[2:]}"
    if {"source_lang", "target_lang"}.issubset(frame.columns):
        return f"{frame['source_lang'].iloc[0]}-{frame['target_lang'].iloc[0]}"
    return "unknown"


def normalized_severity(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("_", "-")
    if text in {"nan", "", "none"}:
        return ""
    if text.startswith("{"):
        return ""
    return text


def severity_penalty(value: object) -> float | None:
    text = normalized_severity(value)
    if not text:
        return None
    if text in SEVERITY_WEIGHTS:
        return SEVERITY_WEIGHTS[text]
    if "no-error" in text or "no error" in text:
        return 0.0
    if "minor" in text:
        return 1.0
    if "major" in text:
        return 5.0
    if "critical" in text:
        return 25.0
    return None


def segment_id_columns(frame: pd.DataFrame) -> tuple[str, str]:
    doc_col = "doc_id" if "doc_id" in frame.columns else "doc"
    if "globalSegId" in frame.columns:
        seg_col = "globalSegId"
    elif "seg_id" in frame.columns:
        seg_col = "seg_id"
    elif "doc_segment_id" in frame.columns:
        seg_col = "doc_segment_id"
    else:
        raise ValueError("No segment id column found")
    return doc_col, seg_col


def system_column(frame: pd.DataFrame) -> str:
    if "system_id" in frame.columns:
        return "system_id"
    return "system"


def target_column(frame: pd.DataFrame) -> str:
    if "candidate" in frame.columns:
        return "candidate"
    return "target"


def prepare_file(path: Path, min_raters: int) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, sep="\t", dtype=str, on_bad_lines="skip")
    required = {"source", "category", "severity"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} missing required columns {sorted(required - set(frame.columns))}")

    sys_col = system_column(frame)
    hyp_col = target_column(frame)
    doc_col, seg_col = segment_id_columns(frame)
    pair = infer_language_pair(path, frame)
    frame["penalty"] = frame["severity"].map(severity_penalty)
    clean = frame.dropna(subset=["penalty", "source", hyp_col, sys_col, seg_col]).copy()
    clean["penalty"] = clean["penalty"].astype(float)
    if clean.empty:
        return pd.DataFrame(), {
            "file": path.name,
            "language_pair": pair,
            "raw_rows": int(len(frame)),
            "clean_rows": 0,
            "segment_rows": 0,
            "dropped_reason": "no valid severity rows",
        }

    group_cols = [sys_col, doc_col, seg_col, "source", hyp_col]
    if "reference" in clean.columns:
        group_cols.append("reference")
    if "rater" in clean.columns:
        rater_col = "rater"
    elif "rater_id" in clean.columns:
        rater_col = "rater_id"
    else:
        rater_col = None

    rows = []
    for keys, group in clean.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = dict(zip(group_cols, keys))
        if rater_col and group[rater_col].nunique() < min_raters:
            continue
        penalty = float(group["penalty"].sum())
        doc = str(values[doc_col])
        seg = str(values[seg_col])
        rows.append(
            {
                "Target_Index": f"{pair}:{doc}:{seg}",
                "system": str(values[sys_col]),
                "src": str(values["source"]),
                "ref": str(values.get("reference", "")),
                "hyps": str(values[hyp_col]),
                "mqm_score": -penalty,
                "language_pair": pair,
                "source_file": path.name,
                "raw_penalty": penalty,
            }
        )

    output = pd.DataFrame(rows)
    metadata = {
        "file": path.name,
        "language_pair": pair,
        "raw_rows": int(len(frame)),
        "clean_rows": int(len(clean)),
        "segment_rows": int(len(output)),
        "systems": int(output["system"].nunique()) if not output.empty else 0,
        "segments": int(output["Target_Index"].nunique()) if not output.empty else 0,
    }
    return output, metadata


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    reports = []
    for path in sorted(input_dir.glob("*.tsv")):
        try:
            frame, report = prepare_file(path, args.min_raters)
        except Exception as exc:
            reports.append(
                {
                    "file": path.name,
                    "language_pair": "unknown",
                    "raw_rows": 0,
                    "clean_rows": 0,
                    "segment_rows": 0,
                    "error": str(exc),
                }
            )
            continue
        reports.append(report)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError("No Google MQM segment rows were prepared")

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["language_pair", "Target_Index", "system", "hyps", "mqm_score"])
    data.to_csv(output_dir / "google_mqm_segment_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reports).to_csv(output_dir / "google_mqm_prepare_report.csv", index=False, encoding="utf-8-sig")

    summary = {
        "rows": int(len(data)),
        "language_pairs": sorted(data["language_pair"].unique().tolist()),
        "rows_by_pair": data.groupby("language_pair").size().to_dict(),
        "segments_by_pair": data.groupby("language_pair")["Target_Index"].nunique().to_dict(),
        "systems_by_pair": data.groupby("language_pair")["system"].nunique().to_dict(),
    }
    (output_dir / "google_mqm_prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="revision_analysis/google_mqm_raw")
    parser.add_argument("--output_dir", default="revision_analysis/google_mqm_prepared")
    parser.add_argument("--min_raters", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
