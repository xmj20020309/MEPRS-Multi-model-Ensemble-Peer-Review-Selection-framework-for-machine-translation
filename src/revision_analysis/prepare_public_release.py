from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[4]
OUT_DIR = WORKSPACE / "release_ready_20260707"
GITHUB_STAGING = OUT_DIR / "MEPRS_github_release_20260707"
ZENODO_STAGING = OUT_DIR / "MEPRS_zenodo_full_release_20260707"
GITHUB_ZIP = OUT_DIR / "MEPRS_github_release_20260707.zip"
ZENODO_ZIP = OUT_DIR / "MEPRS_zenodo_full_release_20260707.zip"


INTERNAL_PATTERNS = [
    "revision_analysis/__pycache__/**",
    "revision_analysis/**/*.pyc",
    "revision_analysis/api_micro_eval_outputs/**",
    "revision_analysis/*working_draft*",
    "revision_analysis/*submission_draft*",
    "revision_analysis/*submission_checklist*",
    "revision_analysis/*acceptable_submission_state*",
    "revision_analysis/*unfinished_items*",
    "revision_analysis/*mitigation_strategy*",
    "revision_analysis/*remote_training_status*",
    "revision_analysis/*gpu_revision_experiment_plan*",
    "revision_analysis/*high_cost_experiment_push_plan*",
    "revision_analysis/*reviewer_response_experiment_coverage*",
    "revision_analysis/apply_round*.py",
    "revision_analysis/prepare_submission_files.py",
    "revision_analysis/human_eval_annotator_package.zip",
    "revision_analysis/human_eval_to_send_annotators/**",
    "revision_analysis/**/*answer_key*",
    "revision_analysis/**/qualitative_error_case_answer_key.csv",
]

GITHUB_EXTRA_EXCLUDES = [
    "revision_analysis/google_mqm_raw/**",
    "revision_analysis/google_mqm_prepared/google_mqm_segment_scores.csv",
    "revision_analysis/**/*.pt",
    "revision_analysis/**/*.pth",
    "revision_analysis/**/*.ckpt",
    "revision_analysis/**/*.bin",
    "revision_analysis/**/*.safetensors",
    "revision_analysis/google_mqm_charcnn_outputs_full/external_mqm_charcnn_predictions.csv",
    "revision_analysis/google_mqm_charcnn_outputs_fast/external_mqm_charcnn_predictions.csv",
]

ZENODO_EXTRA_EXCLUDES = []

INCLUDE_ROOTS = ["README.md", "code", "dataset", "revision_analysis"]

KEY_OUTPUT_DIRS = [
    "revision_analysis/outputs",
    "revision_analysis/remaining_low_cost_outputs",
    "revision_analysis/remaining_gap_audit_outputs",
    "revision_analysis/sacrebleu_overlap_outputs",
    "revision_analysis/surface_metric_outputs",
    "revision_analysis/surface_mbr_outputs",
    "revision_analysis/learned_reranker_outputs_gpu",
    "revision_analysis/google_mqm_prepared",
    "revision_analysis/google_mqm_charcnn_outputs_extended",
    "revision_analysis/google_mqm_charcnn_outputs_hyp_only_extended",
    "revision_analysis/google_mqm_charcnn_outputs_ref_hyp_extended",
    "revision_analysis/google_mqm_charcnn_outputs_wide_full",
    "revision_analysis/google_mqm_charcnn_outputs_more_seeds_full",
    "revision_analysis/google_mqm_charcnn_outputs_full_ckpt",
]


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for item in INCLUDE_ROOTS:
        path = REPO / item
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
    return sorted(files, key=lambda p: rel(p).lower())


def should_include(path: Path, mode: str) -> tuple[bool, str]:
    relative = rel(path)
    if matches(relative, INTERNAL_PATTERNS):
        return False, "internal_or_blind_eval_material"
    if mode == "github" and matches(relative, GITHUB_EXTRA_EXCLUDES):
        return False, "large_file_for_zenodo_only"
    if mode == "zenodo" and matches(relative, ZENODO_EXTRA_EXCLUDES):
        return False, "zenodo_exclude"
    return True, ""


def copy_tree(mode: str, staging: Path) -> list[dict]:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    rows: list[dict] = []
    for src in iter_repo_files():
        include, reason = should_include(src, mode)
        relative = rel(src)
        size = src.stat().st_size
        rows.append({"mode": mode, "path": relative, "size": size, "included": include, "reason": reason})
        if not include:
            continue
        dst = staging / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_zip(src_dir: Path, dst_zip: Path) -> None:
    if dst_zip.exists():
        dst_zip.unlink()
    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(src_dir).as_posix())


def summarize_dir(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()] if path.exists() else []
    checkpoints = [p for p in files if p.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}]
    return {
        "exists": path.exists(),
        "files": len(files),
        "checkpoint_files": len(checkpoints),
        "bytes": sum(p.stat().st_size for p in files),
    }


def write_manifest(all_rows: list[dict], package_info: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "release_file_manifest_20260707.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "path", "size", "included", "reason"])
        writer.writeheader()
        writer.writerows(all_rows)

    key_dirs = {d: summarize_dir(REPO / d) for d in KEY_OUTPUT_DIRS}
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(REPO),
        "remote_autodl_status": "SSH endpoint refused connection on 2026-07-07; local pulled artifacts are used.",
        "packages": package_info,
        "key_output_dirs": key_dirs,
        "public_exclusion_policy": {
            "excluded_from_both": INTERNAL_PATTERNS,
            "excluded_from_github_only": GITHUB_EXTRA_EXCLUDES,
        },
    }
    (OUT_DIR / "release_manifest_20260707.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# MEPRS Public Release Manifest",
        "",
        "Date: 2026-07-07",
        "",
        "This release was prepared from the local MEPRS revision workspace. The previous AutoDL/SeetaCloud SSH endpoint refused connection during the final check, so the packaged artifacts are the locally pulled code, data, outputs, logs, and checkpoints available on this machine.",
        "",
        "## Packages",
        "",
    ]
    for package in package_info:
        lines.extend(
            [
                f"- `{package['name']}`",
                f"  - size: {package['bytes']:,} bytes",
                f"  - sha256: `{package['sha256']}`",
                f"  - purpose: {package['purpose']}",
            ]
        )
    lines.extend(
        [
            "",
            "## GitHub Package",
            "",
            "Use `MEPRS_github_release_20260707.zip` for the public GitHub repository. It contains the original MEPRS code and dataset, revision analysis scripts, and lightweight result tables. Large checkpoints, raw Google MQM TSV files, the prepared 227 MB Google MQM segment CSV, and large prediction CSVs are intentionally excluded and should be linked through Zenodo.",
            "",
            "## Zenodo Package",
            "",
            "Use `MEPRS_zenodo_full_release_20260707.zip` for the DOI archive. It includes the large Google MQM raw/prepared data available locally, saved checkpoints, summary tables, fold tables, metadata, and logs, while still excluding internal submission-planning files, local API probe outputs, Python caches, and blind-evaluation answer keys.",
            "",
            "## Key Output Directories",
            "",
        ]
    )
    for directory, info in key_dirs.items():
        lines.append(
            f"- `{directory}`: files={info['files']}, checkpoints={info['checkpoint_files']}, size={info['bytes']:,} bytes"
        )
    lines.extend(
        [
            "",
            "## Excluded Public Materials",
            "",
            "- Local API probe outputs are excluded because they describe the local environment.",
            "- Internal Chinese planning/checklist files and response drafts are excluded because they are not reproducibility artifacts.",
            "- Answer-key files for blind human-evaluation sheets are excluded to preserve the possibility of future blind annotation. They can be archived later after annotation is complete if the authors decide to release them.",
        ]
    )
    (OUT_DIR / "RELEASE_MANIFEST_20260707.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(copy_tree("github", GITHUB_STAGING))
    rows.extend(copy_tree("zenodo", ZENODO_STAGING))

    write_zip(GITHUB_STAGING, GITHUB_ZIP)
    write_zip(ZENODO_STAGING, ZENODO_ZIP)

    package_info = [
        {
            "name": GITHUB_ZIP.name,
            "bytes": GITHUB_ZIP.stat().st_size,
            "sha256": sha256(GITHUB_ZIP),
            "purpose": "Lightweight GitHub upload package.",
        },
        {
            "name": ZENODO_ZIP.name,
            "bytes": ZENODO_ZIP.stat().st_size,
            "sha256": sha256(ZENODO_ZIP),
            "purpose": "Full DOI archive package with large local artifacts.",
        },
    ]
    write_manifest(rows, package_info)
    print(OUT_DIR)
    for item in package_info:
        print(f"{item['name']}\t{item['bytes']}\t{item['sha256']}")


if __name__ == "__main__":
    main()
