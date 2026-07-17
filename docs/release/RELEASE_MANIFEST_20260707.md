# MEPRS Public Release Manifest

Date: 2026-07-07

This release was prepared from the local MEPRS revision workspace. The previous AutoDL/SeetaCloud SSH endpoint refused connection during the final check, so the packaged artifacts are the locally pulled code, data, outputs, logs, and checkpoints available on this machine.

## Packages

- `MEPRS_github_release_20260707.zip`
  - size: 17,520,449 bytes
  - sha256: `dcf7bfe9a41a338b66dd83ce9232a1b02dfba8d708a84a9718df47aca8398dd7`
  - purpose: Lightweight GitHub upload package.
- `MEPRS_zenodo_full_release_20260707.zip`
  - size: 539,454,828 bytes
  - sha256: `9689785ee601b6ba56e23906ddc549504a4fba33b40137aa5285afef92e60c9c`
  - purpose: Full DOI archive package with large local artifacts.

## GitHub Package

Use `MEPRS_github_release_20260707.zip` for the public GitHub repository. It contains the original MEPRS code and dataset, revision analysis scripts, and lightweight result tables. Large checkpoints, raw Google MQM TSV files, the prepared 227 MB Google MQM segment CSV, and large prediction CSVs are intentionally excluded and should be linked through Zenodo.

## Zenodo Package

Use `MEPRS_zenodo_full_release_20260707.zip` for the DOI archive. It includes the large Google MQM raw/prepared data available locally, saved checkpoints, summary tables, fold tables, metadata, and logs, while still excluding internal submission-planning files, local API probe outputs, Python caches, and blind-evaluation answer keys.

## Key Output Directories

- `revision_analysis/outputs`: files=19, checkpoints=0, size=1,923,784 bytes
- `revision_analysis/remaining_low_cost_outputs`: files=6, checkpoints=0, size=1,608,430 bytes
- `revision_analysis/remaining_gap_audit_outputs`: files=10, checkpoints=0, size=193,833 bytes
- `revision_analysis/sacrebleu_overlap_outputs`: files=1, checkpoints=0, size=5,889 bytes
- `revision_analysis/surface_metric_outputs`: files=2, checkpoints=0, size=456,726 bytes
- `revision_analysis/surface_mbr_outputs`: files=2, checkpoints=0, size=511,418 bytes
- `revision_analysis/learned_reranker_outputs_gpu`: files=3, checkpoints=0, size=17,924 bytes
- `revision_analysis/google_mqm_prepared`: files=3, checkpoints=0, size=227,176,653 bytes
- `revision_analysis/google_mqm_charcnn_outputs_extended`: files=19, checkpoints=15, size=54,150,032 bytes
- `revision_analysis/google_mqm_charcnn_outputs_hyp_only_extended`: files=19, checkpoints=15, size=47,929,456 bytes
- `revision_analysis/google_mqm_charcnn_outputs_ref_hyp_extended`: files=19, checkpoints=15, size=47,930,293 bytes
- `revision_analysis/google_mqm_charcnn_outputs_wide_full`: files=19, checkpoints=15, size=84,108,814 bytes
- `revision_analysis/google_mqm_charcnn_outputs_more_seeds_full`: files=29, checkpoints=25, size=90,335,936 bytes
- `revision_analysis/google_mqm_charcnn_outputs_full_ckpt`: files=16, checkpoints=10, size=36,103,123 bytes

## Excluded Public Materials

- Local API probe outputs are excluded because they describe the local environment.
- Internal Chinese planning/checklist files and response drafts are excluded because they are not reproducibility artifacts.
- Answer-key files for blind human-evaluation sheets are excluded to preserve the possibility of future blind annotation. They can be archived later after annotation is complete if the authors decide to release them.
