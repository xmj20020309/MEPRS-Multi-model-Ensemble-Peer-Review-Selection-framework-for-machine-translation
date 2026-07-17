# Repository layout

This archive uses a standard public-repository layout.

- `src/` contains executable scripts.
- `data/` contains released or prepared public data.
- `experiments/` contains experiment inputs and raw human-evaluation materials.
- `results/` contains generated outputs and summary tables.
- `docs/` contains notes, manifests, and public-archive documentation.

The original July 2026 release used `code/`, `dataset/`, and `revision_analysis/`. Those materials have been reorganized here as follows:

- `code/` -> `src/meprs_selection/`
- `dataset/` -> `data/meprs_released_dataset/`
- `revision_analysis/*.py` -> `src/revision_analysis/`
- `revision_analysis/google_mqm_prepared/` -> `data/external_mqm/google_mqm_prepared/`
- `revision_analysis/human_eval_materials/` -> `experiments/human_evaluation/prepared_materials/`
- `revision_analysis/*outputs*/` and related result folders -> `results/revision_analysis/`
- July 2026 human-evaluation summaries -> `results/human_evaluation/`
- July 2026 latest model-judge code, inputs, and outputs -> `src/latest_model_judges/`, `data/latest_model_judge_inputs/`, and `results/latest_model_judges/`
