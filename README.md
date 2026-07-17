# MEPRS reproducibility repository

This repository contains public code, released data, revision experiments, and result files for the MEPRS framework: Multi-model Ensemble Peer Review Selection for machine translation.

This public repository is a reproducibility archive for code, released data, revision experiments, and generated result files.

## Repository layout

- `src/meprs_selection/`: original MEPRS selection and ablation scripts.
- `src/revision_analysis/`: scripts used for revision analyses, robustness checks, external MQM experiments, human-evaluation preparation, and result summarization.
- `src/latest_model_judges/`: scripts for the July 2026 language-model judge comparison.
- `data/meprs_released_dataset/`: released candidate translations, BLEURT scores, language-model reviewer scores, and dimension scores.
- `data/external_mqm/`: prepared public MQM data used in external validation experiments.
- `data/latest_model_judge_inputs/`: input pairs used for the latest model-judge comparison.
- `experiments/human_evaluation/`: human-evaluation materials and raw returned annotations.
- `experiments/latest_model_judges/`: input materials for the latest model-judge experiment.
- `results/revision_analysis/`: revision outputs, robustness diagnostics, external MQM outputs, surface-metric baselines, and learned-reranker outputs.
- `results/human_evaluation/`: human-evaluation summaries.
- `results/latest_model_judges/`: raw and summarized latest model-judge results.
- `docs/`: experiment notes, public-archive notes, and release manifests.

## Quick start

Create an environment and install the lightweight Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Reproduce the original BLEURT-based MEPRS selection example:

```bash
python src/meprs_selection/model.py \
  --src_lan en \
  --tgt_lan zh \
  --forward it \
  --metric bleurt \
  --models gpt-3.5-turbo gpt-4o claude-3-opus claude-3.5-sonnet gemini-pro \
  --data_dir data/meprs_released_dataset
```

Run the core revision analysis on the released data:

```bash
python src/revision_analysis/run_revision_analysis.py \
  --data_dir data/meprs_released_dataset \
  --output_dir results/revision_analysis/outputs \
  --bootstrap 10000
```

Run the learned-reranker revision experiment:

```bash
python src/revision_analysis/train_learned_reranker.py \
  --data_dir data/meprs_released_dataset \
  --output_dir results/revision_analysis/learned_reranker_outputs_gpu \
  --epochs 400 \
  --seeds 1 2 3 4 5
```

Summaries for the completed human evaluation and latest model-judge experiments are already included:

```text
results/human_evaluation/summaries/
results/latest_model_judges/summaries/
docs/human_evaluation/eval_writeup_20260712.md
docs/latest_model_judges/experiment_summary_20260712.md
```

## Notes on live API experiments

The latest model-judge scripts use an OpenAI-compatible chat-completions endpoint. To rerun live calls, set local credentials outside the repository, for example through environment variables. API keys, latency logs with private credentials, and account-specific billing records are not included in this public archive.

## Citation and archival use

For journal submission, upload this repository to GitHub and archive the corresponding code/data release on Zenodo or an equivalent long-term repository. After the DOI is minted, cite the DOI in the Data Availability and Code Availability statements.
