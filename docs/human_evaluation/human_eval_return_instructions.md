# Human-evaluation return processing

This note records the human-evaluation processing workflow. The public repository already includes the final returned annotations and summary tables, so this workflow normally does not need to be rerun.

## File locations

- Prepared annotation materials: `experiments/human_evaluation/prepared_materials/`
- Returned raw annotations: `experiments/human_evaluation/raw_results/`
- Summary results: `results/human_evaluation/summaries/`
- Scoring script: `src/revision_analysis/score_human_eval.py`

## Recompute one returned annotation file

Replace the input filename as needed:

```bash
python src/revision_analysis/score_human_eval.py \
  --sheet experiments/human_evaluation/raw_results/annotator1_filled.csv \
  --key experiments/human_evaluation/prepared_materials/human_eval_answer_key_en_he_tear_aligned_60.csv \
  --out_dir results/human_evaluation/summaries/recomputed_annotator1
```

If multiple annotators are processed separately, keep one output directory per annotator. In reporting, describe this material as a focused qualitative human evaluation or targeted human evaluation, not as a full-scale random human evaluation.
