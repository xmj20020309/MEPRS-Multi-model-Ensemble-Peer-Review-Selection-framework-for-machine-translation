# Google MQM char-CNN full checkpoint run

## Fold meaning

Here, one fold means one leave-one-language-pair-out training/evaluation round.
For example, the `en-de` fold trains on the other language pairs and tests on
`en-de`. This run has five held-out language pairs and two random seeds, so it
produces 10 folds and 10 saved checkpoints.

## Data

- Prepared data file:
  `revision_analysis/google_mqm_prepared/google_mqm_segment_scores.csv`
- Rows: 403,489 segment-level MQM samples
- Source segments: 12,612
- Language pairs: en-de, en-es, he-en, ja-zh, zh-en

## Training command

```bash
python revision_analysis/train_external_mqm_charcnn.py \
  --data_dir revision_analysis/google_mqm_prepared/google_mqm_segment_scores.csv \
  --output_dir revision_analysis/google_mqm_charcnn_outputs_full_ckpt \
  --device cuda \
  --epochs 8 \
  --seeds 1 2 \
  --batch_size 256 \
  --max_chars 512 \
  --patience 3 \
  --input_mode full \
  --skip_predictions \
  --save_checkpoints
```

## Saved files

- `external_mqm_charcnn_summary.csv`: aggregate metrics.
- `external_mqm_charcnn_folds.csv`: per-fold metrics, best epoch, validation
  loss, and checkpoint path.
- `external_mqm_charcnn_folds.partial.csv`: same fold table written during
  training so intermediate progress is preserved.
- `external_mqm_charcnn_metadata.json`: run arguments and data inventory.
- `google_mqm_charcnn_full_ckpt.log`: remote run log.
- `checkpoints/*.pt`: one best checkpoint per seed and held-out language pair.

Each checkpoint stores:

- `model_state_dict`
- `vocab`
- `feature_mean` and `feature_std`
- `label_mean` and `label_std`
- `model_config`
- `training` history, best epoch, completed epochs, and best validation loss

## Checkpoints

- `checkpoints/charcnn_seed1_ende.pt`
- `checkpoints/charcnn_seed1_enes.pt`
- `checkpoints/charcnn_seed1_heen.pt`
- `checkpoints/charcnn_seed1_jazh.pt`
- `checkpoints/charcnn_seed1_zhen.pt`
- `checkpoints/charcnn_seed2_ende.pt`
- `checkpoints/charcnn_seed2_enes.pt`
- `checkpoints/charcnn_seed2_heen.pt`
- `checkpoints/charcnn_seed2_jazh.pt`
- `checkpoints/charcnn_seed2_zhen.pt`

## Main result

- Sentence-level Spearman: 0.651
- System-level Spearman: 0.695
- Pairwise accuracy: 0.696
- Top-1 accuracy: 0.947

This is the main saved checkpoint run for the larger Google MQM supplementary
experiment.
