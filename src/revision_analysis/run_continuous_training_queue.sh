#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/autodl-tmp/meprs
source /root/autodl-tmp/venvs/meprs-cu128/bin/activate

LOG_DIR="revision_analysis/logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/continuous_training_queue.log"
TRAIN_SCRIPT="revision_analysis/train_external_mqm_charcnn.py"
GOOGLE_MQM_CSV="revision_analysis/google_mqm_prepared/google_mqm_segment_scores.csv"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

wait_for_existing_extended() {
  log "Checking whether the manually started extended Google MQM job is still running."
  while pgrep -af "train_external_mqm_charcnn.py .*google_mqm_charcnn_outputs_extended" \
    | grep -v "run_continuous_training_queue.sh" >/dev/null; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv,noheader \
      | tee -a "${QUEUE_LOG}" >/dev/null || true
    sleep 60
  done
  log "No existing extended Google MQM job detected; continuing queue."
}

run_charcnn_job() {
  local name="$1"
  shift
  local out_dir="revision_analysis/${name}"
  local job_log="${LOG_DIR}/${name}.log"
  log "START ${name}"
  log "Output directory: ${out_dir}"
  python "${TRAIN_SCRIPT}" \
    --data_dir "${GOOGLE_MQM_CSV}" \
    --output_dir "${out_dir}" \
    --device cuda \
    --batch_size 256 \
    --max_chars 512 \
    --save_checkpoints \
    --skip_predictions \
    "$@" 2>&1 | tee -a "${job_log}" | tee -a "${QUEUE_LOG}"
  log "END ${name}"
}

log "Continuous training queue started."
wait_for_existing_extended

run_charcnn_job google_mqm_charcnn_outputs_hyp_only_extended \
  --epochs 20 \
  --seeds 1 2 3 \
  --input_mode hyp_only \
  --char_dim 96 \
  --channels 96

run_charcnn_job google_mqm_charcnn_outputs_ref_hyp_extended \
  --epochs 20 \
  --seeds 1 2 3 \
  --input_mode ref_hyp \
  --char_dim 96 \
  --channels 96

run_charcnn_job google_mqm_charcnn_outputs_wide_full \
  --epochs 20 \
  --seeds 1 2 3 \
  --input_mode full \
  --char_dim 128 \
  --channels 128

run_charcnn_job google_mqm_charcnn_outputs_more_seeds_full \
  --epochs 20 \
  --seeds 6 7 8 9 10 \
  --input_mode full \
  --char_dim 96 \
  --channels 96

log "Continuous training queue completed all scheduled jobs."
