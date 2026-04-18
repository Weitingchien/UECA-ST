#!/usr/bin/env bash

set -euo pipefail

# Default settings (can be overridden by environment variables before running)
BASE="${BASE:-/mnt/f/experiments/ep_split10_t1te1v1_u7_disjoint/prompt_ECPE_few_shot_ST_2026_04_13_17_39_17_f1-10_i70_lr1e-5_bs8_wd0.01_nest_k5_cause_clause_knncause_clause_mask_nestmul1_nbeta0.1_nm0.6_gamma0.5_nlmnest_st5_ste20_seed60_remove_pseudo_consistency_v2_CE}"
REPO_ROOT="${REPO_ROOT:-/mnt/f/experiments}"
DATASET="${DATASET:-split10_train1_test1_val1_unlabeled7_disjoint/}"
GT_DIR="${GT_DIR:-/mnt/d/GithubRepo/UECA_ST/${DATASET%/}}"
START_FOLD="${START_FOLD:-1}"
END_FOLD="${END_FOLD:-10}"
AVG3_CKPT_DIR="${AVG3_CKPT_DIR:-${BASE}_avg3_ckpt_alias}"
OUT="${OUT:-${BASE}/avg3_test_outputs_$(date +%Y%m%d_%H%M%S)}"

cd /mnt/d/GithubRepo/UECA_ST

mkdir -p "${AVG3_CKPT_DIR}/self_training_models"

export BASE REPO_ROOT AVG3_CKPT_DIR START_FOLD END_FOLD

echo "=== Config ==="
echo "BASE=${BASE}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "DATASET=${DATASET}"
echo "GT_DIR=${GT_DIR}"
echo "START_FOLD=${START_FOLD}"
echo "END_FOLD=${END_FOLD}"
echo "AVG3_CKPT_DIR=${AVG3_CKPT_DIR}"
echo "OUT=${OUT}"

echo "=== Step 1: Build avg3 checkpoints from emo/cause/pair best models ==="
python - <<'PY'
import os
from NeST.model_weight_averaging import build_averaged_model_for_fold, save_averaged_model

base = os.environ["BASE"]
repo_root = os.environ["REPO_ROOT"]
avg3_dir = os.environ["AVG3_CKPT_DIR"]
start_fold = int(os.environ["START_FOLD"])
end_fold = int(os.environ["END_FOLD"])

ok = []
failed = []

for fold in range(start_fold, end_fold + 1):
    try:
        result = build_averaged_model_for_fold(
            save_path=base,
            fold=fold,
            weights=None,
            map_location="cpu",
            repo_root=repo_root,
        )

        out_model = f"{avg3_dir}/self_training_models/fold{fold}_self_training_best_pair.pth"
        out_sd = f"{avg3_dir}/self_training_models/fold{fold}_self_training_best_pair_state_dict.pth"

        save_averaged_model(
            model=result["model"],
            state_dict=result["state_dict"],
            output_model_path=out_model,
            output_state_dict_path=out_sd,
        )

        print(f"[fold{fold}] OK -> {out_model}")
        ok.append(fold)
    except Exception as exc:
        print(f"[fold{fold}] FAILED -> {exc}")
        failed.append((fold, str(exc)))

print("Success folds:", ok)
if failed:
    print("Failed folds:")
    for fold, msg in failed:
        print(f" - fold{fold}: {msg}")
    raise SystemExit(1)
PY

echo "=== Step 2: Test using avg3 checkpoints to a new output directory ==="
python UECA_CE_few_shot_ST_nest_consistency_somc_v2.py \
  --test_only True \
  --checkpoint True \
  --checkpointpath "${AVG3_CKPT_DIR}" \
  --test_model_type self_training \
    --dataset "${DATASET}" \
  --experiment_output_dir "${OUT}" \
  --start_fold "${START_FOLD}" \
  --end_fold "${END_FOLD}"

echo "=== Step 3: Evaluate ==="
python run_evaluation.py \
  --gt_dir "${GT_DIR}" \
  --pred_dir "${OUT}"

echo "=== Done ==="
echo "All outputs are in: ${OUT}"
