#!/usr/bin/env bash
# ==============================================================================
# impl_gen3 全フェーズ一括実行バッチ (phase1_preprocess 〜 phase5_inference)
# どのカレントディレクトリからでもファイルを直接指定して実行可能:
#   bash /workspaces/slm/impl_gen3/src/batches/run_all.sh
#   bash impl_gen3/src/batches/run_all.sh
# ==============================================================================
set -euo pipefail

# カレントディレクトリを変更せずにスクリプト自身の絶対パスから基準ディレクトリを解決
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
SRC_DIR="$(dirname "${SCRIPT_DIR}")"
GEN_DIR="$(dirname "${SRC_DIR}")"
REPO_ROOT="$(dirname "${GEN_DIR}")"
PIPELINES_DIR="${SRC_DIR}/pipelines/training"

# 仮想環境の Python を優先使用 (存在しない場合は PATH 上の python)
if [[ -x "/opt/venvs/.venv/bin/python" ]]; then
  PYTHON_BIN="/opt/venvs/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

# ジョブIDと出力ベースディレクトリ (第1引数または環境変数 JOB_ID で上書き可能)
JOB_ID="${1:-${JOB_ID:-job_gen3_1234}}"
JOB_DIR="/tmp/${JOB_ID}"
CHABSA_ORIGIN_DIR="${REPO_ROOT}/samples/chabsa_data/data"
LIVEDOOR_ORIGIN_DIR="${REPO_ROOT}/samples/livedoor/text"

echo "============================================================"
echo "[impl_gen3] Starting full pipeline"
echo "  GEN_DIR             : ${GEN_DIR}"
echo "  CHABSA_ORIGIN_DIR   : ${CHABSA_ORIGIN_DIR}"
echo "  LIVEDOOR_ORIGIN_DIR : ${LIVEDOOR_ORIGIN_DIR}"
echo "  JOB_DIR             : ${JOB_DIR}"
echo "  PYTHON_BIN          : ${PYTHON_BIN}"
echo "============================================================"

# Phase 1: 前処理（共通フォーマット抽出・正規化）
echo "[Phase 1/5] phase1_preprocess ..."
PYTHONPATH="${PIPELINES_DIR}/phase1_preprocess" \
"${PYTHON_BIN}" -m phase1_preprocess \
  --chabsa-directory="${CHABSA_ORIGIN_DIR}" \
  --livedoor-directory="${LIVEDOOR_ORIGIN_DIR}" \
  --out-slim-file="${JOB_DIR}/phase1_preprocess/slim.json" \
  --out-normalized-file="${JOB_DIR}/phase1_preprocess/normalized.json" \
  --out-meta-file="${JOB_DIR}/phase1_preprocess/metadata.json" \
  --out-context-file="${JOB_DIR}/phase1_preprocess/context.json"

# Phase 2: トークン化・語彙サイズ探索
echo "[Phase 2/5] phase2_tokenize ..."
PYTHONPATH="${PIPELINES_DIR}/phase2_tokenize" \
"${PYTHON_BIN}" -m phase2_tokenize \
  --input-context-file="${JOB_DIR}/phase1_preprocess/context.json" \
  --out-directory="${JOB_DIR}/phase2_tokenize" \
  --out-meta-file="${JOB_DIR}/phase2_tokenize/metadata.json" \
  --out-context-file="${JOB_DIR}/phase2_tokenize/context.json" \
  --vocab-size-start=1500 \
  --vocab-size-end=16000 \
  --vocab-size-exponent=1.03 \
  --model-type=unigram \
  --character-coverage=0.9995 \
  --user-defined-symbols='<s>,</s>,<n>' \
  --output-format=piece

# Phase 3: 特徴化
echo "[Phase 3/5] phase3_featurize ..."
PYTHONPATH="${PIPELINES_DIR}/phase3_featurize" \
"${PYTHON_BIN}" -m phase3_featurize \
  --input-context-file="${JOB_DIR}/phase2_tokenize/context.json" \
  --out-x-file="${JOB_DIR}/phase3_featurize/x.npz" \
  --out-y-file="${JOB_DIR}/phase3_featurize/y.npy" \
  --out-meta-file="${JOB_DIR}/phase3_featurize/metadata.json" \
  --out-context-file="${JOB_DIR}/phase3_featurize/context.json" \
  --context-length=32

# Phase 4: モデル学習
echo "[Phase 4/5] phase4_training ..."
PYTHONPATH="${PIPELINES_DIR}/phase4_training" \
"${PYTHON_BIN}" -m phase4_training \
  --input-context-file="${JOB_DIR}/phase3_featurize/context.json" \
  --work-directory="${JOB_DIR}/phase4_training" \
  --out-model-file="${JOB_DIR}/phase4_training/model.keras" \
  --out-meta-file="${JOB_DIR}/phase4_training/metadata.json" \
  --out-context-file="${JOB_DIR}/phase4_training/context.json" \
  --epochs=50 \
  --min-delta=0.001 \
  --patience=3

# Phase 5: 推論・文章生成
echo "[Phase 5/5] phase5_inference ..."
PYTHONPATH="${PIPELINES_DIR}/phase5_inference" \
"${PYTHON_BIN}" -m phase5_inference \
  --input-context-file="${JOB_DIR}/phase4_training/context.json" \
  --out-texts-file="${JOB_DIR}/phase5_inference/generated_texts.json" \
  --out-readable-file="${JOB_DIR}/phase5_inference/generated_texts.txt" \
  --out-meta-file="${JOB_DIR}/phase5_inference/metadata.json" \
  --out-context-file="${JOB_DIR}/phase5_inference/context.json" \
  --num-documents=10 \
  --max-chars=1000 \
  --sampling-mode=random \
  --prob-power=2.0

echo "============================================================"
echo "[impl_gen3] Completed! Outputs saved to: ${JOB_DIR}"
echo "  Readable texts: ${JOB_DIR}/phase5_inference/generated_texts.txt"
echo "============================================================"
