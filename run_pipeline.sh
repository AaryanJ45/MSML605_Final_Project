set -euo pipefail

# Configuration
MODEL="${MODEL:-bert}"
LOCAL="${LOCAL:-true}"
BUCKET="${BUCKET:-}"
FILE_NAME="${FILE_NAME:-bias_clean.csv}"
PYTHON="${PYTHON:-python3}"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"


log()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }
fail() { log "ERROR: $*"; exit 1; }

mkdir -p "${LOG_DIR}"

[[ "${MODEL}" == "bert" || "${MODEL}" == "distilbert" ]] \
    || fail "MODEL must be 'bert' or 'distilbert', got '${MODEL}'"

if [[ "${LOCAL}" == "false" && -z "${BUCKET}" ]]; then
    fail "BUCKET env var must be set when LOCAL=false"
fi

log "====== MLOps Pipeline Start ======"
log "Model=${MODEL} | Local=${LOCAL} | File=${FILE_NAME}"
log "Full log: ${LOG_FILE}"

# Preprocessing step
log "Step 1: Preprocessing"

PREPROCESS_ARGS="--file_name ${FILE_NAME} --local ${LOCAL}"
[[ -n "${BUCKET}" ]] && PREPROCESS_ARGS+=" --bucket ${BUCKET}"

${PYTHON} preprocess.py ${PREPROCESS_ARGS} 2>&1 | tee -a "${LOG_FILE}"
log "Preprocessing complete."

# Training step
log "Step 2: Training (model=${MODEL})"

TRAIN_ARGS="--model ${MODEL}"
[[ "${LOCAL}" == "true" ]] && TRAIN_ARGS+=" --local"

${PYTHON} train.py ${TRAIN_ARGS} 2>&1 | tee -a "${LOG_FILE}"
log "Training complete."

