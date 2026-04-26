set -euo pipefail

# Configuration
MODEL="${MODEL:-bert}"
LOCAL="${LOCAL:-true}"
BUCKET="${BUCKET:-}"
FILE_NAME="${FILE_NAME:-bias_clean.csv}"
PYTHON="${PYTHON:-python3}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"       # true = skip preprocess+train, pull model from S3
PREPROCESS_ONLY="${PREPROCESS_ONLY:-false}" # true = preprocess + upload to S3, then stop
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
log "Model=${MODEL} | Local=${LOCAL} | SkipTrain=${SKIP_TRAIN} | PreprocessOnly=${PREPROCESS_ONLY} | File=${FILE_NAME}"
log "Full log: ${LOG_FILE}"

# ── Mode 1: PREPROCESS_ONLY ───────────────────────────────────────────────────
# Run preprocess.py, upload splits to S3, then stop.
# Used when training will happen in Colab.
if [[ "${PREPROCESS_ONLY}" == "true" ]]; then
    if [[ "${LOCAL}" == "true" ]]; then
        fail "PREPROCESS_ONLY=true requires LOCAL=false and a BUCKET to upload splits to S3."
    fi

    log "Step 1: Preprocessing (preprocess-only mode)"

    PREPROCESS_ARGS="--file_name ${FILE_NAME} --bucket ${BUCKET}"
    ${PYTHON} preprocess.py ${PREPROCESS_ARGS} 2>&1 | tee -a "${LOG_FILE}"
    log "Preprocessing complete."

    log "Uploading preprocessed data to s3://${BUCKET}/preprocessed_data/ ..."
    aws s3 sync preprocessed_data/ "s3://${BUCKET}/preprocessed_data/" 2>&1 | tee -a "${LOG_FILE}"

    log "Uploading preprocessed CSVs to s3://${BUCKET}/ ..."
    aws s3 cp train.csv "s3://${BUCKET}/train.csv" 2>&1 | tee -a "${LOG_FILE}"
    aws s3 cp val.csv   "s3://${BUCKET}/val.csv"   2>&1 | tee -a "${LOG_FILE}"
    aws s3 cp test.csv  "s3://${BUCKET}/test.csv"  2>&1 | tee -a "${LOG_FILE}"

    log "Splits uploaded. Run train_colab.ipynb in Colab to train, then re-run with SKIP_TRAIN=true."
    log "====== Preprocessing Complete ======"
    exit 0
fi

# ── Mode 2: SKIP_TRAIN ────────────────────────────────────────────────────────
# Pull pre-trained model + preprocessed splits from S3, then validate + test.
# Used after Colab training is done.
if [[ "${SKIP_TRAIN}" == "true" ]]; then
    if [[ "${LOCAL}" == "true" ]]; then
        fail "SKIP_TRAIN=true requires LOCAL=false and a BUCKET so the model can be pulled from S3."
    fi

    log "Pulling preprocessed data from s3://${BUCKET}/preprocessed_data/ ..."
    aws s3 sync "s3://${BUCKET}/preprocessed_data/" preprocessed_data/ 2>&1 | tee -a "${LOG_FILE}"

    log "Pulling preprocessed CSVs from s3://${BUCKET}/ ..."
    aws s3 cp "s3://${BUCKET}/val.csv"  val.csv  2>&1 | tee -a "${LOG_FILE}"
    aws s3 cp "s3://${BUCKET}/test.csv" test.csv 2>&1 | tee -a "${LOG_FILE}"

    log "Pulling saved model from s3://${BUCKET}/saved_models/${MODEL}/ ..."
    aws s3 sync "s3://${BUCKET}/saved_models/${MODEL}/" "saved_models/${MODEL}/" 2>&1 | tee -a "${LOG_FILE}"

    log "Model and data ready."

# ── Mode 3: Full pipeline (default) ───────────────────────────────────────────
# Preprocess → train on EC2 → upload → validate → test.
else
    log "Step 1: Preprocessing"

    PREPROCESS_ARGS="--file_name ${FILE_NAME}"
    [[ "${LOCAL}" == "true" ]] && PREPROCESS_ARGS+=" --local"
    [[ -n "${BUCKET}" ]] && PREPROCESS_ARGS+=" --bucket ${BUCKET}"

    ${PYTHON} preprocess.py ${PREPROCESS_ARGS} 2>&1 | tee -a "${LOG_FILE}"
    log "Preprocessing complete."

    log "Step 2: Training (model=${MODEL})"

    TRAIN_ARGS="--model ${MODEL}"
    [[ "${LOCAL}" == "true" ]] && TRAIN_ARGS+=" --local"

    ${PYTHON} train.py ${TRAIN_ARGS} 2>&1 | tee -a "${LOG_FILE}"
    log "Training complete."

    if [[ "${LOCAL}" == "false" ]]; then
        log "Uploading saved model to s3://${BUCKET}/saved_models/${MODEL}/ ..."
        aws s3 sync "saved_models/${MODEL}/" "s3://${BUCKET}/saved_models/${MODEL}/" 2>&1 | tee -a "${LOG_FILE}"

        log "Uploading preprocessed data to s3://${BUCKET}/preprocessed_data/ ..."
        aws s3 sync preprocessed_data/ "s3://${BUCKET}/preprocessed_data/" 2>&1 | tee -a "${LOG_FILE}"

        log "Uploading preprocessed CSVs to s3://${BUCKET}/ ..."
        aws s3 cp train.csv "s3://${BUCKET}/train.csv" 2>&1 | tee -a "${LOG_FILE}"
        aws s3 cp val.csv   "s3://${BUCKET}/val.csv"   2>&1 | tee -a "${LOG_FILE}"
        aws s3 cp test.csv  "s3://${BUCKET}/test.csv"  2>&1 | tee -a "${LOG_FILE}"

        log "Model artifact uploaded to S3."
    fi
fi

# ── Validate + Test (all modes except PREPROCESS_ONLY) ────────────────────────
log "Step 3: Validation (val.csv)"

VALIDATE_ARGS="--model ${MODEL}"
[[ "${LOCAL}" == "true" ]] && VALIDATE_ARGS+=" --local"

${PYTHON} validate.py ${VALIDATE_ARGS} 2>&1 | tee -a "${LOG_FILE}"
log "Validation complete."

log "Step 4: Testing (test.csv)"

TEST_ARGS="--model ${MODEL}"
[[ "${LOCAL}" == "true" ]] && TEST_ARGS+=" --local"

${PYTHON} test.py ${TEST_ARGS} 2>&1 | tee -a "${LOG_FILE}"
log "Testing complete."

log "====== Pipeline Complete ======"