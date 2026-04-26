set -euo pipefail

# Configuration
MODEL="${MODEL:-bert}"
LOCAL="${LOCAL:-true}"
BUCKET="${BUCKET:-}"
FILE_NAME="${FILE_NAME:-bias_clean.csv}"
PYTHON="${PYTHON:-python3}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
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

log "Starting UP"
log "Model=${MODEL} | Local=${LOCAL} | SkipTrain=${SKIP_TRAIN} | File=${FILE_NAME}"
log "Full log: ${LOG_FILE}"

# If we skip the training, it will pull the pre-trained model and preprocessed data from the S3
if [[ "${SKIP_TRAIN}" == "true" ]]; then
    log "SKIP_TRAIN=true — pulling pre-trained model and preprocessed data from S3."

    if [[ "${LOCAL}" == "true" ]]; then
        fail "SKIP_TRAIN=true requires LOCAL=false and a BUCKET so the model can be pulled from S3."
    fi

    log "Pulling preprocessed data from s3://${BUCKET}/preprocessed_data/ ..."
    aws s3 sync "s3://${BUCKET}/preprocessed_data/" preprocessed_data/ 2>&1 | tee -a "${LOG_FILE}"

    log "Pulling preprocessed CSVs from s3://${BUCKET}/ ..."
    aws s3 cp "s3://${BUCKET}/train.csv" train.csv 2>&1 | tee -a "${LOG_FILE}"
    aws s3 cp "s3://${BUCKET}/val.csv"   val.csv   2>&1 | tee -a "${LOG_FILE}"
    aws s3 cp "s3://${BUCKET}/test.csv"  test.csv  2>&1 | tee -a "${LOG_FILE}"

    log "Pulling saved model from s3://${BUCKET}/saved_models/${MODEL}/ ..."
    aws s3 sync "s3://${BUCKET}/saved_models/${MODEL}/" "saved_models/${MODEL}/" 2>&1 | tee -a "${LOG_FILE}"

    log "Pre-trained model and data ready."
else
    # Preprocessing step
    log "Step 1: Preprocessing"

    PREPROCESS_ARGS="--file_name ${FILE_NAME}"
    [[ "${LOCAL}" == "true" ]] && PREPROCESS_ARGS+=" --local"
    [[ -n "${BUCKET}" ]] && PREPROCESS_ARGS+=" --bucket ${BUCKET}"

    ${PYTHON} preprocess.py ${PREPROCESS_ARGS} 2>&1 | tee -a "${LOG_FILE}"
    log "Preprocessing complete."

    # Training step
    log "Step 2: Training (model=${MODEL})"

    TRAIN_ARGS="--model ${MODEL}"
    [[ "${LOCAL}" == "true" ]] && TRAIN_ARGS+=" --local"

    ${PYTHON} train.py ${TRAIN_ARGS} 2>&1 | tee -a "${LOG_FILE}"
    log "Training complete."

    # Uploading the model artifact to S3 after training
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

# Validation step
log "Step 3: Validation (val.csv)"

VALIDATE_ARGS="--model ${MODEL}"
[[ "${LOCAL}" == "true" ]] && VALIDATE_ARGS+=" --local"

${PYTHON} validate.py ${VALIDATE_ARGS} 2>&1 | tee -a "${LOG_FILE}"
log "Validation complete."

# Testing step
log "Step 4: Testing (test.csv)"

TEST_ARGS="--model ${MODEL}"
[[ "${LOCAL}" == "true" ]] && TEST_ARGS+=" --local"

${PYTHON} test.py ${TEST_ARGS} 2>&1 | tee -a "${LOG_FILE}"
log "Testing complete."

log "Completed successfully."