import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

# In-memory job store. On EC2 this lives for the process lifetime — good enough for Option A.
_lock = threading.Lock()
_training_lock = threading.Lock()  # serialises pipeline runs so they don't fight for GPU/disk

JOBS: dict[str, dict] = {}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _sanitize(model_id: str) -> str:
    """Turn a HF model ID into a safe filesystem key: microsoft/deberta → microsoft__deberta."""
    return model_id.replace("/", "__")


def _log(job_id: str, line: str) -> None:
    with _lock:
        JOBS[job_id]["logs"].append(line)


def _set_phase(job_id: str, phase: str) -> None:
    with _lock:
        JOBS[job_id]["status"] = phase
        JOBS[job_id]["logs"].append(f"[pipeline] ── {phase.upper()} ──")


def _fail(job_id: str, reason: str) -> None:
    with _lock:
        JOBS[job_id].update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": reason,
            }
        )
    _log(job_id, f"[pipeline] ✗ FAILED: {reason}")


def _run(cmd: list[str], job_id: str) -> int:
    """Run a subprocess, streaming every output line into the job log."""
    _log(job_id, f"[cmd] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        stripped = line.rstrip()
        if stripped:
            _log(job_id, stripped)
    proc.stdout.close()
    proc.wait()
    return proc.returncode


def _parse_metrics(logs: list[str]) -> dict[str, float]:
    """Scrape test-phase metric lines from captured logs."""
    metrics: dict[str, float] = {}
    patterns = {
        "accuracy":  r"Accuracy\s*:\s*([0-9.]+)",
        "precision": r"Precision\s*:\s*([0-9.]+)",
        "recall":    r"Recall\s*:\s*([0-9.]+)",
        "f1":        r"F1 Score\s*:\s*([0-9.]+)",
    }
    for line in logs:
        for key, pattern in patterns.items():
            if key not in metrics:
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    metrics[key] = float(m.group(1))
    return metrics


# ── Worker ────────────────────────────────────────────────────────────────────

def _pipeline_worker(
    job_id: str,
    model_id: str,
    bucket: Optional[str],
    local: bool,
) -> None:
    python = sys.executable
    save_key = _sanitize(model_id)
    model_save_path = f"saved_models/{save_key}"

    local_flags = ["--local"] if local else []
    bucket_flags = ["--bucket", bucket] if bucket else []

    # Serialise: only one training job runs at a time on this server.
    with _training_lock:
        try:
            # ── Step 1: Preprocess ────────────────────────────────────────
            _set_phase(job_id, "preprocessing")
            rc = _run(
                [python, "preprocess.py", "--file_name", "bias_clean.csv"]
                + local_flags
                + bucket_flags,
                job_id,
            )
            if rc != 0:
                _fail(job_id, "Preprocessing failed — see logs above.")
                return

            # ── Step 2: Train ─────────────────────────────────────────────
            _set_phase(job_id, "training")
            rc = _run(
                [python, "train.py", "--model", model_id] + local_flags,
                job_id,
            )
            if rc != 0:
                _fail(job_id, "Training failed — see logs above.")
                return

            # ── Step 3: Validate ──────────────────────────────────────────
            _set_phase(job_id, "validating")
            rc = _run(
                [python, "validate.py", "--model", model_id, "--model-path", model_save_path]
                + local_flags,
                job_id,
            )
            if rc != 0:
                _fail(job_id, "Validation failed — see logs above.")
                return

            # ── Step 4: Test ──────────────────────────────────────────────
            _set_phase(job_id, "testing")
            rc = _run(
                [python, "test.py", "--model", model_id, "--model-path", model_save_path]
                + local_flags,
                job_id,
            )
            if rc != 0:
                _fail(job_id, "Testing failed — see logs above.")
                return

            # ── Done ──────────────────────────────────────────────────────
            with _lock:
                logs_snapshot = list(JOBS[job_id]["logs"])
            metrics = _parse_metrics(logs_snapshot)
            with _lock:
                JOBS[job_id].update(
                    {
                        "status": "completed",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "metrics": metrics,
                        "save_key": save_key,
                    }
                )
            _log(job_id, f"[pipeline] ✓ COMPLETED — model saved to {model_save_path}")

        except Exception as exc:
            _fail(job_id, f"Unexpected error: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────

def start_pipeline(model_id: str, bucket: Optional[str], local: bool) -> str:
    job_id = str(uuid.uuid4())[:8]
    save_key = _sanitize(model_id)
    with _lock:
        JOBS[job_id] = {
            "job_id": job_id,
            "model_id": model_id,
            "save_key": save_key,
            "status": "queued",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "logs": [f"[pipeline] Job {job_id} queued for model: {model_id}"],
            "metrics": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_pipeline_worker,
        args=(job_id, model_id, bucket, local),
        daemon=True,
        name=f"pipeline-{job_id}",
    )
    thread.start()
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def list_jobs() -> list[dict]:
    with _lock:
        return sorted(JOBS.values(), key=lambda j: j["started_at"], reverse=True)
