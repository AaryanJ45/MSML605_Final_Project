import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from api.services.utils import parse_metrics, run_subprocess

# In-memory job store. Lives for the process lifetime — good enough for a single-server deployment.
_lock = threading.Lock()
_training_lock = threading.Lock()  # serialises pipeline runs so they don't fight for GPU/disk

JOBS: dict[str, dict] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────


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


def _exec(job_id: str, cmd: list[str]) -> tuple[int, list[str]]:
    """Run a subprocess, stream every output line into the job log, return (rc, lines)."""
    return run_subprocess(cmd, lambda line: _log(job_id, line))


# ── Worker ────────────────────────────────────────────────────────────────────


def _pipeline_worker(
    job_id: str,
    model_id: str,
    bucket: Optional[str],
    local: bool,
    mode: str,
) -> None:
    python = sys.executable
    save_key = _sanitize(model_id)
    model_save_path = f"saved_models/{save_key}"

    local_flags = ["--local"] if local else []
    bucket_flags = ["--bucket", bucket] if bucket else []

    with _training_lock:
        try:
            if mode == "preprocess_only":
                _set_phase(job_id, "preprocessing")
                rc, _ = _exec(
                    job_id,
                    [python, "preprocess.py", "--file_name", "bias_clean.csv"]
                    + local_flags
                    + bucket_flags,
                )
                if rc != 0:
                    _fail(job_id, "Preprocessing failed — see logs above.")
                    return
                with _lock:
                    JOBS[job_id].update(
                        {
                            "status": "completed",
                            "finished_at": datetime.now(timezone.utc).isoformat(),
                            "metrics": None,
                            "save_key": save_key,
                        }
                    )
                _log(job_id, "[pipeline] ✓ COMPLETED — preprocessed data uploaded.")
                return

            if mode == "skip_train":
                _set_phase(job_id, "validating")
                rc, _ = _exec(
                    job_id,
                    [python, "validate.py", "--model", model_id, "--model-path", model_save_path]
                    + local_flags,
                )
                if rc != 0:
                    _fail(job_id, "Validation failed — see logs above.")
                    return

                _set_phase(job_id, "testing")
                rc, test_lines = _exec(
                    job_id,
                    [python, "test.py", "--model", model_id, "--model-path", model_save_path]
                    + local_flags,
                )
                if rc != 0:
                    _fail(job_id, "Testing failed — see logs above.")
                    return

                with _lock:
                    JOBS[job_id].update(
                        {
                            "status": "completed",
                            "finished_at": datetime.now(timezone.utc).isoformat(),
                            "metrics": parse_metrics(test_lines),
                            "save_key": save_key,
                        }
                    )
                _log(job_id, "[pipeline] ✓ COMPLETED — evaluate-only run finished.")
                return

            # ── Full pipeline ─────────────────────────────────────────────────

            _set_phase(job_id, "preprocessing")
            rc, _ = _exec(
                job_id,
                [python, "preprocess.py", "--file_name", "bias_clean.csv"]
                + local_flags
                + bucket_flags,
            )
            if rc != 0:
                _fail(job_id, "Preprocessing failed — see logs above.")
                return

            _set_phase(job_id, "training")
            rc, _ = _exec(job_id, [python, "train.py", "--model", model_id] + local_flags)
            if rc != 0:
                _fail(job_id, "Training failed — see logs above.")
                return

            _set_phase(job_id, "validating")
            rc, _ = _exec(
                job_id,
                [python, "validate.py", "--model", model_id, "--model-path", model_save_path]
                + local_flags,
            )
            if rc != 0:
                _fail(job_id, "Validation failed — see logs above.")
                return

            _set_phase(job_id, "testing")
            rc, test_lines = _exec(
                job_id,
                [python, "test.py", "--model", model_id, "--model-path", model_save_path]
                + local_flags,
            )
            if rc != 0:
                _fail(job_id, "Testing failed — see logs above.")
                return

            with _lock:
                JOBS[job_id].update(
                    {
                        "status": "completed",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "metrics": parse_metrics(test_lines),
                        "save_key": save_key,
                    }
                )
            _log(job_id, f"[pipeline] ✓ COMPLETED — model saved to {model_save_path}")

        except Exception as exc:
            _fail(job_id, f"Unexpected error: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────


def start_pipeline(model_id: str, bucket: Optional[str], local: bool, mode: str = "full") -> str:
    job_id = str(uuid.uuid4())[:8]
    save_key = _sanitize(model_id)
    with _lock:
        JOBS[job_id] = {
            "job_id":      job_id,
            "model_id":    model_id,
            "save_key":    save_key,
            "mode":        mode,
            "status":      "queued",
            "started_at":  datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "logs":        [f"[pipeline] Job {job_id} queued — model: {model_id}, mode: {mode}"],
            "metrics":     None,
            "error":       None,
        }

    thread = threading.Thread(
        target=_pipeline_worker,
        args=(job_id, model_id, bucket, local, mode),
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
