import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from api.services.utils import parse_metrics, parse_per_class, run_subprocess

_lock = threading.Lock()
_compare_lock = threading.Lock()

COMPARE_JOBS: dict[str, dict] = {}

CLASSES = ["center", "left", "right"]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _log(job_id: str, line: str) -> None:
    with _lock:
        COMPARE_JOBS[job_id]["logs"].append(line)


def _fail(job_id: str, reason: str) -> None:
    with _lock:
        COMPARE_JOBS[job_id].update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": reason,
            }
        )
    _log(job_id, f"[compare] ✗ FAILED: {reason}")


def _exec(job_id: str, cmd: list[str], model_tag: str) -> tuple[int, list[str]]:
    """Run a subprocess, prefix each output line with the model tag, return (rc, lines)."""
    return run_subprocess(
        cmd,
        lambda line: _log(job_id, f"[{model_tag}] {line}"),
        f"cmd:{model_tag}",
    )


# ── Worker ────────────────────────────────────────────────────────────────────


def _compare_worker(job_id: str, model_a: str, model_b: str, local: bool) -> None:
    python = sys.executable
    local_flags = ["--local"] if local else []

    with _compare_lock:
        try:
            # ── Model A ──────────────────────────────────────────────────────
            with _lock:
                COMPARE_JOBS[job_id]["status"] = "evaluating_a"
            _log(job_id, f"[compare] ── Evaluating {model_a.upper()} ──")
            rc_a, lines_a = _exec(
                job_id,
                [python, "test.py", "--model", model_a, "--model-path", f"saved_models/{model_a}"]
                + local_flags,
                model_a,
            )
            if rc_a != 0:
                _fail(job_id, f"Evaluation of '{model_a}' failed — check that the model has been trained first.")
                return

            # ── Model B ──────────────────────────────────────────────────────
            with _lock:
                COMPARE_JOBS[job_id]["status"] = "evaluating_b"
            _log(job_id, f"[compare] ── Evaluating {model_b.upper()} ──")
            rc_b, lines_b = _exec(
                job_id,
                [python, "test.py", "--model", model_b, "--model-path", f"saved_models/{model_b}"]
                + local_flags,
                model_b,
            )
            if rc_b != 0:
                _fail(job_id, f"Evaluation of '{model_b}' failed — check that the model has been trained first.")
                return

            # ── Done ─────────────────────────────────────────────────────────
            with _lock:
                COMPARE_JOBS[job_id].update(
                    {
                        "status": "completed",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "results": {
                            model_a: {
                                "overall":   parse_metrics(lines_a),
                                "per_class": parse_per_class(lines_a, CLASSES),
                            },
                            model_b: {
                                "overall":   parse_metrics(lines_b),
                                "per_class": parse_per_class(lines_b, CLASSES),
                            },
                        },
                    }
                )
            _log(job_id, "[compare] ✓ COMPLETED")

        except Exception as exc:
            _fail(job_id, f"Unexpected error: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────


def start_compare(model_a: str, model_b: str, local: bool) -> str:
    job_id = str(uuid.uuid4())[:8]
    with _lock:
        COMPARE_JOBS[job_id] = {
            "job_id":      job_id,
            "model_a":     model_a,
            "model_b":     model_b,
            "status":      "queued",
            "started_at":  datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "logs":        [f"[compare] Job {job_id} queued — comparing {model_a} vs {model_b}"],
            "results":     None,
            "error":       None,
        }

    thread = threading.Thread(
        target=_compare_worker,
        args=(job_id, model_a, model_b, local),
        daemon=True,
        name=f"compare-{job_id}",
    )
    thread.start()
    return job_id


def get_compare_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = COMPARE_JOBS.get(job_id)
        return dict(job) if job else None


def list_compare_jobs() -> list[dict]:
    with _lock:
        return sorted(COMPARE_JOBS.values(), key=lambda j: j["started_at"], reverse=True)
