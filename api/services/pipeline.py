"""
Training pipeline service.

Two execution modes selected at startup:
  ECS mode  — when ECS_CLUSTER is set in .env.
              Calls ecs.run_task(), then polls ECS status and streams
              CloudWatch Logs back into the job's log list.
  Local mode — when ECS_CLUSTER is empty.
              Spawns subprocesses directly (dev / single-EC2 usage).
"""

import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from api.core.config import settings
from api.services.utils import parse_metrics, run_subprocess

_lock = threading.Lock()
_training_lock = threading.Lock()  # serialises local runs; unused in ECS mode

JOBS: dict[str, dict] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _sanitize(model_id: str) -> str:
    return model_id.replace("/", "__")


def _log(job_id: str, line: str) -> None:
    with _lock:
        JOBS[job_id]["logs"].append(line)


def _set_phase(job_id: str, phase: str) -> None:
    with _lock:
        JOBS[job_id]["status"] = phase
        JOBS[job_id]["logs"].append(f"[pipeline] ── {phase.upper()} ──")


def _complete(job_id: str, metrics: dict, save_key: str) -> None:
    with _lock:
        JOBS[job_id].update(
            {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics or None,
                "save_key": save_key,
            }
        )


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


def _detect_phase(line: str) -> Optional[str]:
    """Map a run_pipeline.sh log line to a pipeline phase name."""
    if "Step 1" in line:
        return "preprocessing"
    if "Step 2" in line:
        return "training"
    if "Step 3" in line:
        return "validating"
    if "Step 4" in line:
        return "testing"
    return None


def _fetch_logs(
    logs_client,
    log_group: str,
    log_stream: str,
    next_token: Optional[str],
) -> tuple[list[str], Optional[str]]:
    """Pull the next batch of CloudWatch log events. Returns (lines, next_token)."""
    kwargs: dict = {"logGroupName": log_group, "logStreamName": log_stream}
    if next_token is None:
        kwargs["startFromHead"] = True
    else:
        kwargs["nextToken"] = next_token
    resp = logs_client.get_log_events(**kwargs)
    lines = [e["message"].rstrip() for e in resp.get("events", []) if e.get("message", "").strip()]
    return lines, resp.get("nextForwardToken")


# ── ECS worker ────────────────────────────────────────────────────────────────


def _ecs_worker(
    job_id: str,
    model_id: str,
    bucket: Optional[str],
    mode: str,
) -> None:
    import boto3

    save_key = _sanitize(model_id)
    ecs = boto3.client("ecs", region_name=settings.aws_region)
    logs = boto3.client("logs", region_name=settings.aws_region)

    subnets = [s.strip() for s in settings.ecs_subnet_ids.split(",") if s.strip()]
    sgs = [s.strip() for s in settings.ecs_security_group_ids.split(",") if s.strip()]

    # Environment variables forwarded to run_pipeline.sh inside the container
    env = [
        {"name": "MODEL",            "value": model_id},
        {"name": "BUCKET",           "value": bucket or ""},
        {"name": "LOCAL",            "value": "false"},
        {"name": "PREPROCESS_ONLY",  "value": "true" if mode == "preprocess_only" else "false"},
        {"name": "SKIP_TRAIN",       "value": "true" if mode == "skip_train" else "false"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]

    # ── Launch ECS task ───────────────────────────────────────────────────────
    try:
        resp = ecs.run_task(
            cluster=settings.ecs_cluster,
            taskDefinition=settings.ecs_task_definition,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": subnets,
                    "securityGroups": sgs,
                    "assignPublicIp": "ENABLED",
                }
            },
            overrides={
                "containerOverrides": [{
                    "name": settings.ecs_container_name,
                    "environment": env,
                }]
            },
        )
    except Exception as exc:
        _fail(job_id, f"Failed to launch ECS task: {exc}")
        return

    failures = resp.get("failures", [])
    if not resp.get("tasks") or failures:
        reason = failures[0]["reason"] if failures else "No tasks returned"
        _fail(job_id, f"ECS run_task rejected: {reason}")
        return

    task_arn    = resp["tasks"][0]["taskArn"]
    task_id     = task_arn.split("/")[-1]
    log_stream  = f"{settings.log_stream_prefix}/{settings.ecs_container_name}/{task_id}"

    _log(job_id, f"[ecs] Task launched  : {task_id}")
    _log(job_id, f"[ecs] Cluster        : {settings.ecs_cluster}")
    _log(job_id, f"[ecs] Log stream     : {settings.log_group}/{log_stream}")

    # Set initial phase based on mode
    initial_phase = "validating" if mode == "skip_train" else "preprocessing"
    _set_phase(job_id, initial_phase)

    # ── Poll ECS + CloudWatch until the task stops ────────────────────────────
    log_token:  Optional[str] = None
    task_desc:  dict          = {}

    while True:
        time.sleep(5)

        # Task status
        try:
            task_desc = ecs.describe_tasks(
                cluster=settings.ecs_cluster, tasks=[task_arn]
            )["tasks"][0]
            task_status = task_desc["lastStatus"]
        except Exception as exc:
            _log(job_id, f"[ecs] Warning: describe_tasks failed: {exc}")
            continue

        # CloudWatch log drain
        try:
            new_lines, log_token = _fetch_logs(logs, settings.log_group, log_stream, log_token)
            for line in new_lines:
                _log(job_id, line)
                phase = _detect_phase(line)
                if phase:
                    with _lock:
                        JOBS[job_id]["status"] = phase
        except logs.exceptions.ResourceNotFoundException:
            pass  # log stream not created yet while task is PENDING/PROVISIONING
        except Exception as exc:
            _log(job_id, f"[ecs] Warning: CloudWatch fetch failed: {exc}")

        if task_status == "STOPPED":
            # Final log drain
            try:
                final_lines, _ = _fetch_logs(logs, settings.log_group, log_stream, log_token)
                for line in final_lines:
                    _log(job_id, line)
            except Exception:
                final_lines = []

            exit_code = task_desc.get("containers", [{}])[0].get("exitCode", 1)
            if exit_code == 0:
                with _lock:
                    all_logs = list(JOBS[job_id]["logs"])
                _complete(job_id, parse_metrics(all_logs), save_key)
                _log(job_id, "[ecs] ✓ Task completed successfully.")
            else:
                reason = task_desc.get("stoppedReason", f"Exit code {exit_code}")
                _fail(job_id, f"ECS task stopped with error: {reason} (exit {exit_code})")
            break


# ── Local subprocess worker ───────────────────────────────────────────────────


def _local_worker(
    job_id: str,
    model_id: str,
    bucket: Optional[str],
    local: bool,
    mode: str,
) -> None:
    python = sys.executable
    save_key = _sanitize(model_id)
    model_save_path = f"saved_models/{save_key}"

    local_flags  = ["--local"] if local else []
    bucket_flags = ["--bucket", bucket] if bucket else []

    def _exec(cmd: list[str]) -> tuple[int, list[str]]:
        return run_subprocess(cmd, lambda line: _log(job_id, line))

    with _training_lock:
        try:
            if mode == "preprocess_only":
                _set_phase(job_id, "preprocessing")
                rc, _ = _exec(
                    [python, "preprocess.py", "--file_name", "bias_clean.csv"]
                    + local_flags + bucket_flags
                )
                if rc != 0:
                    _fail(job_id, "Preprocessing failed — see logs above.")
                    return
                _complete(job_id, {}, save_key)
                _log(job_id, "[pipeline] ✓ COMPLETED — preprocessed data ready.")
                return

            if mode == "skip_train":
                _set_phase(job_id, "validating")
                rc, _ = _exec(
                    [python, "validate.py", "--model", model_id, "--model-path", model_save_path]
                    + local_flags
                )
                if rc != 0:
                    _fail(job_id, "Validation failed — see logs above.")
                    return

                _set_phase(job_id, "testing")
                rc, test_lines = _exec(
                    [python, "test.py", "--model", model_id, "--model-path", model_save_path]
                    + local_flags
                )
                if rc != 0:
                    _fail(job_id, "Testing failed — see logs above.")
                    return
                _complete(job_id, parse_metrics(test_lines), save_key)
                _log(job_id, "[pipeline] ✓ COMPLETED — evaluate-only run finished.")
                return

            # Full pipeline
            _set_phase(job_id, "preprocessing")
            rc, _ = _exec(
                [python, "preprocess.py", "--file_name", "bias_clean.csv"]
                + local_flags + bucket_flags
            )
            if rc != 0:
                _fail(job_id, "Preprocessing failed — see logs above.")
                return

            _set_phase(job_id, "training")
            rc, _ = _exec([python, "train.py", "--model", model_id] + local_flags)
            if rc != 0:
                _fail(job_id, "Training failed — see logs above.")
                return

            _set_phase(job_id, "validating")
            rc, _ = _exec(
                [python, "validate.py", "--model", model_id, "--model-path", model_save_path]
                + local_flags
            )
            if rc != 0:
                _fail(job_id, "Validation failed — see logs above.")
                return

            _set_phase(job_id, "testing")
            rc, test_lines = _exec(
                [python, "test.py", "--model", model_id, "--model-path", model_save_path]
                + local_flags
            )
            if rc != 0:
                _fail(job_id, "Testing failed — see logs above.")
                return

            _complete(job_id, parse_metrics(test_lines), save_key)
            _log(job_id, f"[pipeline] ✓ COMPLETED — model saved to {model_save_path}")

        except Exception as exc:
            _fail(job_id, f"Unexpected error: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────


def _use_ecs() -> bool:
    return bool(settings.ecs_cluster)


def start_pipeline(model_id: str, bucket: Optional[str], local: bool, mode: str = "full") -> str:
    job_id   = str(uuid.uuid4())[:8]
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

    if _use_ecs():
        _log(job_id, f"[pipeline] Execution mode: ECS (cluster={settings.ecs_cluster})")
        target = _ecs_worker
        args   = (job_id, model_id, bucket or settings.bucket or None, mode)
    else:
        _log(job_id, "[pipeline] Execution mode: local subprocess")
        target = _local_worker
        args   = (job_id, model_id, bucket or settings.bucket or None, local, mode)

    threading.Thread(target=target, args=args, daemon=True, name=f"pipeline-{job_id}").start()
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def list_jobs() -> list[dict]:
    with _lock:
        return sorted(JOBS.values(), key=lambda j: j["started_at"], reverse=True)
