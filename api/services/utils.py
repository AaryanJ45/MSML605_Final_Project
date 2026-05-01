import re
import subprocess
from typing import Callable


def run_subprocess(
    cmd: list[str],
    log_fn: Callable[[str], None],
    cmd_tag: str = "cmd",
) -> tuple[int, list[str]]:
    """Run a subprocess, stream each output line through log_fn, and return (returncode, lines).

    cmd_tag labels the command-echo line: [cmd_tag] python train.py ...
    The caller controls how individual output lines are prefixed by wrapping log_fn.
    """
    log_fn(f"[{cmd_tag}] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for raw in iter(proc.stdout.readline, ""):
        stripped = raw.rstrip()
        if stripped:
            log_fn(stripped)
            lines.append(stripped)
    proc.stdout.close()
    proc.wait()
    return proc.returncode, lines


def parse_metrics(lines: list[str]) -> dict[str, float]:
    """Scrape aggregate metric values from captured log/output lines."""
    metrics: dict[str, float] = {}
    patterns = {
        "accuracy":  r"Accuracy\s*:\s*([0-9.]+)",
        "precision": r"Precision\s*:\s*([0-9.]+)",
        "recall":    r"Recall\s*:\s*([0-9.]+)",
        "f1":        r"F1 Score\s*:\s*([0-9.]+)",
    }
    for line in lines:
        for key, pattern in patterns.items():
            if key not in metrics:
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    metrics[key] = float(m.group(1))
    return metrics


def parse_per_class(
    lines: list[str],
    classes: list[str],
) -> dict[str, dict[str, float]]:
    """Parse sklearn classification_report lines into per-class precision/recall/f1."""
    per_class: dict[str, dict[str, float]] = {}
    for cls in classes:
        for line in lines:
            m = re.match(
                rf"^\s*{re.escape(cls)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
                line,
                re.IGNORECASE,
            )
            if m:
                per_class[cls] = {
                    "precision": float(m.group(1)),
                    "recall":    float(m.group(2)),
                    "f1":        float(m.group(3)),
                }
                break
    return per_class
