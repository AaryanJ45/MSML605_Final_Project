from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from api.services import pipeline as svc
from api.core.config import settings

router = APIRouter(prefix="/jobs", tags=["jobs"])


class RunRequest(BaseModel):
    model_id: str
    local: bool = True
    mode: str = "full"  # "full" | "preprocess_only" | "skip_train"

    @field_validator("model_id")
    @classmethod
    def model_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_id must not be empty")
        return v.strip()

    @field_validator("mode")
    @classmethod
    def mode_valid(cls, v: str) -> str:
        allowed = {"full", "preprocess_only", "skip_train"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}")
        return v


@router.post("/run")
async def run_pipeline(req: RunRequest) -> dict:
    job_id = svc.start_pipeline(
        model_id=req.model_id,
        bucket=settings.bucket or None,
        local=req.local,
        mode=req.mode,
    )
    return {"job_id": job_id, "status": "queued"}


@router.get("")
async def list_jobs() -> list:
    return svc.list_jobs()


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    job = svc.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job
