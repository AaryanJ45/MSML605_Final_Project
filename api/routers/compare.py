from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from api.services import compare as svc

router = APIRouter(prefix="/compare", tags=["compare"])

VALID_MODELS = {"bert", "distilbert"}


class CompareRequest(BaseModel):
    model_a: str = "bert"
    model_b: str = "distilbert"
    local: bool = True

    @field_validator("model_a", "model_b")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in VALID_MODELS:
            raise ValueError(f"model must be one of {sorted(VALID_MODELS)}")
        return v


@router.post("/run")
async def run_compare(req: CompareRequest) -> dict:
    if req.model_a == req.model_b:
        raise HTTPException(status_code=400, detail="model_a and model_b must be different")
    job_id = svc.start_compare(
        model_a=req.model_a,
        model_b=req.model_b,
        local=req.local,
    )
    return {"job_id": job_id, "status": "queued"}


@router.get("")
async def list_compare_jobs() -> list:
    return svc.list_compare_jobs()


@router.get("/{job_id}")
async def get_compare_job(job_id: str) -> dict:
    job = svc.get_compare_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Compare job '{job_id}' not found")
    return job
