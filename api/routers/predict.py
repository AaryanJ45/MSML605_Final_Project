from fastapi import APIRouter, HTTPException
from api.schemas.models import PredictRequest, PredictResponse
from api.services import inference as svc
from api.core.config import settings

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    if req.model not in ("bert", "distilbert"):
        raise HTTPException(status_code=400, detail="model must be 'bert' or 'distilbert'")

    try:
        svc.load_model(
            model_key=req.model,
            model_save_dir=settings.model_save_dir,
            label_encoder_path=settings.label_encoder_path,
            bucket=settings.bucket or None,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result = svc.predict(text=req.text, model_key=req.model)

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_used=req.model,
    )
