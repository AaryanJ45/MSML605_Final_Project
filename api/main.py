import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.predict import router as predict_router
from api.core.config import settings
from api.services import inference as svc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the default model (bert) on startup so the first request is fast.
    # If no local model exists and BUCKET is configured, it will be downloaded here.
    try:
        logger.info("Pre-loading bert model...")
        svc.load_model(
            model_key="bert",
            model_save_dir=settings.model_save_dir,
            label_encoder_path=settings.label_encoder_path,
            bucket=settings.bucket or None,
        )
        logger.info("bert model ready.")
    except Exception as e:
        logger.warning("Could not pre-load model on startup: %s", e)
    yield


app = FastAPI(
    title="Bias Detector API",
    description="Classify news article text as left / center / right using BERT or DistilBERT.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
