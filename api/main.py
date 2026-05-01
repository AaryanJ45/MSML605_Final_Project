import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from api.routers.predict import router as predict_router
from api.routers.jobs import router as jobs_router
from api.routers.compare import router as compare_router
from api.core.config import settings
from api.services import inference as svc

_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")

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

_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # Add your EC2 public IP/domain here, e.g.:
    # "http://<EC2-PUBLIC-IP>:5173",
    # "http://<YOUR-DOMAIN>",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(jobs_router)
app.include_router(compare_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ── Serve the built React app (production) ────────────────────────────────────
# When frontend/dist exists (i.e. after `npm run build`), FastAPI serves the
# static assets and returns index.html for every unmatched path so that
# React Router handles client-side navigation.
if os.path.isdir(_FRONTEND_DIST):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")),
        name="static-assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(_: str) -> FileResponse:
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))
