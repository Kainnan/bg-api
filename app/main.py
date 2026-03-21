"""
Aplicação FastAPI: processamento de imagens e ZIP em memória (sem pasta `storage/`).
Execute: `uvicorn app.main:app --reload` ou `python app/main.py`.
"""
import logging
import logging.config
import os
from pathlib import Path
import sys

# `python app/main.py`: garante que a raiz do projeto esteja em sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging — configurado antes de qualquer import do app para capturar tudo.
# Nível via env: LOG_LEVEL=DEBUG | INFO | WARNING  (default: INFO)
# ---------------------------------------------------------------------------
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": _LOG_LEVEL,
        "handlers": ["console"],
    },
    # silencia loggers muito verbosos de libs
    "loggers": {
        "uvicorn.access": {"level": "INFO"},
        "PIL": {"level": "WARNING"},
        "onnxruntime": {"level": "WARNING"},
    },
})

logger = logging.getLogger("bg_api.main")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routes.process import router as process_router

app = FastAPI(
    title="bg-api",
    description="Processa imagens e ZIPs de assets; respostas são streams (WebP / ZIP), sem persistir no disco do projeto.",
    version="1.0.0",
)

app.include_router(process_router, tags=["processamento"])


@app.on_event("startup")
async def _on_startup():
    logger.info("bg-api iniciado — LOG_LEVEL=%s", _LOG_LEVEL)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    import time
    logger.info("→ %s %s", request.method, request.url.path)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception("✗ %s %s — erro não tratado (%.2fs)", request.method, request.url.path, elapsed)
        return JSONResponse(status_code=500, content={"detail": "erro interno"})
    elapsed = time.perf_counter() - t0
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(level, "← %s %s %d (%.2fs)", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.get("/health", tags=["sistema"])
def health():
    from app.services.image_service import _REMBG_AVAILABLE
    return {"status": "ok", "rembg": _REMBG_AVAILABLE}


if __name__ == "__main__":
    # Permite rodar com `python3 app/main.py` também.
    import os

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("BG_API_RELOAD", "1") == "1"

    # Usa string de import para habilitar `reload` sem warning.
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)

