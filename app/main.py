"""
Aplicação FastAPI: processamento de imagens e ZIP em memória (sem pasta `storage/`).
Execute: `uvicorn app.main:app --reload` ou `python app/main.py`.
"""
from pathlib import Path
import sys

# `python app/main.py`: garante que a raiz do projeto esteja em sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI

from app.routes.process import router as process_router

app = FastAPI(
    title="bg-api",
    description="Processa imagens e ZIPs de assets; respostas são streams (WebP / ZIP), sem persistir no disco do projeto.",
    version="1.0.0",
)

app.include_router(process_router, tags=["processamento"])


@app.get("/health", tags=["sistema"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # Permite rodar com `python3 app/main.py` também.
    import os

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("BG_API_RELOAD", "1") == "1"

    # Usa string de import para habilitar `reload` sem warning.
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)

