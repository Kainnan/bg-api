"""Endpoints multipart: imagem única ou ZIP → resposta em stream."""

import io
import logging
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from app.services.image_service import (
    _BRIA_MODEL,
    _bria_remove,
    compress_to_webp_only,
    is_background_asset,
    remove_light_background,
)
from app.services.zip_service import process_zip_bytes_to_output_zip_bytes

logger = logging.getLogger("bg_api.routes")
router = APIRouter()


@router.post(
    "/process-image",
    summary="Processar uma imagem",
    response_description="Arquivo WebP",
)
async def process_image(
    file: UploadFile = File(...),
    threshold: int = Form(245),
):
    allowed_types = ["image/png", "image/jpeg", "image/webp"]
    filename = file.filename or "input.png"

    logger.info("process-image | arquivo=%s content_type=%s threshold=%d", filename, file.content_type, threshold)

    if file.content_type not in allowed_types:
        logger.warning("process-image | tipo não suportado: %s", file.content_type)
        raise HTTPException(status_code=400, detail="tipo de arquivo não suportado")

    if threshold < 0 or threshold > 255:
        raise HTTPException(status_code=400, detail="threshold deve estar entre 0 e 255")

    raw = await file.read()
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="arquivo vazio")

    logger.debug("process-image | tamanho lido: %.1f KB", len(raw) / 1024)

    suffix = Path(filename).suffix or ".png"
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="bgapi_img_") as tmp:
        in_path = Path(tmp) / f"input{suffix}"
        out_path = Path(tmp) / "output.webp"
        in_path.write_bytes(raw)

        is_bg = is_background_asset(filename)
        if is_bg:
            logger.info("process-image | %s → background asset (só compressão WebP)", filename)
            compress_to_webp_only(str(in_path), output_path=str(out_path))
        else:
            logger.info("process-image | %s → remoção de fundo (threshold=%d)", filename, threshold)
            remove_light_background(str(in_path), threshold=threshold, output_path=str(out_path))

        output_bytes = out_path.read_bytes()

    elapsed = time.perf_counter() - t0
    logger.info(
        "process-image | concluído em %.2fs | entrada=%.1fKB saída=%.1fKB",
        elapsed, len(raw) / 1024, len(output_bytes) / 1024,
    )

    return StreamingResponse(
        io.BytesIO(output_bytes),
        media_type="image/webp",
        headers={"Content-Disposition": 'attachment; filename="image.webp"'},
    )


@router.post(
    "/process-zip",
    summary="Processar ZIP de imagens",
    response_description="ZIP com WebPs renomeados (images.zip)",
)
async def process_zip(
    file: UploadFile = File(...),
    threshold: int = Form(245),
):
    filename = file.filename or "upload.zip"
    logger.info("process-zip | arquivo=%s threshold=%d", filename, threshold)

    if threshold < 0 or threshold > 255:
        raise HTTPException(status_code=400, detail="threshold deve estar entre 0 e 255")

    zip_bytes = await file.read()
    await file.close()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="zip vazio")

    logger.info("process-zip | ZIP recebido: %.1f KB", len(zip_bytes) / 1024)

    t0 = time.perf_counter()
    output_zip_bytes = process_zip_bytes_to_output_zip_bytes(zip_bytes, threshold=threshold)
    elapsed = time.perf_counter() - t0

    logger.info(
        "process-zip | concluído em %.2fs | entrada=%.1fKB saída=%.1fKB",
        elapsed, len(zip_bytes) / 1024, len(output_zip_bytes) / 1024,
    )

    return StreamingResponse(
        io.BytesIO(output_zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="images.zip"',
        },
    )


@router.post(
    "/process-frame",
    summary="Remover fundo de UMA frame de vídeo (RGB → RGBA PNG)",
    response_description="PNG RGBA lossless, mesma dimensão da entrada",
)
async def process_frame(file: UploadFile = File(...)):
    """
    Endpoint dedicado a frames vindos de pipelines de video diffusion (motion-api,
    Runway, etc).

    Diferenças vs `/process-image`:
      • IGNORA heurísticas de filename (`is_frame_asset`, `is_background_asset`).
        Necessário porque frames de vídeo costumam ter nomes tipo `frame_00001.png`
        que ativariam o pipeline de slot-frame (componentes conectados) por engano.
      • SEMPRE usa BiRefNet — sem fallback de threshold (qualidade consistente).
      • Retorna PNG lossless (compress_level=1, rápido) em vez de WebP lossy,
        porque os frames serão re-encodados em vídeo (WebM/MP4) logo depois e
        cada camada de compressão lossy degrada o resultado final.
      • Sem parâmetros — input simples, contrato direto.

    Espera-se que o input seja RGB (ou RGBA com fundo opaco) — diffusion models
    flatten alpha em fundo neutro antes de gerar.
    """
    if _BRIA_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="BiRefNet não está carregado — necessário para /process-frame",
        )

    raw = await file.read()
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="arquivo vazio")

    t0 = time.perf_counter()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"imagem inválida: {exc}")

    logger.info(
        "process-frame | %s %dx%d modo=%s",
        file.filename or "frame.png", image.width, image.height, image.mode,
    )

    try:
        rgba = _bria_remove(image)
    except Exception as exc:
        logger.exception("process-frame | BiRefNet falhou")
        raise HTTPException(status_code=500, detail=f"falha na segmentação: {exc}")

    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")

    out_buf = io.BytesIO()
    # compress_level=1: ~2-3x mais rápido que default (6), só ~10-15% maior em bytes.
    # Lossless de qualquer forma — a próxima etapa (WebM encode) recomprime.
    rgba.save(out_buf, format="PNG", optimize=False, compress_level=1)
    out_buf.seek(0)

    elapsed = time.perf_counter() - t0
    logger.info(
        "process-frame | concluído em %.2fs | entrada=%.1fKB saída=%.1fKB",
        elapsed, len(raw) / 1024, out_buf.getbuffer().nbytes / 1024,
    )

    return StreamingResponse(
        out_buf,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="frame.png"'},
    )