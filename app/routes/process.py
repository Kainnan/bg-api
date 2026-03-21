"""Endpoints multipart: imagem única ou ZIP → resposta em stream."""

import io
import logging
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.services.image_service import (
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