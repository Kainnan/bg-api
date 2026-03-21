"""Endpoints multipart: imagem única ou ZIP → resposta em stream."""

import io
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.services.image_service import (
    compress_to_webp_only,
    is_background_asset,
    remove_light_background,
)
from app.services.zip_service import process_zip_bytes_to_output_zip_bytes

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

    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="tipo de arquivo não suportado")

    if threshold < 0 or threshold > 255:
        raise HTTPException(status_code=400, detail="threshold deve estar entre 0 e 255")

    raw = await file.read()
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="arquivo vazio")

    suffix = Path(file.filename or "input.png").suffix or ".png"
    with tempfile.TemporaryDirectory(prefix="bgapi_img_") as tmp:
        in_path = Path(tmp) / f"input{suffix}"
        out_path = Path(tmp) / "output.webp"
        in_path.write_bytes(raw)

        if is_background_asset(file.filename or ""):
            compress_to_webp_only(str(in_path), output_path=str(out_path))
        else:
            remove_light_background(str(in_path), threshold=threshold, output_path=str(out_path))

        output_bytes = out_path.read_bytes()

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
    if threshold < 0 or threshold > 255:
        raise HTTPException(status_code=400, detail="threshold deve estar entre 0 e 255")

    zip_bytes = await file.read()
    await file.close()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="zip vazio")

    output_zip_bytes = process_zip_bytes_to_output_zip_bytes(zip_bytes, threshold=threshold)

    return StreamingResponse(
        io.BytesIO(output_zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="images.zip"',
        },
    )