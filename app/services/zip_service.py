from __future__ import annotations

import io
import logging
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from fastapi import HTTPException

from app.services.image_service import (
    compress_to_webp_only,
    is_background_asset,
    is_frame_asset,
    is_logo_asset,
    remove_light_background,
)

logger = logging.getLogger("bg_api.zip_service")

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _is_within_directory(base_dir: Path, target_path: Path) -> bool:
    """Protege contra zip slip: arquivos extraídos devem ficar dentro de `base_dir`."""
    base_resolved = base_dir.resolve()
    target_resolved = target_path.resolve()
    return base_resolved == target_resolved or base_resolved in target_resolved.parents


def _should_skip_zip_member(member_path: Path) -> bool:
    """Ignora `__MACOSX` e arquivos `._*` (AppleDouble em zips do macOS)."""
    if "__MACOSX" in member_path.parts:
        return True
    if member_path.name.startswith("._"):
        return True
    return False


def _extract_zip_archive(zf: zipfile.ZipFile, target_root: Path) -> None:
    """Extrai membros de um ZipFile já aberto para `target_root` (validação zip slip)."""
    target_root.mkdir(parents=True, exist_ok=True)
    for member in zf.infolist():
        member_path = Path(member.filename)

        if member_path.is_absolute():
            raise HTTPException(status_code=400, detail="zip inválido")

        if ".." in member_path.parts:
            raise HTTPException(status_code=400, detail="zip inválido")

        if _should_skip_zip_member(member_path):
            continue

        target_path = target_root / member_path
        if not _is_within_directory(target_root, target_path):
            raise HTTPException(status_code=400, detail="zip inválido")

        if member.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member, "r") as src, target_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def extract_zip_bytes_to_dir(zip_bytes: bytes, dest_dir: Path) -> None:
    """Extrai bytes de um ZIP para `dest_dir` (sem gravar o .zip em disco)."""
    logger.debug("extract_zip_bytes_to_dir: extraindo %.1f KB → %s", len(zip_bytes) / 1024, dest_dir)
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            members = zf.infolist()
            logger.debug("extract_zip_bytes_to_dir: %d membros no ZIP", len(members))
            _extract_zip_archive(zf, dest_dir)
    except HTTPException:
        raise
    except zipfile.BadZipFile as exc:
        logger.error("extract_zip_bytes_to_dir: arquivo ZIP inválido — %s", exc)
        raise HTTPException(status_code=400, detail="arquivo zip inválido")
    except Exception as exc:
        logger.error("extract_zip_bytes_to_dir: erro inesperado — %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail="não foi possível processar o zip")


def list_images_in_folder(folder: Path) -> list[Path]:
    """Lista imagens recursivamente, ignorando lixo comum do macOS."""
    if not folder.exists():
        raise HTTPException(status_code=400, detail="pasta extraída não encontrada")

    images: list[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if "__MACOSX" in path.parts:
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            images.append(path.resolve())

    return sorted(images)


def _extract_symbol_bucket(path_str: str) -> str | None:
    p = path_str.lower()
    if "wild" in p:
        return "W"
    if "scatter" in p:
        return "H"
    if "high" in p:
        return "H"
    if "low" in p:
        return "L"
    return None


def _extract_index(path_str: str) -> str | None:
    p = path_str.lower()
    patterns = [
        r"(?:icon|symbol)?[_\-]?(?:low|high|scatter|wild)[_\-]?(\d+)",
        r"(?:low|high|scatter|wild)[_\-]?(\d+)",
        r"[_\-](\d+)(?:[_\-]|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, p)
        if m:
            return m.group(1)
    return None


def _mapped_output_name(original_path: Path, used_names: set[str]) -> str:
    """
    Retorna o caminho relativo de saída (podendo incluir subpasta).

    Regras de nomeação:
    - frame / grid  → frame.webp
    - background    → body-bg.webp
    - logo          → logo.webp
    - wild          → symbols/W.webp
    - high/scatter  → symbols/H<n>.webp
    - low           → symbols/L<n>.webp
    - outros        → <stem_sanitizado>.webp
    """
    path_str = str(original_path)

    # Frame (grid/moldura)
    if is_frame_asset(path_str):
        name = "frame.webp"
        if name not in used_names:
            used_names.add(name)
            return name
        # múltiplos frames: frame_2.webp, frame_3.webp …
        n = 2
        while True:
            candidate = f"frame_{n}.webp"
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
            n += 1

    # Background
    if is_background_asset(path_str):
        name = "body-bg.webp"
        if name not in used_names:
            used_names.add(name)
            return name
        n = 2
        while True:
            candidate = f"body-bg_{n}.webp"
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
            n += 1

    # Logo
    if is_logo_asset(path_str):
        name = "logo.webp"
        if name not in used_names:
            used_names.add(name)
            return name
        n = 2
        while True:
            candidate = f"logo_{n}.webp"
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
            n += 1

    # Símbolos e wild → pasta symbols/
    bucket = _extract_symbol_bucket(path_str)
    if bucket:
        idx = _extract_index(path_str)
        if bucket == "W":
            base = "symbols/W"
        elif idx:
            base = f"symbols/{bucket}{idx}"
        else:
            base = f"symbols/{bucket}"

        name = f"{base}.webp"
        if name not in used_names:
            used_names.add(name)
            return name
        n = 2
        while True:
            candidate = f"{base}_{n}.webp"
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
            n += 1

    # Fallback
    base = re.sub(r"[^a-zA-Z0-9_\-]+", "_", original_path.stem) or "asset"
    name = f"{base}.webp"
    if name not in used_names:
        used_names.add(name)
        return name
    n = 2
    while True:
        candidate = f"{base}_{n}.webp"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        n += 1


def process_zip_bytes_to_output_zip_bytes(zip_bytes: bytes, *, threshold: int = 245) -> bytes:
    """Processa um ZIP em diretório temporário e retorna o ZIP final em memória."""
    t_total = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="bgapi_") as tmp:
        root = Path(tmp)
        input_dir = root / "in"
        output_dir = root / "out"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        extract_zip_bytes_to_dir(zip_bytes, input_dir)
        images = list_images_in_folder(input_dir)

        if not images:
            logger.warning("process_zip: nenhuma imagem encontrada no ZIP")
            raise HTTPException(status_code=400, detail="nenhuma imagem encontrada no zip")

        logger.info("process_zip: %d imagens encontradas, iniciando processamento (threshold=%d)", len(images), threshold)

        used_names: set[str] = set()
        ok_count = 0
        err_count = 0

        for i, src in enumerate(images, 1):
            out_name = _mapped_output_name(src, used_names)
            out_path = output_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            is_bg = is_background_asset(str(src))
            mode = "background" if is_bg else "birefnet"
            logger.info("[%d/%d] %s → %s (%s)", i, len(images), src.name, out_name, mode)

            t_img = time.perf_counter()
            try:
                if is_bg:
                    compress_to_webp_only(str(src), output_path=str(out_path))
                else:
                    remove_light_background(str(src), threshold=threshold, output_path=str(out_path))
                elapsed_img = time.perf_counter() - t_img
                out_kb = out_path.stat().st_size / 1024
                logger.info("[%d/%d] ✓ %s → %.1f KB (%.2fs)", i, len(images), out_name, out_kb, elapsed_img)
                ok_count += 1
            except Exception as exc:
                logger.error("[%d/%d] ✗ erro em %s: %s", i, len(images), src.name, exc, exc_info=True)
                err_count += 1

        logger.info(
            "process_zip: processamento concluído — %d ok, %d erros (%.2fs total)",
            ok_count, err_count, time.perf_counter() - t_total,
        )

        if ok_count == 0:
            logger.error("process_zip: todas as imagens falharam")
            raise HTTPException(status_code=500, detail="falha ao processar todas as imagens")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(output_dir.rglob("*")):
                if file_path.is_file():
                    arcname = Path("images") / file_path.relative_to(output_dir)
                    zf.write(file_path, arcname=str(arcname))
        zip_buffer.seek(0)
        result = zip_buffer.read()
        logger.info("process_zip: ZIP de saída gerado (%.1f KB)", len(result) / 1024)
        return result
