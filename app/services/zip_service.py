from __future__ import annotations

import io
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import HTTPException

from app.services.image_service import (
    compress_to_webp_only,
    is_background_asset,
    remove_light_background,
)

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
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            _extract_zip_archive(zf, dest_dir)
    except HTTPException:
        raise
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="arquivo zip inválido")
    except Exception:
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
    path_str = str(original_path)
    bucket = _extract_symbol_bucket(path_str)
    idx = _extract_index(path_str)

    if bucket:
        if idx:
            base = f"{bucket}{idx}"
        elif bucket == "W":
            base = "W"
        else:
            base = bucket
    else:
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
    with tempfile.TemporaryDirectory(prefix="bgapi_") as tmp:
        root = Path(tmp)
        input_dir = root / "in"
        output_dir = root / "out"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        extract_zip_bytes_to_dir(zip_bytes, input_dir)
        images = list_images_in_folder(input_dir)

        used_names: set[str] = set()
        for src in images:
            out_name = _mapped_output_name(src, used_names)
            out_path = output_dir / out_name

            if is_background_asset(str(src)):
                compress_to_webp_only(str(src), output_path=str(out_path))
            else:
                remove_light_background(str(src), threshold=threshold, output_path=str(out_path))

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(output_dir.iterdir()):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.name)
        zip_buffer.seek(0)
        return zip_buffer.read()
