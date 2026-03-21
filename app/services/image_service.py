from __future__ import annotations

import logging
import os
import re
import time
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger("bg_api.image_service")

# WebP: compressão agressiva mas estável para assets com alpha (method=6 = melhor ratio, mais lento).
# Ajuste via ambiente sem alterar código: OUTPUT_WEBP_QUALITY, OUTPUT_WEBP_METHOD, OUTPUT_WEBP_ALPHA_QUALITY.
_DEFAULT_WEBP_QUALITY = 72
_DEFAULT_WEBP_METHOD = 6
_DEFAULT_WEBP_ALPHA_QUALITY = 88


def _webp_save_kwargs() -> dict:
    quality = int(os.getenv("OUTPUT_WEBP_QUALITY", str(_DEFAULT_WEBP_QUALITY)))
    method = int(os.getenv("OUTPUT_WEBP_METHOD", str(_DEFAULT_WEBP_METHOD)))
    alpha_q = int(os.getenv("OUTPUT_WEBP_ALPHA_QUALITY", str(_DEFAULT_WEBP_ALPHA_QUALITY)))
    quality = max(1, min(100, quality))
    method = max(0, min(6, method))
    alpha_q = max(1, min(100, alpha_q))
    return {
        "format": "WEBP",
        "lossless": False,
        "quality": quality,
        "method": method,
        "alpha_quality": alpha_q,
        "exact": False,
    }


def save_rgba_as_webp(image: Image.Image, output_path: str | Path) -> Path:
    """Salva RGBA em WebP com encoder otimizado para tamanho (lossy + alpha)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = image.convert("RGBA")
    kwargs = _webp_save_kwargs()
    fmt = kwargs.pop("format")
    img.save(path, fmt, **kwargs)
    return path.resolve()


def save_rgb_as_webp(image: Image.Image, output_path: str | Path) -> Path:
    """Salva RGB em WebP (sem canal alpha — menor que RGBA quando não há transparência)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = image.convert("RGB")
    kwargs = _webp_save_kwargs()
    fmt = kwargs.pop("format")
    kwargs.pop("alpha_quality", None)  # não se aplica a RGB
    img.save(path, fmt, **kwargs)
    return path.resolve()


# Pastas (ou nomes exatos de segmento) que indicam arte de fundo — só compressão WebP.
_BACKGROUND_FOLDER_NAMES = frozenset({"background", "backgrounds"})


def is_background_asset(input_path: str) -> bool:
    """
    True para artes de fundo de jogo: não devem passar pelo extrator (rembg).
    Heurística:
    - pasta `background` ou `backgrounds` no caminho;
    - arquivo `background.ext`, `background_*.ext`, `backgrounds.ext`, `backgrounds_*.ext`.
    """
    path = Path(input_path)
    for part in path.parts:
        if part.lower() in _BACKGROUND_FOLDER_NAMES:
            return True
    stem = path.stem.lower()
    if stem in ("background", "backgrounds"):
        return True
    if stem.startswith("background_") or stem.startswith("backgrounds_"):
        return True
    return False


def compress_to_webp_only(input_path: str, *, output_path: str) -> str:
    """
    Apenas reencode para WebP com as mesmas opções de compressão — sem rembg nem fill_holes.
    Preserva alpha somente se a imagem de entrada tiver transparência.
    """
    input_file = Path(input_path)
    image = Image.open(input_file)
    final_output = Path(output_path)

    has_alpha = False
    if image.mode in ("RGBA", "LA"):
        has_alpha = True
    elif image.mode == "P":
        has_alpha = "transparency" in image.info

    if has_alpha:
        save_rgba_as_webp(image, final_output)
    else:
        save_rgb_as_webp(image, final_output)

    return str(final_output.resolve())

# ---------------------------------------------------------------------------
# rembg session — carregado uma única vez no import do módulo para evitar
# reinicialização a cada requisição (cada sessão carrega ~300 MB em memória).
# ---------------------------------------------------------------------------
try:
    from rembg import new_session, remove as rembg_remove

    import onnxruntime as _ort
    _available = _ort.get_available_providers()
    logger.info("ONNX providers disponíveis: %s", _available)

    # Usa GPU se disponível, com fallback para CPU
    if "CUDAExecutionProvider" in _available:
        _providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Carregando sessão rembg com CUDA (birefnet-general)…")
    else:
        _providers = ["CPUExecutionProvider"]
        logger.warning("⚠ CUDAExecutionProvider não disponível — usando CPU (instale onnxruntime-gpu)")

    _t = time.perf_counter()
    _SESSION = new_session("birefnet-general", providers=_providers)
    _REMBG_AVAILABLE = True
    logger.info("Sessão rembg pronta em %.1fs | providers ativos: %s", time.perf_counter() - _t, _providers)
except ImportError:
    logger.warning("rembg não instalado — usando fallback por threshold")
    _REMBG_AVAILABLE = False
    _SESSION = None
except Exception as _e:
    logger.error("Falha ao inicializar rembg: %s — usando fallback por threshold", _e)
    _REMBG_AVAILABLE = False
    _SESSION = None


# ---------------------------------------------------------------------------
# Fallback: remoção por threshold (fundo claro)
# Mantido para compatibilidade e para casos onde rembg não esteja instalado.
# ---------------------------------------------------------------------------

def _is_light_background_pixel(
    r: int,
    g: int,
    b: int,
    threshold: int = 245,
    max_channel_diff: int = 15,
) -> bool:
    is_light = r >= threshold and g >= threshold and b >= threshold
    low_variation = (
        abs(r - g) <= max_channel_diff
        and abs(r - b) <= max_channel_diff
        and abs(g - b) <= max_channel_diff
    )
    return is_light and low_variation


def _remove_by_threshold(image: Image.Image, threshold: int) -> Image.Image:
    """Remove fundo claro pixel a pixel (método legado / fallback)."""
    logger.warning(
        "_remove_by_threshold: usando fallback por threshold=%d (rembg indisponível). "
        "ATENÇÃO: O(W×H) em Python puro — lento para imagens grandes.",
        threshold,
    )
    img = image.convert("RGBA")
    pixels = img.load()
    width, height = img.size
    logger.debug("_remove_by_threshold: imagem %dx%d (%d pixels)", width, height, width * height)
    t0 = time.perf_counter()
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if _is_light_background_pixel(r, g, b, threshold=threshold):
                pixels[x, y] = (255, 255, 255, 0)
    logger.info("_remove_by_threshold: concluído em %.2fs", time.perf_counter() - t0)
    return img


# ---------------------------------------------------------------------------
# Função pública principal
# ---------------------------------------------------------------------------

def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    filled = ndimage.binary_fill_holes(mask > 128).astype(np.uint8) * 255
    return filled


def _is_frame_asset(input_path: str) -> bool:
    path = Path(input_path)
    tokens = [part.lower() for part in path.parts]
    filename = path.name.lower()
    return "frame" in filename or "frame" in tokens


class FrameKind(str, Enum):
    """Três layouts de moldura / grid para limpeza do interior (fundo claro residual)."""

    GRID_9 = "grid_9"  # 3×3 células para símbolos
    REELS_3 = "reels_3"  # 2 divisórias horizontais → 3 faixas verticais (reels)
    MOLDURA = "moldura"  # um único retângulo interno (moldura simples)


def _detect_frame_kind(input_path: str) -> FrameKind:
    """
    Detecta o tipo de frame pelo caminho/nome do arquivo (case-insensitive).

    Convenções sugeridas nos nomes:
    - GRID_9: grid, 3x3, nine, slotgrid, ninecell, ...
    - REELS_3: reels, 3reel, dividers, triple_strip, ...
    - MOLDURA: moldura, picture, inner_square, portal, ...

    Se só existir o token genérico `frame`, assume MOLDURA (menos agressivo).

    Override: variável de ambiente `FRAME_FORCE_KIND` = `grid_9` | `reels_3` | `moldura`.
    """
    forced = os.getenv("FRAME_FORCE_KIND", "").strip().lower()
    if forced in ("grid_9", "grid9", "grid", "9"):
        return FrameKind.GRID_9
    if forced in ("reels_3", "reels3", "3reel"):
        return FrameKind.REELS_3
    if forced in ("moldura", "single", "1"):
        return FrameKind.MOLDURA

    s = str(Path(input_path)).lower()
    compact = re.sub(r"[^a-z0-9]+", "", s)

    def _has(*subs: str) -> bool:
        return any(sub in s or sub in compact for sub in subs)

    # Ordem: mais específico primeiro
    if _has("3x3", "ninecell", "nine_cell", "slotgrid", "grid9", "9grid", "grid_9"):
        return FrameKind.GRID_9
    if _has("grid", "nine", "3by3", "3-by-3", "slotmatrix", "matrix3", "3matrix"):
        return FrameKind.GRID_9

    if _has("reels3", "3reel", "triplestrip", "triple_strip", "reels_3", "divider", "dividers"):
        return FrameKind.REELS_3
    if _has("reelstrip", "reel_strip", "threecolumn", "three_column", "3column", "3_column"):
        return FrameKind.REELS_3

    if _has("moldura", "pictureframe", "inner_square", "innersquare", "portal", "squareinner"):
        return FrameKind.MOLDURA

    return FrameKind.MOLDURA


def _usable_inner_bounds(h: int, w: int, margin_frac: float) -> tuple[int, int, int, int]:
    """Retângulo interno (evita bordas da textura / sombras)."""
    mx = max(1, int(w * margin_frac))
    my = max(1, int(h * margin_frac))
    return my, h - my, mx, w - mx


def _vector_light_mask(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    threshold: int,
    max_channel_diff: int = 18,
) -> np.ndarray:
    """Pixels claros e pouco saturados (restos de fundo branco/cinza)."""
    is_light = (r >= threshold) & (g >= threshold) & (b >= threshold)
    low_var = (
        (np.abs(r.astype(np.int16) - g) <= max_channel_diff)
        & (np.abs(r.astype(np.int16) - b) <= max_channel_diff)
        & (np.abs(g.astype(np.int16) - b) <= max_channel_diff)
    )
    return is_light & low_var


def _whitish_interior_mask(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    *,
    mean_min: int,
    spread_max: int,
    weakest_channel_min: int,
) -> np.ndarray:
    """
    Restos de fundo com tom frio/calor (ex.: B> R em ~20) que falham no teste
    'RGB quase iguais', mas ainda são claridade de fundo.
    """
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    spread = mx - mn
    mean = (r.astype(np.int32) + g.astype(np.int32) + b.astype(np.int32)) // 3
    return (spread <= spread_max) & (mean >= mean_min) & (mn >= weakest_channel_min)


def _frame_interior_thresholds(request_threshold: int) -> tuple[int, int, int, int]:
    """
    Thresholds para limpeza no interior de frames (mais agressivo que o fundo global).

    Retorna: per_channel_min, mean_luminance_min, chroma_spread_max, weakest_ch_min
    """
    # RGB mínimo por canal (mais baixo que 245 para pegar off-white)
    per_ch = int(os.getenv("FRAME_INTERIOR_PER_CHANNEL", "200"))
    per_ch = min(request_threshold, per_ch)
    mean_min = int(os.getenv("FRAME_INTERIOR_MEAN_MIN", "168"))
    spread_max = int(os.getenv("FRAME_INTERIOR_CHROMA_MAX", "52"))
    weak_min = int(os.getenv("FRAME_INTERIOR_WEAKEST_CH_MIN", "140"))
    return per_ch, mean_min, spread_max, weak_min


def _clear_light_in_rect(
    arr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    threshold: int,
    *,
    also_semi_opaque_light: bool = True,
    frame_interior: bool = False,
) -> None:
    """Zera alpha (e RGB) em pixels claros dentro do retângulo — in-place."""
    h, w = arr.shape[:2]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x0 >= x1 or y0 >= y1:
        return

    sl = arr[y0:y1, x0:x1]
    r = sl[:, :, 0].astype(np.int16)
    g = sl[:, :, 1].astype(np.int16)
    b = sl[:, :, 2].astype(np.int16)
    a = sl[:, :, 3].astype(np.int16)

    if frame_interior:
        pc, mean_min, spread_max, weak_min = _frame_interior_thresholds(threshold)
        # Critério A: canais altos e pouco saturados (diff relaxado para interiores)
        mask = _vector_light_mask(r, g, b, threshold=pc, max_channel_diff=36)
        # Critério B: luminância alta + croma baixo (branco azulado, cinza-claro)
        mask = mask | _whitish_interior_mask(
            r, g, b, mean_min=mean_min, spread_max=spread_max, weakest_channel_min=weak_min
        )
        if also_semi_opaque_light:
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            spread = mx - mn
            mean = (r.astype(np.int32) + g.astype(np.int32) + b.astype(np.int32)) // 3
            # Halo semi-opaco do rembg
            mask = mask | (
                (a > 20)
                & (a < 252)
                & (mean >= mean_min - 12)
                & (spread <= spread_max + 8)
                & (mx >= weak_min)
            )
    else:
        mask = _vector_light_mask(r, g, b, threshold=threshold)
        if also_semi_opaque_light:
            near_white = (r + g + b) >= 3 * (threshold - 25)
            grayish = (
                (np.abs(r - g) <= 22)
                & (np.abs(r - b) <= 22)
                & (np.abs(g - b) <= 22)
            )
            mask = mask | (near_white & grayish & (a > 25) & (a < 250))

    sl[mask] = [0, 0, 0, 0]


def _frame_regions_grid_9(h: int, w: int) -> list[tuple[int, int, int, int]]:
    """Nove retângulos (centro de cada célula), com inset para não comer a arte das divisórias."""
    margin = float(os.getenv("FRAME_MARGIN_FRAC", "0.07"))
    # Inset menor = cobre mais a célula (restos costumam colar nas divisórias).
    cell_inset = float(os.getenv("FRAME_CELL_INSET_FRAC", "0.055"))
    y0b, y1b, x0b, x1b = _usable_inner_bounds(h, w, margin)
    uw, uh = x1b - x0b, y1b - y0b
    cw, ch = max(1, uw // 3), max(1, uh // 3)
    ix = max(1, int(cw * cell_inset))
    iy = max(1, int(ch * cell_inset))
    boxes: list[tuple[int, int, int, int]] = []
    for gy in range(3):
        for gx in range(3):
            cx0 = x0b + gx * cw
            cy0 = y0b + gy * ch
            cx1, cy1 = cx0 + cw, cy0 + ch
            boxes.append((cx0 + ix, cy0 + iy, cx1 - ix, cy1 - iy))
    return boxes


def _frame_regions_reels_3(h: int, w: int) -> list[tuple[int, int, int, int]]:
    """Três faixas verticais (uma coluna por reel), com inset vertical."""
    margin = float(os.getenv("FRAME_MARGIN_FRAC", "0.07"))
    strip_inset_x = float(os.getenv("FRAME_REEL_INSET_X_FRAC", "0.08"))
    y_inset = float(os.getenv("FRAME_REEL_INSET_Y_FRAC", "0.12"))
    y0b, y1b, x0b, x1b = _usable_inner_bounds(h, w, margin)
    uw, uh = x1b - x0b, y1b - y0b
    sw = max(1, uw // 3)
    ix = max(1, int(sw * strip_inset_x))
    iy = max(1, int(uh * y_inset))
    boxes: list[tuple[int, int, int, int]] = []
    for gx in range(3):
        sx0 = x0b + gx * sw
        sx1 = sx0 + sw
        boxes.append((sx0 + ix, y0b + iy, sx1 - ix, y1b - iy))
    return boxes


def _frame_regions_moldura(h: int, w: int) -> list[tuple[int, int, int, int]]:
    """Um único buraco interno (moldura); margem maior que o grid."""
    margin = float(os.getenv("FRAME_MARGIN_FRAC", "0.07"))
    inner = float(os.getenv("FRAME_MOLDURA_INNER_FRAC", "0.11"))
    y0b, y1b, x0b, x1b = _usable_inner_bounds(h, w, margin)
    uw, uh = x1b - x0b, y1b - y0b
    ix = max(1, int(uw * inner))
    iy = max(1, int(uh * inner))
    return [(x0b + ix, y0b + iy, x1b - ix, y1b - iy)]


def _cell_deep_centers(h: int, w: int, kind: FrameKind) -> list[tuple[int, int, int, int]]:
    """
    Retorna regiões do centro profundo de cada célula (inset de 20%) para limpeza incondicional.
    Essas regiões ficam longe de qualquer divisor — podem ser zeradas com segurança.
    Override via FRAME_DEEP_INSET_FRAC (default 0.20).
    """
    deep_inset = float(os.getenv("FRAME_DEEP_INSET_FRAC", "0.20"))
    margin = float(os.getenv("FRAME_MARGIN_FRAC", "0.07"))
    y0b, y1b, x0b, x1b = _usable_inner_bounds(h, w, margin)
    uw, uh = x1b - x0b, y1b - y0b

    if kind is FrameKind.GRID_9:
        cw, ch = max(1, uw // 3), max(1, uh // 3)
        ix = max(1, int(cw * deep_inset))
        iy = max(1, int(ch * deep_inset))
        boxes = []
        for gy in range(3):
            for gx in range(3):
                cx0 = x0b + gx * cw
                cy0 = y0b + gy * ch
                cx1, cy1 = cx0 + cw, cy0 + ch
                boxes.append((cx0 + ix, cy0 + iy, cx1 - ix, cy1 - iy))
        return boxes

    if kind is FrameKind.REELS_3:
        sw = max(1, uw // 3)
        ix = max(1, int(sw * deep_inset))
        iy = max(1, int(uh * deep_inset))
        return [(x0b + gx * sw + ix, y0b + iy, x0b + (gx + 1) * sw - ix, y1b - iy) for gx in range(3)]

    # MOLDURA
    ix = max(1, int(uw * deep_inset))
    iy = max(1, int(uh * deep_inset))
    return [(x0b + ix, y0b + iy, x1b - ix, y1b - iy)]


def _cell_center_seeds(h: int, w: int, kind: FrameKind) -> list[tuple[int, int]]:
    """Retorna coordenadas (y, x) aproximadas dos centros de cada célula para seeding."""
    margin = float(os.getenv("FRAME_MARGIN_FRAC", "0.07"))
    y0b, y1b, x0b, x1b = _usable_inner_bounds(h, w, margin)
    uw, uh = x1b - x0b, y1b - y0b
    if kind is FrameKind.GRID_9:
        cw, ch = max(1, uw // 3), max(1, uh // 3)
        return [
            (y0b + gy * ch + ch // 2, x0b + gx * cw + cw // 2)
            for gy in range(3) for gx in range(3)
        ]
    if kind is FrameKind.REELS_3:
        sw = max(1, uw // 3)
        return [(y0b + uh // 2, x0b + gx * sw + sw // 2) for gx in range(3)]
    # MOLDURA
    return [(y0b + uh // 2, x0b + uw // 2)]


def _connected_background_mask(
    original_arr: np.ndarray,
    h: int,
    w: int,
    cell_seeds_yx: list[tuple[int, int]],
    threshold: int,
) -> np.ndarray:
    """
    Detecta pixels de fundo usando componentes conectados no original.

    1. Marca pixels "claros" no original (R,G,B >= threshold-10) como candidatos
    2. Faz labeling de componentes conectados (4-conectividade)
    3. Seleciona componentes que tocam a borda da imagem (fundo externo)
       OU contêm os centros das células (fundo interno das células)
    4. Retorna máscara booleana de fundo

    Vantagem sobre inset: para naturalmente nos divisores coloridos do frame,
    sem precisar definir zonas de exclusão manualmente.
    """
    thr = threshold - 10
    rgb = original_arr[:, :, :3].astype(np.int16)
    is_light = (rgb[:, :, 0] >= thr) & (rgb[:, :, 1] >= thr) & (rgb[:, :, 2] >= thr)

    labeled, _ = ndimage.label(is_light)

    bg_labels: set[int] = set()

    # Componentes que tocam as bordas da imagem = fundo externo
    border_labels = (
        set(labeled[0, :].tolist())
        | set(labeled[-1, :].tolist())
        | set(labeled[:, 0].tolist())
        | set(labeled[:, -1].tolist())
    )
    border_labels.discard(0)
    bg_labels.update(border_labels)

    # Componentes que contêm os centros das células = fundo interno
    for cy, cx in cell_seeds_yx:
        if 0 <= cy < h and 0 <= cx < w:
            label = int(labeled[cy, cx])
            if label > 0:
                bg_labels.add(label)

    if not bg_labels:
        return np.zeros((h, w), dtype=bool)

    return np.isin(labeled, list(bg_labels))


def _cleanup_frame_interior_by_kind(
    arr: np.ndarray,
    kind: FrameKind,
    threshold: int,
    original_arr: np.ndarray | None = None,
) -> np.ndarray:
    """Aplica limpeza de fundo claro só nas regiões internas esperadas para cada tipo."""
    h, w = arr.shape[:2]
    if kind is FrameKind.GRID_9:
        regions = _frame_regions_grid_9(h, w)
    elif kind is FrameKind.REELS_3:
        regions = _frame_regions_reels_3(h, w)
    else:
        regions = _frame_regions_moldura(h, w)

    # Passagem 1: limpeza por cor (rembg output) — conservadora, protege bordas
    for x0, y0, x1, y1 in regions:
        _clear_light_in_rect(arr, x0, y0, x1, y1, threshold, frame_interior=True)

    # Passagem 2: componentes conectados no original
    # Identifica precisamente o fundo branco das células sem depender de insets:
    # para naturalmente nos divisores coloridos do frame.
    if original_arr is not None:
        seeds = _cell_center_seeds(h, w, kind)
        bg_mask = _connected_background_mask(original_arr, h, w, seeds, threshold)
        logger.debug("_cleanup_frame: bg_mask cobre %d px (%.1f%%)", bg_mask.sum(), 100 * bg_mask.mean())
        arr[bg_mask, :] = 0  # força RGBA=0 em todo pixel que era fundo no original

        # Passagem 3: limpeza complementar por cor nas regiões de inset
        # Pixels "órfãos" que ficaram desconectados do seed por anti-aliasing
        # dos divisores não são pegos pelo connected components. Uma varredura
        # por cor no original dentro das regiões de inset (mais conservadoras)
        # limpa esses resíduos sem risco de comer arte do frame.
        thr = threshold - 10
        for x0, y0, x1, y1 in regions:
            x0c = max(0, min(w, x0))
            x1c = max(0, min(w, x1))
            y0c = max(0, min(h, y0))
            y1c = max(0, min(h, y1))
            if x0c >= x1c or y0c >= y1c:
                continue
            orig_sl = original_arr[y0c:y1c, x0c:x1c, :3].astype(np.int16)
            was_white = (
                (orig_sl[:, :, 0] >= thr)
                & (orig_sl[:, :, 1] >= thr)
                & (orig_sl[:, :, 2] >= thr)
            )
            arr[y0c:y1c, x0c:x1c][was_white] = 0

    return arr


def remove_background(
    input_path: str,
    *,
    output_path: str,
    threshold: int | None = None,
) -> str:
    input_file = Path(input_path)
    thr = int(threshold if threshold is not None else 245)

    logger.debug("remove_background: %s (threshold=%d, rembg=%s)", input_file.name, thr, _REMBG_AVAILABLE)
    t0 = time.perf_counter()

    image = Image.open(input_file)
    logger.debug("remove_background: imagem aberta %dx%d modo=%s", image.width, image.height, image.mode)

    is_frame = _is_frame_asset(input_path)
    # Captura o array original antes do rembg para uso no cleanup de frames
    original_arr = np.array(image.convert("RGBA")) if is_frame else None

    if _REMBG_AVAILABLE:
        logger.debug("remove_background: chamando rembg…")
        try:
            result: Image.Image = rembg_remove(image, session=_SESSION)
        except Exception as exc:
            logger.error(
                "remove_background: rembg falhou (%s) — aplicando fallback por threshold=%d",
                exc, thr,
            )
            result = _remove_by_threshold(image, thr)
    else:
        result = _remove_by_threshold(image, thr)

    result = result.convert("RGBA")

    # Para frames, não usar fill_holes no alpha inteiro (preserva buracos internos).
    # Tipos: grid 3×3, 3 reels (faixas), moldura única — limpeza direcionada no interior.
    arr = np.array(result)
    alpha = arr[:, :, 3]
    if not is_frame:
        logger.debug("remove_background: aplicando fill_holes no alpha")
        arr[:, :, 3] = ndimage.binary_fill_holes(alpha > 128).astype(np.uint8) * 255
    else:
        kind = _detect_frame_kind(input_path)
        logger.info("remove_background: frame detectado — tipo=%s arquivo=%s", kind.value, input_file.name)
        arr = _cleanup_frame_interior_by_kind(arr, kind, thr, original_arr=original_arr)

    result = Image.fromarray(arr, "RGBA")

    final_output = Path(output_path)
    save_rgba_as_webp(result, final_output)

    out_size = final_output.stat().st_size
    logger.debug(
        "remove_background: salvo em %.2fs → %s (%.1f KB)",
        time.perf_counter() - t0, final_output.name, out_size / 1024,
    )
    return str(final_output.resolve())


def remove_light_background(
    input_path: str,
    threshold: int = 245,
    *,
    output_path: str,
) -> str:
    """
    Alias de compatibilidade para `remove_background`.

    Quando rembg estiver disponível, o parâmetro `threshold` é ignorado e o
    BiRefNet é usado — garantindo qualidade máxima mesmo para fundos escuros
    ou complexos como os seus game assets.
    """
    return remove_background(
        input_path=input_path,
        output_path=output_path,
        threshold=threshold,
    )