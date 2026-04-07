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
import torch
from torchvision import transforms

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
# BRIA RMBG 2.0 — modelo de segmentação estado da arte para remoção de fundo.
# Carregado uma única vez no startup; usa CUDA quando disponível.
# ---------------------------------------------------------------------------
_BRIA_MODEL = None
_BRIA_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_BRIA_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _git_version() -> str:
    """Retorna o commit hash atual + flag de dirty, pra logar qual versão tá rodando."""
    try:
        import subprocess
        repo_root = Path(__file__).resolve().parents[2]
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "-C", str(repo_root), "diff", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) != 0
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


_BG_API_VERSION = _git_version()
logger.info("=" * 60)
logger.info("  bg-api iniciando | versão git: %s", _BG_API_VERSION)
logger.info("=" * 60)


try:
    from transformers import AutoModelForImageSegmentation

    torch.set_float32_matmul_precision("high")
    logger.info("Carregando BiRefNet (ZhengPeng7) em %s…", _BRIA_DEVICE)
    _t = time.perf_counter()
    _BRIA_MODEL = (
        AutoModelForImageSegmentation
        .from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        .eval()
        .to(_BRIA_DEVICE)
    )
    logger.info("BiRefNet pronto em %.1fs | device=%s", time.perf_counter() - _t, _BRIA_DEVICE)
except Exception as _e:
    logger.error("Falha ao carregar BiRefNet: %s — usando fallback por threshold", _e)
    _BRIA_MODEL = None


def _detect_bg_mode(rgb_arr: np.ndarray) -> str:
    """
    Detecta o tipo de fundo da imagem amostrando os 4 cantos.

    Retorna:
      "dark"   — fundo preto/escuro (lum < 20/255 nos cantos)
      "light"  — fundo branco/claro (lum > 235/255 nos cantos)
      "mixed"  — qualquer outra coisa (fallback heurístico)
    """
    h, w = rgb_arr.shape[:2]
    s = max(8, min(h, w) // 32)  # tamanho do patch dos cantos
    corners = np.concatenate([
        rgb_arr[:s, :s].reshape(-1, 3),
        rgb_arr[:s, -s:].reshape(-1, 3),
        rgb_arr[-s:, :s].reshape(-1, 3),
        rgb_arr[-s:, -s:].reshape(-1, 3),
    ])
    mean_lum = float(corners.mean())
    if mean_lum < 20.0:
        return "dark"
    if mean_lum > 235.0:
        return "light"
    return "mixed"


def _bria_remove(image: Image.Image) -> Image.Image:
    """
    Remove o fundo usando BiRefNet + pós-processamento adaptativo ao tipo de bg.

    Detecta o fundo (preto/branco/misto) pelos cantos da imagem e roteia pra
    o pipeline dedicado:
      - "dark"  → pipeline matemático (premultiplied alpha)
      - "light" → pipeline heurístico (chroma augmentation + decontamination)
      - "mixed" → fallback (mesmo do light, sem garantias)
    Retorna RGBA.
    """
    rgb = image.convert("RGB")
    original_size = rgb.size  # (W, H)

    # ── 1. Inferência BiRefNet → máscara crua ─────────────────────────────
    t0 = time.perf_counter()
    tensor = _BRIA_TRANSFORM(rgb).unsqueeze(0).to(_BRIA_DEVICE)
    tensor = tensor.to(next(_BRIA_MODEL.parameters()).dtype)
    with torch.no_grad():
        preds = _BRIA_MODEL(tensor)[-1].sigmoid().cpu()
    raw_mask_pil = transforms.ToPILImage()(preds[0].squeeze())
    raw_mask_pil = raw_mask_pil.resize(original_size, Image.LANCZOS)
    logger.debug("_bria_remove: inferência em %.2fs", time.perf_counter() - t0)

    rgb_arr = np.array(rgb)
    rgb_f = rgb_arr.astype(np.float32) / 255.0
    mask_f = np.array(raw_mask_pil).astype(np.float32) / 255.0

    bg_mode = _detect_bg_mode(rgb_arr)
    logger.info("_bria_remove: bg_mode detectado=%s", bg_mode)

    if bg_mode == "dark":
        rgba = _process_dark_bg(rgb_arr, rgb_f, mask_f)
    else:
        rgba = _process_light_bg(rgb_arr, rgb_f, mask_f)

    logger.debug("_bria_remove: pipeline completo em %.2fs", time.perf_counter() - t0)
    return Image.fromarray(rgba, "RGBA")


def _process_dark_bg(
    rgb_arr: np.ndarray,
    rgb_f: np.ndarray,
    birefnet_mask: np.ndarray,
) -> np.ndarray:
    """
    Pipeline matemático pra fundo preto puro.

    Em fundo preto, cada pixel observado é premultiplied alpha:
        C_obs = C_real × α + 0 × (1 − α) = C_real × α

    Logo:
        α_natural = max(R, G, B) / 255           (alpha exato por pixel)
        C_real    = C_obs / α_natural             (decontamination exata)

    O alpha final combina o BiRefNet (objeto sólido) com o alpha natural
    (glows e VFX), restrito a uma região de interesse expandida do BiRefNet
    pra não capturar ruído distante do objeto.
    """
    t0 = time.perf_counter()

    # 1. Alpha natural — premultiplied alpha matematicamente exato
    alpha_natural = rgb_f.max(axis=2)  # (H, W) [0, 1]

    # 2. Região de interesse: expandir o BiRefNet pra alcançar glows próximos
    fg_seed = birefnet_mask > 0.5
    if fg_seed.any():
        h, w = birefnet_mask.shape
        diag = float(np.sqrt(h * h + w * w))
        sigma = max(20.0, diag * 0.12)
        distance = ndimage.distance_transform_edt(~fg_seed).astype(np.float32)
        roi = np.exp(-distance / sigma)  # falloff suave do objeto pra fora
    else:
        roi = np.ones_like(birefnet_mask)

    # 3. Combina: dentro do BiRefNet usa max(birefnet, natural).
    #    Fora, usa só alpha_natural × roi (decai com distância).
    combined = np.maximum(birefnet_mask, alpha_natural * roi)

    # 4. Refino de borda com guided filter (mais suave que no light mode)
    t1 = time.perf_counter()
    combined = _guided_filter_mask(rgb_f, combined, radius=4, eps=1e-3)
    logger.debug("_process_dark_bg: guided filter em %.2fs", time.perf_counter() - t1)

    alpha = np.clip(combined, 0.0, 1.0)

    # 5. Decontamination matemática exata: rgb / alpha
    #    (só onde alpha > epsilon pra evitar divisão por zero)
    rgb_unmix = np.zeros_like(rgb_f)
    safe = alpha > 0.01
    rgb_unmix[safe] = rgb_f[safe] / alpha[safe, np.newaxis]
    rgb_unmix = np.clip(rgb_unmix, 0.0, 1.0)

    rgba = np.zeros((rgb_arr.shape[0], rgb_arr.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = (rgb_unmix * 255).astype(np.uint8)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)

    n_opaque = int((rgba[..., 3] > 240).sum())
    n_semi = int(((rgba[..., 3] > 5) & (rgba[..., 3] <= 240)).sum())
    logger.info(
        "_process_dark_bg: total %.2fs — opaque=%d semi=%d",
        time.perf_counter() - t0, n_opaque, n_semi,
    )
    return rgba


def _process_light_bg(
    rgb_arr: np.ndarray,
    rgb_f: np.ndarray,
    mask_f: np.ndarray,
) -> np.ndarray:
    """
    Pipeline heurístico pra fundo branco/claro (e fallback pra fundos mistos).
    Mantém a lógica original: chroma augmentation + guided filter +
    adaptive feather + decontamination.
    """
    t0 = time.perf_counter()
    bg_color = _estimate_bg_color(rgb_arr.astype(np.float32), mask_f)
    logger.debug("_process_light_bg: bg_color=%s", bg_color.astype(int))

    t1 = time.perf_counter()
    mask_before = mask_f.copy()
    mask_f, augmented_region = _chroma_augment_mask(rgb_arr, mask_f, bg_color)
    delta = mask_f - mask_before
    n_aug = int((delta > 0.01).sum())
    logger.info(
        "_process_light_bg: chroma augment em %.2fs — recuperados=%d max=%.3f mean=%.3f",
        time.perf_counter() - t1,
        n_aug,
        float(delta.max()) if n_aug else 0.0,
        float(delta[delta > 0.01].mean()) if n_aug else 0.0,
    )

    t2 = time.perf_counter()
    mask_f = _guided_filter_mask(rgb_f, mask_f, radius=8, eps=1e-4)
    logger.debug("_process_light_bg: guided filter em %.2fs", time.perf_counter() - t2)

    t3 = time.perf_counter()
    mask_f = _adaptive_feather(rgb_f, mask_f)
    logger.debug("_process_light_bg: adaptive feather em %.2fs", time.perf_counter() - t3)

    alpha = (np.clip(mask_f, 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([rgb_arr, alpha])

    t4 = time.perf_counter()
    rgba = _decontaminate_colors(rgba, bg_color, skip_mask=augmented_region)
    logger.debug("_process_light_bg: decontamination em %.2fs", time.perf_counter() - t4)
    logger.info("_process_light_bg: total %.2fs", time.perf_counter() - t0)
    return rgba


# ---------------------------------------------------------------------------
# Pós-processamento de máscara e cores — qualidade profissional
# ---------------------------------------------------------------------------

def _chroma_augment_mask(
    rgb_arr: np.ndarray,
    mask: np.ndarray,
    bg_color: np.ndarray,
    *,
    signal_threshold: float = 0.06,
    max_boost: float = 0.90,
    proximity_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recupera efeitos coloridos semi-transparentes (glows, raios, chamas)
    que o BiRefNet classificou como background.

    BiRefNet foi treinado em fotos de objetos físicos — efeitos VFX de jogo
    são fora da distribuição. Glows azuis/laranjas/coloridos com baixa
    opacidade frequentemente recebem alpha=0 do modelo, mesmo sendo parte
    visível do asset.

    Sinal usado (combinação de métricas direcionais — só dispara pra glow real):
      a) saturation    — (max - min)/max do RGB local (pixel é colorido)
      b) color_cast    — desvio padrão dos canais (pixel "puxa" pra uma cor)
      c) brighter      — max(0, lum - bg_lum) (pixel mais claro que o bg = glow)

    Crucial: nunca usa "mais escuro que bg" — isso captaria contornos
    anti-aliased / halos de compressão JPEG ao redor do objeto, gerando
    falsos positivos visíveis como "recortes do branco".

    Pondera por proximidade ao foreground (distance transform) e combina
    com a máscara original via max() — só aumenta, nunca reduz.

    Retorna (mask_aumentada, augmented_region_bool) onde augmented_region
    marca pixels que foram recuperados pela augmentation (para que a
    decontamination posterior possa pulá-los).
    """
    fg = mask > 0.5
    if not fg.any():
        return mask, np.zeros_like(mask, dtype=bool)

    rgb_f = rgb_arr.astype(np.float32) / 255.0
    bg_f = bg_color.astype(np.float32).reshape(1, 1, 3) / 255.0

    # (a) Saturação — pixel "puxa" pra uma cor (não dispara em cinzas)
    rgb_max = rgb_f.max(axis=2)
    rgb_min = rgb_f.min(axis=2)
    saturation = (rgb_max - rgb_min) / np.maximum(rgb_max, 1e-6)

    # (b) Color cast — std dos canais, captura tons levemente coloridos
    color_cast = rgb_f.std(axis=2) * 2.0  # *2 pra deixar na mesma escala

    # (c) "Mais brilhante que o bg" — captura glows brancos/claros sobre fundo escuro.
    # Usa max() por canal pra também capturar saturação direcional (ex.: azul puro
    # mais saturado que o bg branco em um dos canais).
    lum = rgb_f.mean(axis=2)
    bg_lum = float(bg_f.mean())
    brighter = np.clip(lum - bg_lum, 0.0, 1.0)

    # Sinal combinado — max das três (todas direcionais, não disparam em halo escuro)
    signal = np.maximum.reduce([saturation, color_cast, brighter])

    # Soft threshold: pixels acima do threshold ganham peso linearmente
    chroma_mask = np.clip((signal - signal_threshold) * 3.0, 0.0, 1.0)

    # Proximidade ao foreground via distance transform
    h, w = mask.shape
    diag = float(np.sqrt(h * h + w * w))
    sigma = max(20.0, diag * proximity_frac)
    distance = ndimage.distance_transform_edt(~fg).astype(np.float32)
    proximity = np.exp(-distance / sigma)

    # Augmentation final
    augmentation = chroma_mask * proximity * max_boost

    new_mask = np.maximum(mask, augmentation)
    augmented_region = (new_mask - mask) > 0.05  # marca pixels efetivamente boostados

    return new_mask, augmented_region


def _guided_filter_mask(
    guide_rgb: np.ndarray,
    mask: np.ndarray,
    radius: int = 8,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Refina a máscara usando guided filter com a imagem original como guia.

    A máscara do BiRefNet (1024×1024 upscaled) tem bordas borradas/dentadas.
    O guided filter usa os gradientes da imagem em resolução completa para
    alinhar a máscara às bordas reais do objeto — preservando detalhes finos
    (pontas de espada, joias, letras) que se perdem no upscale.

    Complexidade: O(N) independente do raio (uniform_filter é separável).

    guide_rgb: (H, W, 3) float32 [0,1]
    mask:      (H, W)    float32 [0,1]
    """
    from scipy.ndimage import uniform_filter

    guide_gray = np.mean(guide_rgb, axis=2)
    k = 2 * radius + 1

    mean_I = uniform_filter(guide_gray, size=k)
    mean_p = uniform_filter(mask, size=k)
    mean_Ip = uniform_filter(guide_gray * mask, size=k)
    mean_II = uniform_filter(guide_gray * guide_gray, size=k)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size=k)
    mean_b = uniform_filter(b, size=k)

    return np.clip(mean_a * guide_gray + mean_b, 0.0, 1.0)


def _adaptive_feather(
    guide_rgb: np.ndarray,
    mask: np.ndarray,
    sigma_sharp: float = 0.5,
    sigma_soft: float = 2.5,
    boundary_px: int = 4,
) -> np.ndarray:
    """
    Feathering adaptativo baseado no gradiente local da imagem.

    - Bordas com alto contraste (metal, contornos nítidos) → transição fina (sigma_sharp)
    - Bordas com gradiente suave (glow, fumaça, reflexos) → transição ampla (sigma_soft)
    - Só modifica pixels na vizinhança da fronteira da máscara (boundary_px).

    guide_rgb: (H, W, 3) float32 [0,1]
    mask:      (H, W)    float32 [0,1]
    """
    gray = np.mean(guide_rgb, axis=2)
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    p99 = np.percentile(grad_mag, 99) + 1e-8
    grad_norm = np.clip(grad_mag / p99, 0.0, 1.0)

    # Fronteira da máscara (dilatação − erosão)
    fg = mask > 0.5
    dilated = ndimage.binary_dilation(fg, iterations=boundary_px)
    eroded = ndimage.binary_erosion(fg, iterations=boundary_px)
    boundary = dilated & ~eroded

    if not boundary.any():
        return mask

    blur_sharp = ndimage.gaussian_filter(mask, sigma=sigma_sharp)
    blur_soft = ndimage.gaussian_filter(mask, sigma=sigma_soft)

    # Alto gradiente → sharp, baixo gradiente → soft
    blended = blur_sharp * grad_norm + blur_soft * (1.0 - grad_norm)

    result = mask.copy()
    result[boundary] = blended[boundary]
    return result


def _estimate_bg_color(rgb_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Estima a cor dominante do fundo a partir dos pixels com baixo valor de máscara.

    rgb_arr: (H, W, 3) float32 [0, 255]
    mask:    (H, W)    float32 [0, 1]
    Retorna: (3,) float32 [0, 255]
    """
    bg_pixels = mask < 0.1
    if bg_pixels.sum() < 100:
        return np.array([255.0, 255.0, 255.0])
    return np.median(rgb_arr[bg_pixels], axis=0)


def _decontaminate_colors(
    rgba: np.ndarray,
    bg_color: np.ndarray,
    skip_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Remove sangramento de cor do fundo em pixels semi-transparentes (halo).

    Quando um asset está sobre fundo claro, os pixels da borda são uma mistura:
        C_observada = C_real × α + C_fundo × (1 − α)

    Invertendo para recuperar a cor original do foreground:
        C_real = (C_observada − C_fundo × (1 − α)) / α

    Isso elimina o halo branco/claro que aparece quando o asset processado
    é colocado sobre um background escuro (ex.: tema de jogo noturno).

    rgba:     (H, W, 4) uint8
    bg_color: (3,) float32 [0, 255]
    """
    result = rgba.astype(np.float32)
    alpha = result[:, :, 3] / 255.0

    # Só processa pixels semi-transparentes (borda real do objeto)
    edge = (alpha > 0.02) & (alpha < 0.92)
    if skip_mask is not None:
        # Pixels recuperados pela chroma augmentation não são misturas
        # físicas com o bg — pular para preservar suas cores reais.
        edge = edge & ~skip_mask
    if not edge.any():
        return rgba

    a = alpha[edge, np.newaxis]                     # (N, 1)
    rgb = result[edge, :3]                           # (N, 3)
    bg = bg_color.astype(np.float32).reshape(1, 3)   # (1, 3)

    a_safe = np.maximum(a, 0.05)
    decontaminated = (rgb - bg * (1.0 - a)) / a_safe
    result[edge, :3] = np.clip(decontaminated, 0.0, 255.0)

    return np.clip(result, 0.0, 255.0).astype(np.uint8)


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


def is_frame_asset(input_path: str) -> bool:
    """Public wrapper para detecção de frame (grid/moldura)."""
    return _is_frame_asset(input_path)


def is_logo_asset(input_path: str) -> bool:
    """True se o arquivo for um logo de jogo."""
    return "logo" in Path(input_path).stem.lower()


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
    bg_mode: str = "light",
) -> np.ndarray:
    """
    Detecta pixels de fundo usando componentes conectados no original.

    bg_mode="light": pixels claros (R,G,B >= threshold-10) e acromáticos
    bg_mode="dark":  pixels escuros (R,G,B <= dark_threshold) e acromáticos

    1. Marca pixels candidatos a fundo
    2. Faz labeling de componentes conectados (4-conectividade)
    3. Seleciona componentes que tocam a borda da imagem (fundo externo)
       OU contêm os centros das células (fundo interno das células)
    4. Retorna máscara booleana de fundo

    Vantagem sobre inset: para naturalmente nos divisores coloridos do frame,
    sem precisar definir zonas de exclusão manualmente.
    """
    rgb = original_arr[:, :, :3].astype(np.int16)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    mn = np.minimum(np.minimum(r_ch, g_ch), b_ch)

    if bg_mode == "dark":
        # Fundo preto: pixels muito escuros e acromáticos.
        # Glows coloridos (verde, azul, laranja) têm canais altos → NÃO são fundo.
        dark_thr = int(os.getenv("FRAME_DARK_BG_THRESHOLD", "20"))
        is_bg = (r_ch <= dark_thr) & (g_ch <= dark_thr) & (b_ch <= dark_thr) & ((mx - mn) <= 15)
    else:
        # Fundo branco: pixels claros e pouco saturados (acromáticos).
        thr = threshold - 10
        is_bg = (r_ch >= thr) & (g_ch >= thr) & (b_ch >= thr) & ((mx - mn) <= 15)

    labeled, _ = ndimage.label(is_bg)

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

    # lookup table O(n_pixels) — muito mais rápido que np.isin para muitos labels
    lut = np.zeros(labeled.max() + 1, dtype=bool)
    for lbl in bg_labels:
        lut[lbl] = True
    return lut[labeled]


def _is_light_background(original_arr: np.ndarray, threshold: int) -> bool:
    """Verifica se o fundo da imagem é claro amostrando os cantos das 4 bordas."""
    thr = threshold - 20
    h, w = original_arr.shape[:2]
    n = min(15, h // 4, w // 4)
    corners = np.concatenate([
        original_arr[:n, :n, :3].reshape(-1, 3),
        original_arr[:n, -n:, :3].reshape(-1, 3),
        original_arr[-n:, :n, :3].reshape(-1, 3),
        original_arr[-n:, -n:, :3].reshape(-1, 3),
    ])
    light = (corners[:, 0] >= thr) & (corners[:, 1] >= thr) & (corners[:, 2] >= thr)
    return bool(light.mean() > 0.7)


def _remove_frame_by_components(
    original_arr: np.ndarray,
    kind: FrameKind,
    threshold: int,
) -> np.ndarray:
    """
    Remove o fundo de um frame usando APENAS componentes conectados — sem rembg.

    Detecta automaticamente se o fundo é claro ou escuro pelos cantos da
    imagem e aplica a busca de componentes apropriada.

    Vantagens sobre BiRefNet para frames:
    - Borda externa preservada 100% (nunca tocamos pixels do frame art)
    - Resultado determinístico, sem variação entre runs
    - Muito mais rápido (sem inferência de modelo)
    - Funciona para qualquer estilo de frame, qualquer cor de divisor

    Algoritmo:
    1. Detecta bg_mode (light/dark) pelos cantos
    2. Marca pixels do fundo (claros OU escuros conforme bg_mode)
    3. Labels de componentes conectados (4-conectividade)
    4. Fundo externo  = componentes tocando bordas da imagem
       Fundo interno  = componentes tocando centros das células
    5. Ambos recebem alpha=0; tudo mais recebe alpha=255 (arte do frame intacta)
    6. Aplica Gaussian feather de 1px na fronteira para suavizar a borda externa
    """
    h, w = original_arr.shape[:2]
    bg_mode = _detect_bg_mode(original_arr[:, :, :3])
    # Frames são "light" ou "dark" — "mixed" cai pra light por compatibilidade
    if bg_mode != "dark":
        bg_mode = "light"
    logger.info("_remove_frame_by_components: bg_mode=%s kind=%s", bg_mode, kind.value)

    seeds = _cell_center_seeds(h, w, kind)
    bg_mask = _connected_background_mask(original_arr, h, w, seeds, threshold, bg_mode=bg_mode)
    logger.debug("_remove_frame_by_components: bg_mask=%.1f%% do total", 100 * bg_mask.mean())

    result = original_arr.copy()
    result[bg_mask, 3] = 0
    result[~bg_mask, 3] = 255

    # Feather de 1px na fronteira: suaviza a borda externa do frame
    alpha_f = result[:, :, 3].astype(np.float32)
    blurred = ndimage.gaussian_filter(alpha_f, sigma=0.8)
    boundary = (
        ndimage.binary_dilation(bg_mask, iterations=2)
        & ~ndimage.binary_erosion(bg_mask, iterations=2)
    )
    alpha_f[boundary] = blurred[boundary]
    result[:, :, 3] = np.clip(alpha_f, 0, 255).astype(np.uint8)

    return result


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
        # Erode 3px antes de aplicar: preserva os pixels de anti-aliasing nas bordas
        # dos divisores que o rembg suavizou — evita cortes duros/grosseiros
        safe_mask = ndimage.binary_erosion(bg_mask, iterations=3)
        arr[safe_mask, :] = 0

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

    logger.debug("remove_background: %s (threshold=%d, birefnet=%s)", input_file.name, thr, _BRIA_MODEL is not None)
    t0 = time.perf_counter()

    image = Image.open(input_file)
    logger.debug("remove_background: imagem aberta %dx%d modo=%s", image.width, image.height, image.mode)

    is_frame = _is_frame_asset(input_path)

    if is_frame:
        # Frames: sempre componentes conectados puro (fundo sempre branco).
        # Sem rembg — borda 100% preservada, determinístico, rápido.
        original_arr = np.array(image.convert("RGBA"))
        kind = _detect_frame_kind(input_path)
        logger.info("remove_background: frame tipo=%s → componentes conectados", kind.value)
        arr = _remove_frame_by_components(original_arr, kind, thr)
        result = Image.fromarray(arr, "RGBA")
    else:
        # Símbolos e logos: BRIA RMBG 2.0
        if _BRIA_MODEL is not None:
            logger.debug("remove_background: chamando BiRefNet…")
            try:
                result: Image.Image = _bria_remove(image)
            except Exception as exc:
                logger.error("BRIA falhou (%s) — fallback threshold=%d", exc, thr)
                result = _remove_by_threshold(image, thr)
        else:
            result = _remove_by_threshold(image, thr)
        result = result.convert("RGBA")

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