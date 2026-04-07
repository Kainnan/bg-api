# bg-api — Documentação Técnica Completa

> API de processamento de assets visuais para jogos de cassino.
> Remove fundos, classifica assets automaticamente e entrega WebP otimizado.

---

## 1. Visão Geral

**bg-api** é uma REST API construída com FastAPI que recebe imagens (ou ZIPs de imagens) geradas por um sistema de criação de cassinos e as processa automaticamente:

- **Remove fundos** de símbolos, logos e frames usando o modelo BiRefNet (deep learning)
- **Classifica assets** por tipo (background, frame, logo, símbolo) via heurísticas de nome/caminho
- **Comprime para WebP** com qualidade configurável
- **Renomeia** arquivos seguindo convenção padronizada (L1, H1, W, etc.)
- **Empacota** o resultado em ZIP estruturado

Toda operação acontece em memória / diretórios temporários — **nada é persistido em disco**.

---

## 2. Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI App                          │
│                      (app/main.py)                          │
│                                                             │
│  ┌─────────────┐    ┌──────────────────────────────────┐   │
│  │   Routes     │    │          Middleware               │   │
│  │ /process-*   │    │  - Request/Response logging       │   │
│  │ /health      │    │  - Error catching (500 → JSON)    │   │
│  └──────┬───────┘    └──────────────────────────────────┘   │
│         │                                                    │
│  ┌──────▼──────────────────────────────────────────────┐    │
│  │                   Services                           │    │
│  │                                                      │    │
│  │  ┌─────────────────────┐  ┌──────────────────────┐  │    │
│  │  │   image_service.py  │  │   zip_service.py     │  │    │
│  │  │   (736 linhas)      │  │   (296 linhas)       │  │    │
│  │  │                     │  │                       │  │    │
│  │  │  • Classificação    │  │  • Extração segura    │  │    │
│  │  │  • BiRefNet (ML)    │  │  • Mapeamento nomes   │  │    │
│  │  │  • Connected Comp.  │  │  • Orquestração       │  │    │
│  │  │  • WebP encoding    │  │  • Reempacotamento    │  │    │
│  │  │  • Threshold fallbk │  │                       │  │    │
│  │  └─────────────────────┘  └──────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Modelo ML (carregado no startup)        │    │
│  │  BiRefNet (ZhengPeng7) — CUDA ou CPU                │    │
│  │  Input: 1024×1024 | Output: máscara sigmoid [0,1]   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Estrutura de Diretórios

```
bg-api/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, logging, middleware, startup
│   ├── routes/
│   │   ├── __init__.py
│   │   └── process.py           # POST /process-image, POST /process-zip
│   └── services/
│       ├── __init__.py
│       ├── image_service.py     # Core: classificação, ML, processamento de pixels
│       └── zip_service.py       # Extração ZIP, mapeamento de nomes, reempacotamento
├── requirements.txt
├── run_api.sh                   # Script de execução (usa .venv)
├── setup_runpod.sh              # Instalação para RunPod/servidor
├── README.md
└── .gitignore
```

---

## 3. Endpoints

### `GET /health`

Retorna status da API e disponibilidade do modelo ML.

**Resposta:**
```json
{"status": "ok", "rembg": true}
```

> **Bug conhecido:** O endpoint importa `_REMBG_AVAILABLE` que não existe mais no `image_service.py` (removido quando migrou de rembg para BiRefNet). Deveria referenciar `_BRIA_MODEL is not None`. Isso causa `ImportError` ao acessar `/health`.

### `POST /process-image`

Processa uma única imagem.

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `file` | UploadFile | obrigatório | PNG, JPEG ou WebP |
| `threshold` | int (Form) | 245 | Limiar para fallback de remoção por threshold (0-255) |

**Resposta:** `image/webp` como attachment (`image.webp`)

### `POST /process-zip`

Processa um ZIP inteiro de assets.

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `file` | UploadFile | obrigatório | Arquivo .zip com imagens |
| `threshold` | int (Form) | 245 | Limiar para fallback (0-255) |

**Resposta:** `application/zip` como attachment (`images.zip`)

O ZIP de saída contém diretório `images/` com todos os assets processados e renomeados.

---

## 4. Pipeline de Processamento

### 4.1 Classificação de Assets

Cada imagem é classificada automaticamente pelo caminho/nome do arquivo. A classificação determina qual algoritmo de processamento será aplicado:

| Tipo | Detecção (heurística) | Processamento |
|------|----------------------|---------------|
| **Background** | Pasta `background`/`backgrounds` ou nome `background_*` | Apenas compressão WebP (sem remoção de fundo) |
| **Frame** | `frame` no caminho ou nome do arquivo | Componentes conectados (sem ML) |
| **Logo** | `logo` no stem do arquivo | BiRefNet (ML) |
| **Símbolo** | Qualquer outro asset | BiRefNet (ML) com fallback threshold |

### 4.2 Processamento de Backgrounds

Função: `compress_to_webp_only()`

Apenas converte para WebP preservando transparência se existente. Sem nenhuma alteração de conteúdo.

### 4.3 Processamento de Símbolos e Logos (BiRefNet + Pós-processamento)

Função: `_bria_remove()` → `remove_background()`

Pipeline completo (5 etapas):

**Etapa 1 — Inferência BiRefNet (máscara crua):**
1. Converte imagem para RGB
2. Redimensiona para **1024×1024** (input fixo do modelo)
3. Normaliza com stats do ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
4. Cast para dtype do modelo (pode ser float16 em GPU)
5. Inferência sem gradientes → saída sigmoid produz máscara [0,1]
6. Redimensiona máscara de volta ao tamanho original (LANCZOS)

**Etapa 2 — Guided Filter (refinamento de bordas):**
- Usa a imagem original em resolução completa como guia
- Alinha a máscara low-res às bordas reais do objeto
- Preserva detalhes finos (pontas, joias, letras) perdidos no upscale
- Implementação: box filter separável via `scipy.ndimage.uniform_filter` — O(N) independente do raio
- Parâmetros: `radius=8`, `eps=1e-4`

**Etapa 3 — Adaptive Feathering (transições inteligentes):**
- Calcula gradiente local da imagem original
- Bordas de alto contraste (metal, contornos) → transição fina (`sigma=0.5`)
- Bordas de gradiente suave (glow, fumaça) → transição ampla (`sigma=2.5`)
- Só modifica pixels na vizinhança da fronteira (±4px)

**Etapa 4 — Montagem RGBA:**
- Máscara refinada convertida para canal alpha uint8
- Combinada com RGB original

**Etapa 5 — Color Decontamination (remoção de halo):**
- Estima cor dominante do fundo via mediana dos pixels com mask < 0.1
- Para pixels semi-transparentes (alpha entre 2%-92%):
  - `C_real = (C_observada - C_fundo × (1 - α)) / α`
- Elimina o halo branco/claro visível quando asset é colocado sobre fundo escuro

**Fallback (threshold):** Se BiRefNet não estiver disponível ou falhar:
- Iteração pixel a pixel em Python puro
- Pixel com R,G,B ≥ threshold E variação entre canais ≤ 15 → alpha = 0
- **Muito lento** — O(W×H) em Python puro, logado como WARNING

### 4.4 Processamento de Frames (Componentes Conectados)

Função: `_remove_frame_by_components()` + `_cleanup_frame_interior_by_kind()`

Frames são processados **sem ML** — usa algoritmo determinístico baseado em componentes conectados:

#### Detecção do tipo de frame (`_detect_frame_kind()`)

| Tipo | Keywords no nome/caminho | Layout |
|------|-------------------------|--------|
| **GRID_9** | `grid`, `3x3`, `nine`, `slotgrid`, `grid9`, `9grid` | 3×3 células para símbolos |
| **REELS_3** | `reels3`, `3reel`, `dividers`, `triple_strip`, `3_column` | 3 faixas verticais |
| **MOLDURA** | `moldura`, `inner_square`, `portal` (ou default) | Retângulo interno único |

Override via `FRAME_FORCE_KIND` env var.

#### Algoritmo de remoção (5 etapas)

1. **Detecção de pixels claros:** R,G,B ≥ threshold-10 E baixa saturação (spread ≤ 15)
2. **Labeling de componentes conectados** (4-conectividade via `scipy.ndimage.label`)
3. **Classificação de fundo:**
   - *Externo:* componentes que tocam bordas da imagem
   - *Interno:* componentes que contêm seeds (centros das células)
4. **Aplicação de alpha:** fundo → alpha=0, arte → alpha=255
5. **Feathering:** Gaussian blur σ=0.8 na fronteira (dilation-erosion de 2px) para suavizar bordas

#### Limpeza do interior (3 passagens)

**Passagem 1 — Limpeza por cor (conservadora):**
- Nas regiões de inset de cada célula
- Critério A: canais altos + diff relaxado (36) para interiores
- Critério B: luminância alta + croma baixo (brancos azulados/cinzentos)
- Também detecta halos semi-opacos (alpha entre 20-252)

**Passagem 2 — Componentes conectados no original:**
- Usa a imagem original (antes do processamento) para identificar o fundo branco real
- Erode 3px antes de aplicar para preservar anti-aliasing nas bordas dos divisores
- Resultado preciso sem depender de insets manuais

**Passagem 3 — Limpeza complementar por cor:**
- Pixels "órfãos" desconectados do seed por anti-aliasing
- Varredura por cor no original dentro das regiões de inset
- Limpa resíduos sem risco de afetar arte do frame

---

## 5. Pipeline ZIP (Orquestração)

Arquivo: `zip_service.py`

### 5.1 Extração Segura

- Validação contra **Zip Slip**: rejeita paths absolutos e `..`
- Filtra artefatos macOS: `__MACOSX/`, `._*`
- Extensões permitidas: `.png`, `.jpg`, `.jpeg`, `.webp`

### 5.2 Mapeamento de Nomes de Saída

Função: `_mapped_output_name()`

| Asset | Detecção | Saída |
|-------|----------|-------|
| Frame | `frame` no path/nome | `frame.webp` |
| Background | `background`/`backgrounds` | `body-bg.webp` |
| Logo | `logo` no filename | `logo.webp` |
| Wild | `wild` no path/nome | `symbols/W.webp` |
| High/Scatter | `high`/`scatter` | `symbols/H<N>.webp` |
| Low | `low` | `symbols/L<N>.webp` |
| Outro | fallback | `<nome_sanitizado>.webp` |

Duplicatas recebem sufixo numérico: `frame_2.webp`, `logo_3.webp`, etc.

Índices são extraídos do nome do arquivo via regex (ex: `icon_high_5.png` → `H5`).

### 5.3 Reempacotamento

- ZIP com compressão DEFLATE
- Todos os arquivos dentro de `images/` (ex: `images/symbols/L1.webp`, `images/frame.webp`)
- Tolerância a falhas parciais: se ≥1 imagem processar com sucesso, retorna o ZIP. Se todas falharem → HTTP 500.

---

## 6. Modelo ML — BiRefNet

### Especificações

| Propriedade | Valor |
|-------------|-------|
| Modelo | `ZhengPeng7/BiRefNet` (Hugging Face) |
| Tipo | Segmentação de imagem (foreground/background) |
| Input | 1024×1024, normalizado (ImageNet) |
| Output | Máscara probabilística via sigmoid |
| Carregamento | `AutoModelForImageSegmentation` com `trust_remote_code=True` |
| Device | CUDA se disponível, senão CPU |
| Precisão | float32 ou float16 (auto-detect do modelo) |
| Memória GPU | ~2GB |

### Ciclo de Vida

1. Carregado **uma única vez** no startup do módulo `image_service.py`
2. Colocado em modo `.eval()` (sem dropout/batchnorm training)
3. Movido para device (CUDA/CPU)
4. Se falhar o carregamento, `_BRIA_MODEL = None` e todo processamento usa fallback

### Otimização PyTorch

```python
torch.set_float32_matmul_precision("high")  # matmul otimizado
```

---

## 7. Compressão WebP

### Configuração via Variáveis de Ambiente

| Variável | Default | Descrição |
|----------|---------|-----------|
| `OUTPUT_WEBP_QUALITY` | 72 | Qualidade lossy (1-100) |
| `OUTPUT_WEBP_METHOD` | 6 | Método de compressão (0=rápido, 6=melhor ratio) |
| `OUTPUT_WEBP_ALPHA_QUALITY` | 88 | Qualidade do canal alpha (1-100) |

### Comportamento

- Imagens com transparência → `save_rgba_as_webp()` (lossy + alpha)
- Imagens sem transparência → `save_rgb_as_webp()` (lossy, sem alpha = menor)
- Modo `lossless=False`, `exact=False` (permite otimizações do encoder)

---

## 8. Variáveis de Ambiente

### Servidor

| Variável | Default | Descrição |
|----------|---------|-----------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Porta |
| `BG_API_RELOAD` | `1` | Auto-reload (só via `python app/main.py`) |
| `LOG_LEVEL` | `INFO` | Nível de log: DEBUG, INFO, WARNING |

### Compressão WebP

| Variável | Default | Descrição |
|----------|---------|-----------|
| `OUTPUT_WEBP_QUALITY` | `72` | Qualidade lossy |
| `OUTPUT_WEBP_METHOD` | `6` | Método de compressão |
| `OUTPUT_WEBP_ALPHA_QUALITY` | `88` | Qualidade alpha |

### Frame Interior (Parâmetros de Limpeza)

| Variável | Default | Descrição |
|----------|---------|-----------|
| `FRAME_MARGIN_FRAC` | `0.07` | Margem da borda da imagem (0-0.5) |
| `FRAME_CELL_INSET_FRAC` | `0.055` | Encolhimento por célula no grid |
| `FRAME_INTERIOR_PER_CHANNEL` | `200` | Min R/G/B por canal no interior |
| `FRAME_INTERIOR_MEAN_MIN` | `168` | Luminância média mínima |
| `FRAME_INTERIOR_CHROMA_MAX` | `52` | Saturação máxima (max-min canais) |
| `FRAME_INTERIOR_WEAKEST_CH_MIN` | `140` | Min para o canal mais fraco |
| `FRAME_REEL_INSET_X_FRAC` | `0.08` | Inset horizontal (reels) |
| `FRAME_REEL_INSET_Y_FRAC` | `0.12` | Inset vertical (reels) |
| `FRAME_MOLDURA_INNER_FRAC` | `0.11` | Tamanho interno (moldura) |
| `FRAME_DEEP_INSET_FRAC` | `0.20` | Inset profundo para erosão segura |
| `FRAME_FORCE_KIND` | *(vazio)* | Override: `grid_9`, `reels_3`, `moldura` |

---

## 9. Logging

### Loggers

| Logger | Uso |
|--------|-----|
| `bg_api.main` | Startup, middleware |
| `bg_api.routes` | Endpoints HTTP |
| `bg_api.image_service` | Processamento de imagem |
| `bg_api.zip_service` | Operações ZIP |

### Middleware de Logging

Cada request é logado com:
- Entrada: `→ METHOD PATH`
- Saída: `← METHOD PATH STATUS_CODE (ELAPSED)`
- Exceção: `✗ METHOD PATH — erro (ELAPSED)` + traceback

### Loggers Silenciados

- `PIL` → WARNING (evita spam de debug)
- `onnxruntime` → WARNING
- `uvicorn.access` → INFO (mantido)

---

## 10. Segurança

- **Validação de input:** content-type whitelist, range de threshold, rejeição de arquivos vazios
- **Zip Slip:** validação de paths (absolutos, `..`, resolução para dentro do target)
- **Artefatos macOS:** filtrados automaticamente (`__MACOSX`, `._*`)
- **Sem persistência:** todo I/O em `tempfile.TemporaryDirectory` (limpeza automática)
- **Erros genéricos:** respostas HTTP não vazam paths internos; detalhes ficam nos logs

---

## 11. Dependências

| Pacote | Propósito |
|--------|-----------|
| `fastapi` | Framework web |
| `uvicorn[standard]` | Servidor ASGI (com extensões C) |
| `pillow` | I/O de imagens (PIL) |
| `python-multipart` | Parsing de multipart/form-data |
| `torch` | Framework deep learning (BiRefNet) |
| `torchvision` | Transforms e utilidades de visão |
| `transformers` | Carregamento de modelos Hugging Face |
| `scipy` | Operações morfológicas (ndimage) |
| `numpy` | Operações vetorizadas em arrays |
| `kornia` | Lib de visão computacional (presente mas sem uso ativo) |

---

## 12. Execução

### Desenvolvimento Local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### RunPod / Servidor

```bash
./setup_runpod.sh   # git pull + venv + pip install
./run_api.sh         # executa via .venv/bin/python
```

> **Importante:** Sempre usar o Python do `.venv`, nunca o `uvicorn` global do sistema.

### Startup

1. Logging configurado (dict config)
2. FastAPI app criado com metadata
3. Router incluído
4. Evento startup: log de inicialização
5. BiRefNet carregado para GPU/CPU (ou falha graceful)
6. Uvicorn escuta na porta configurada

---

## 13. Problemas Conhecidos

### ~~Bug: `/health` endpoint quebrado~~ (corrigido)

Migrado de `_REMBG_AVAILABLE` para `_BRIA_MODEL is not None`. Endpoint agora retorna `{"status": "ok", "birefnet": true/false}`.

### Dependência sem uso: `kornia`

Listada em `requirements.txt` mas sem import em nenhum arquivo do projeto.

### README desatualizado

O README ainda referencia `rembg` como motor de remoção de fundo. O projeto agora usa `BiRefNet`.

---

## 14. Fluxo End-to-End

### Imagem Individual

```
Request (multipart: file + threshold)
  │
  ▼
Validação (content-type, threshold, arquivo vazio)
  │
  ▼
Classificação do asset pelo nome
  │
  ├─ Background? ──► compress_to_webp_only() ──► WebP stream
  │
  ├─ Frame? ──► _detect_frame_kind()
  │              ──► _remove_frame_by_components()
  │              ──► _cleanup_frame_interior_by_kind()
  │              ──► save_rgba_as_webp() ──► WebP stream
  │
  └─ Símbolo/Logo? ──► _bria_remove() (BiRefNet)
                        OU _remove_by_threshold() (fallback)
                        ──► save_rgba_as_webp() ──► WebP stream
```

### ZIP de Assets

```
Request (multipart: file.zip + threshold)
  │
  ▼
Validação (threshold, ZIP vazio)
  │
  ▼
Extração segura para temp dir
  │
  ▼
Listagem de imagens (recursiva, filtrada)
  │
  ▼
Para cada imagem:
  ├─ _mapped_output_name() → nome padronizado
  ├─ Classificação → processamento adequado
  └─ Log de resultado (✓ ou ✗)
  │
  ▼
Reempacotamento em ZIP (images/ prefix, DEFLATE)
  │
  ▼
ZIP stream como resposta
```
