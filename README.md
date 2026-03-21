# bg-api

API **FastAPI** para processar imagens de jogos (remoção de fundo com **rembg** / fallback por threshold), gerar **WebP** e empacotar um **ZIP** de saída com nomes no padrão do tema (L/H/W).

**Não grava uploads nem resultados no disco do projeto** — só usa diretórios temporários do sistema e devolve o arquivo na resposta HTTP.

## Estrutura

```
bg-api/
├── app/
│   ├── main.py              # FastAPI + uvicorn
│   ├── routes/
│   │   └── process.py       # POST /process-image, /process-zip
│   └── services/
│       ├── image_service.py # WebP, rembg, fundos, frames
│       └── zip_service.py   # extrai ZIP em memória, mapeia nomes, recompacta
├── requirements.txt
├── setup_runpod.sh   # git pull + venv + pip (RunPod / servidor)
├── run_api.sh        # uvicorn via .venv (não uses o uvicorn global)
└── README.md
```

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Na primeira execução o **rembg** pode baixar o modelo ONNX (~centenas de MB).

## Executar

```bash
# Na raiz, com venv ativo:
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Garantir o mesmo interpretador (evita “python-multipart” em pods com uvicorn global):
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# ou
python app/main.py
```

**RunPod / Docker com `uvicorn` no PATH do sistema:** esse binário **não** usa o teu `.venv`. Usa `./run_api.sh` ou `.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`.

### RunPod (rápido)

```bash
./setup_runpod.sh
./run_api.sh
```

Variáveis úteis:

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `HOST` | `0.0.0.0` | Host do uvicorn (`python app/main.py`) |
| `PORT` | `8000` | Porta |
| `BG_API_RELOAD` | `1` | `1` ativa reload ao rodar `main.py` |
| `OUTPUT_WEBP_QUALITY` | `72` | Qualidade WebP (1–100) |
| `OUTPUT_WEBP_METHOD` | `6` | Método de compressão WebP (0–6) |
| `OUTPUT_WEBP_ALPHA_QUALITY` | `88` | Qualidade do canal alpha |
| `FRAME_MARGIN_FRAC` | `0.07` | Margem em relação à borda da textura (0–0.5) |
| `FRAME_CELL_INSET_FRAC` | `0.055` | No grid 3×3, encolhe cada célula (menor = mais cobertura contra resíduo nas bordas) |
| `FRAME_INTERIOR_PER_CHANNEL` | `200` | Mínimo R/G/B no interior (mais baixo que o threshold global) |
| `FRAME_INTERIOR_MEAN_MIN` | `168` | Luminância média mínima para considerar “resto de fundo” |
| `FRAME_INTERIOR_CHROMA_MAX` | `52` | Croma máximo (max−min dos canais) para brancos levemente azulados |
| `FRAME_INTERIOR_WEAKEST_CH_MIN` | `140` | Canal mais escuro ainda “claro” o suficiente para ser resíduo |
| `FRAME_FORCE_KIND` | *(vazio)* | Força tipo: `grid_9`, `reels_3` ou `moldura` (se o nome do ficheiro não bater) |
| `FRAME_REEL_INSET_X_FRAC` | `0.08` | Faixas de reel: inset horizontal em cada coluna |
| `FRAME_REEL_INSET_Y_FRAC` | `0.12` | Faixas de reel: inset vertical |
| `FRAME_MOLDURA_INNER_FRAC` | `0.11` | Moldura única: quanto “entrar” a partir do retângulo útil |

### Frames (molduras / grids)

Arquivos cujo **caminho ou nome** contém `frame` recebem pós-processamento extra no **interior** (remove restos de fundo claro que o rembg deixa nas “janelas”).

Há **3 modos**, escolhidos por palavras-chave no path/nome (case-insensitive):

| Modo | O que limpa | Palavras-chave (exemplos) |
|------|-------------|---------------------------|
| **Grid 3×3** | 9 retângulos (centro de cada célula) | `grid`, `3x3`, `nine`, `slotgrid`, `grid9`, `9grid` |
| **3 reels** | 3 faixas verticais (2 divisórias ≈ 3 colunas) | `reels3`, `3reel`, `dividers`, `triple_strip`, `3_column`, … |
| **Moldura** | um único retângulo interno | padrão se só aparecer `frame`; ou `moldura`, `inner_square`, `portal`, … |

**Dica:** renomeie no ZIP algo como `ui/grid_frame_3x3.png` para ativar o modo 9 células.

Se o modo **grid** não estiver a ativar (ficheiro sem `grid` no path), use `FRAME_FORCE_KIND=grid_9`.

A limpeza do interior usa critérios **extra** (luminância + baixo croma) para apanhar restos branco-azulados que o rembg deixa. Se ainda sobrar mancha, baixe `FRAME_INTERIOR_MEAN_MIN` (ex.: `150`) ou suba `FRAME_INTERIOR_CHROMA_MAX` (ex.: `60`).

## Endpoints

| Método | Caminho | Descrição |
|--------|---------|-----------|
| `GET` | `/health` | Status da API |
| `POST` | `/process-image` | multipart: `file` + opcional `threshold` (0–255) → `image/webp` |
| `POST` | `/process-zip` | multipart: `file` (.zip) + opcional `threshold` → `application/zip` (`images.zip`) |

### Exemplo (ZIP)

```bash
curl -sS -X POST "http://127.0.0.1:8000/process-zip" \
  -F "file=@seu_pacote.zip" \
  -F "threshold=245" \
  -o images.zip
```

### Mapeamento de nomes no ZIP de saída

Heurística pelo caminho/nome do arquivo dentro do ZIP:

- **low** → `L` + índice (ex.: `L1.webp`)
- **high** / **scatter** → `H` + índice
- **wild** → `W` (ou `W` + índice se houver número)

Documentação interativa: `http://127.0.0.1:8000/docs`
