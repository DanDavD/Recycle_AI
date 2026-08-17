# Recycle_AI

Detección de residuos reciclables (plástico, papel, metal) en tiempo real usando YOLOv8, entrenado sobre el dataset "Plastic-Paper-Metal" de Roboflow.

## Especificaciones

- **Hardware**: Laptop Gigabyte G5 (GPU RTX 4060 8 GB, i5-11va, 16GB RAM)
- **Dataset**: Plastic-Paper-Metal (Roboflow), versión 2 — 2234 imágenes, 3 clases
- **Split**: 1956 train / 185 valid / 93 test
- **Objetivo**: Modelo de detección con >90% de precisión

## Estructura del proyecto

```
Recycle_AI/
├── venv/
├── datasets/                     # se descarga con el script 04 (no va en git)
│   └── plastic-paper-metal/
│       ├── train/ valid/ test/   # cada uno con images/ y labels/
│       └── data.yaml
├── models/                       # pesos entrenados (no van en git)
├── runs/detect/<run>/            # gráficas y métricas de cada entrenamiento
├── scripts/
│   ├── 01_setup_environment.py
│   ├── 02_verify_gpu.py
│   ├── 03_test_camera.py
│   ├── 04_load_dataset.py
│   ├── 05_train_model.py
│   └── 06_run_bin.py             # inferencia en vivo del basurero
├── .env.example                  # plantilla: copiar a .env y poner la API key
├── requirements.txt
└── README.md
```

## Setup

**1. Entorno virtual y dependencias** (con el Python del sistema, fuera de cualquier venv):

```powershell
python scripts\01_setup_environment.py
.\venv\Scripts\Activate.ps1
```

**2. API key de Roboflow** — copia la plantilla y pega tu *Private API Key*
(app.roboflow.com → Settings → API Keys):

```powershell
copy .env.example .env
```

**3. Verificaciones** (opcional pero recomendado):

```powershell
python scripts\02_verify_gpu.py      # que torch vea la GPU vía CUDA
python scripts\03_test_camera.py     # cámara + YOLO pre-entrenado en vivo (q para salir)
```

**4. Descargar el dataset**:

```powershell
python scripts\04_load_dataset.py
```

**5. Entrenar**:

```powershell
python scripts\05_train_model.py
```

Los pesos finales quedan en `models\<run>.pt` y las gráficas/métricas en `runs\detect\<run>\`.

Por defecto entrena con rotación aleatoria (`--degrees 180`) y volteo vertical
(`--flipud 0.5`), para que reconozca los objetos boca abajo y en ángulos raros —
los valores por defecto de ultralytics no cubren eso.

**6. Correr el basurero**:

```powershell
python scripts\06_run_bin.py
```

Toma el modelo más reciente de `models\`, detecta por cámara y decide una sola
compuerta. Cuando hay varios objetos en el frame elige uno con `--select`
(`conf` = el más confiable, `area` = el más grande/cercano, `center` = el más
centrado). Espera `--stable N` frames seguidos del mismo material antes de actuar,
para que una detección parpadeante no active el servo de ida y vuelta.

El código del servo va en la función `abrir_compuerta()` del script, que ahora
solo imprime la decisión por consola para poder probar la lógica sin hardware.
Las clases se traducen a número de compuerta en el diccionario `COMPARTMENTS`.

## Usar otro dataset de Roboflow

En la página del dataset: **Download Dataset → Show download code**, guarda ese
snippet en un archivo y pásalo al script; no hay que tocar código:

```powershell
python scripts\04_load_dataset.py --snippet snippet.txt
python scripts\04_load_dataset.py --snippet -      # o pegarlo directo (Ctrl+Z, Enter para terminar)
python scripts\05_train_model.py --dataset <nombre-del-proyecto>
```

Para que quede fijo en el repo, añade una entrada en el diccionario `DATASETS` de
[scripts/04_load_dataset.py](scripts/04_load_dataset.py).

## Notas

- **No instalar `opencv-python-headless`**: sobrescribe el módulo `cv2` de
  `opencv-python` con una build sin GUI y `cv2.imshow` deja de funcionar (rompe el
  script 03). `roboflow` lo arrastra como dependencia, por eso el script 01 lo
  desinstala al final.
- Si el entrenamiento da `CUDA out of memory`, baja el batch:
  `python scripts\05_train_model.py --batch 8`.
- `--resume` reanuda el último entrenamiento interrumpido desde
  `runs\detect\<run>\weights\last.pt`.
