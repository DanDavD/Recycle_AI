# ☁️ Entrenar Recycle_AI en Google Colab (GPU Gratis)

Si no tienes GPU o quieres entrenar en la nube sin configurar nada, puedes usar Google Colab con GPU T4 gratuita.

---

## Paso 1 — Abrir una sesión con GPU

1. Ve a [colab.research.google.com](https://colab.research.google.com)
2. Crea un nuevo cuaderno (Archivo → Nuevo cuaderno)
3. Activa la GPU: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**

---

## Paso 2 — Clonar el repo e instalar dependencias

```python
!git clone https://github.com/Hectovargas/Recycle_AI.git /content/Recycle_AI
%cd /content/Recycle_AI
!pip install -r requirements.txt
```

---

## Paso 3 — Configurar la API Key de Roboflow

```python
import os

API_KEY = "TU_API_KEY_AQUI"   # app.roboflow.com → Settings → API Keys
os.environ["ROBOFLOW_API_KEY"] = API_KEY

with open(".env", "w") as f:
    f.write(f"ROBOFLOW_API_KEY={API_KEY}\n")

print("✅ API Key configurada.")
```

---

## Paso 4 — Descargar los datasets

```python
!python scripts/04_load_dataset.py
```

Descarga los 7 datasets registrados (Papel, Plástico, Metal, Botellas, Latas...).
Si uno ya está descargado, lo detecta y lo salta.

---

## Paso 5 — Combinar y balancear clases

```python
!python scripts/combine_datasets.py
```

Genera `datasets/combined/` con split 80% train / 10% val / 10% test
e imprime la tabla de distribución final (🥫 Metal / 📄 Papel / 🧴 Plástico).

---

## Paso 6 — Entrenar el modelo

**Opción A — Modelo nuevo desde cero:**
```python
!python scripts/05_train_model.py --dataset combined --epochs 100
```

**Opción B — Fine-Tuning sobre el modelo anterior (subir el `.pt` previo primero):**
```python
from google.colab import files
uploaded = files.upload()   # selecciona tu .pt desde tu computadora

import shutil
shutil.move(list(uploaded.keys())[0], "models/")

!python scripts/05_train_model.py --dataset combined --epochs 100 --fine-tune
```

---

## Paso 7 — Descargar el modelo generado

```python
from pathlib import Path
from google.colab import files

modelos = sorted(Path("models").glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
if modelos:
    print(f"Descargando: {modelos[0].name}")
    files.download(str(modelos[0]))
```

El archivo `.pt` se descargará directamente a tu computadora.
Cópialo a la carpeta `models/` del repo local para usarlo con `06_run_bin.py` o `07_run_bin_web.py`.

---

## Notas

- La sesión gratuita de Colab tiene un límite de ~4–5 horas continuas. Si se corta el entrenamiento, guarda el `last.pt` antes y reanuda con `--resume`.
- Los datasets se pierden al cerrar la sesión de Colab. El paso 4 siempre hay que repetirlo.
- Para sesiones largas sin interrupciones considera [Colab Pro](https://colab.research.google.com/signup).
