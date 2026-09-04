# ☁️ Recycle_AI — Google Colab (Rama V2)

> **Antes de empezar:** Activar GPU: Entorno de ejecución → Cambiar tipo de entorno → **GPU T4**

---

### Celda 1 — Clonar rama `V2` e instalar dependencias
```python
!git clone -b V2 https://github.com/DanDavD/Recycle_AI.git /content/Recycle_AI
%cd /content/Recycle_AI
!pip install roboflow python-dotenv ultralytics --quiet
```

---

### Celda 2 — Configurar API Key de Roboflow
```python
import os

API_KEY = "loIOyQL5vSDizjI7ZiT8"
os.environ["ROBOFLOW_API_KEY"] = API_KEY

with open(".env", "w") as f:
    f.write(f"ROBOFLOW_API_KEY={API_KEY}\n")

print("✅ Clave configurada.")
```

---

### Celda 3 — Descargar datasets positivos
```python
!python scripts/04_load_dataset.py
```

---

### Celda 3b — Incorporar imágenes negativas (Rechazo de Falsos Positivos)
Puedes elegir **una** o **ambas** opciones:

#### Opción A: Descargar dataset de negativos desde Roboflow (madera, telas, orgánico, desechos generales)
```python
!python scripts/04_load_dataset.py --download-negatives
```

#### Opción B: Subir tus propias fotos del basurero/materiales ajenos (madera, tela, piso, manos)
```python
from google.colab import files
import os

os.makedirs("datasets/negatives", exist_ok=True)
print("Sube tus fotos de madera, tela u objetos no reciclables:")
uploaded = files.upload()
for filename in uploaded.keys():
    os.rename(filename, os.path.join("datasets/negatives", filename))
print("✅ Fotos subidas a datasets/negatives/")
```

---

### Celda 4 — Combinar, balancear clases y agregar negativos
El script calcula automáticamente el mínimo de cajas entre las 3 clases (`min(counts.values())`) y recorta las 3 al mismo nivel uniforme, integrando un 10% de imágenes negativas con etiquetas vacías.
```python
!python scripts/combine_datasets.py --negatives-ratio 0.10
```

---

### Celda 5 — Entrenar modelo YOLOv8
```python
!python scripts/05_train_model.py --dataset combined --epochs 100
```

---

### Celda 6 — Validar supresión de falsos positivos
Evalúa el modelo resultante contra el banco de negativos para verificar que la tasa de falsas alarmas sea mínima o nula:
```python
!python scripts/08_validate_negatives.py --source datasets/negatives --conf 0.40
```

---

### Celda 7 — Descargar el modelo generado a tu PC
```python
from pathlib import Path
from google.colab import files

modelos = sorted(Path("models").glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
if modelos:
    print(f"Descargando: {modelos[0].name}")
    files.download(str(modelos[0]))
```
