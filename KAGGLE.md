# 🦅 Recycle_AI — Guía de Entrenamiento en Kaggle Notebooks (Rama V2)

> **Ventajas en Kaggle:**
> - **30 horas gratis de GPU por semana** (NVIDIA Tesla T4 x2 o P100).
> - Sesiones de hasta **12 horas continuas** sin desconexiones repentinas.

---

## ⚙️ Paso Previo: Configurar el Notebook en Kaggle

1. Ve a [kaggle.com](https://www.kaggle.com/) e inicia sesión.
2. Haz clic en **Create** (arriba a la izquierda) → **New Notebook**.
3. En el panel lateral derecho (`Notebook options`):
   - **Accelerator:** Selecciona **GPU T4 x2** (o GPU P100).
   - **Internet:** Actívalo a **On** *(si te pide verificar número de teléfono, hazlo una sola vez para habilitar internet)*.

---

## 📋 Celdas de Código (Copiar y Pegar)

### Celda 1 — Clonar rama `V2` e instalar dependencias
```python
!git clone -b V2 https://github.com/DanDavD/Recycle_AI.git /kaggle/working/Recycle_AI
%cd /kaggle/working/Recycle_AI
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

### Celda 3b — Incorporar negativos (Rechazo de falsos positivos)
Elige una de las siguientes opciones:

#### Opción A: Descarga automática desde Roboflow (madera, telas, residuos orgánicos)
```python
!python scripts/04_load_dataset.py --download-negatives
```

#### Opción B: Subir tus propias fotos
Puedes arrastrar tus fotos directamente al panel izquierdo de archivos de Kaggle dentro de `Recycle_AI/datasets/negatives/` o usar este código:
```python
import os
os.makedirs("datasets/negatives", exist_ok=True)
print("Arrastra tus fotos a /kaggle/working/Recycle_AI/datasets/negatives/ en el panel izquierdo de Kaggle.")
```

---

### Celda 4 — Combinar y balancear datasets
Recorta las 3 clases (`metal`, `paper`, `plastic`) al mínimo uniforme exacto y añade un 10% de imágenes negativas con etiquetas vacías:
```python
!python scripts/combine_datasets.py --negatives-ratio 0.10
```

---

### Celda 5 — Entrenar modelo YOLOv8
```python
!python scripts/05_train_model.py --dataset combined --epochs 100
```

---

### Celda 6 — Validar rechazo de falsos positivos
Evalúa la tasa de falsas alarmas con tus negativos:
```python
!python scripts/08_validate_negatives.py --source datasets/negatives --conf 0.40
```

---

### Celda 7 — Descargar el modelo entrenado a tu PC
En Kaggle puedes generar un enlace directo de descarga en la misma celda:
```python
from pathlib import Path
from IPython.display import FileLink, display

modelos = sorted(Path("models").glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
if modelos:
    modelo_final = modelos[0]
    print(f"Haz clic en el siguiente enlace para descargar {modelo_final.name}:")
    display(FileLink(str(modelo_final)))
else:
    print("No se encontraron modelos en models/")
```
*(También puedes descargarlo desde el panel lateral derecho de Kaggle en la sección **Output**)*.
