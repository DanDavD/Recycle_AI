# ☁️ Recycle_AI — Google Colab

> **Antes de empezar:** Activar GPU: Entorno de ejecución → Cambiar tipo → **GPU T4**

---

### Celda 1 — Clonar repo e instalar dependencias
```python
!git clone https://github.com/DanDavD/Recycle_AI.git /content/Recycle_AI
%cd /content/Recycle_AI
!pip install roboflow python-dotenv --quiet
```

---

### Celda 2 — Configurar API Key de Roboflow
```python
import os

API_KEY = "loIOyQL5vSDizjI7ZiT8"
os.environ["ROBOFLOW_API_KEY"] = API_KEY

with open(".env", "w") as f:
    f.write(f"ROBOFLOW_API_KEY={API_KEY}\n")

print("✅ Listo.")
```

---

### Celda 3 — Descargar datasets
```python
!python scripts/04_load_dataset.py
```

---

### Celda 4 — Combinar y balancear clases
```python
!python scripts/combine_datasets.py
```

---

### Celda 5 — Entrenar (modelo nuevo desde cero)
```python
!python scripts/05_train_model.py --dataset combined --epochs 100
```

---

### Celda 6 — Descargar el modelo generado a tu PC
```python
from pathlib import Path
from google.colab import files

modelos = sorted(Path("models").glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
if modelos:
    print(f"Descargando: {modelos[0].name}")
    files.download(str(modelos[0]))
```
