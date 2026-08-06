# Recycle_AI

Detección de residuos reciclables (plástico, papel, metal) en tiempo real usando YOLOv8, entrenado sobre el dataset "Plastic-Paper-Metal" de Roboflow.

## Especificaciones

- **Hardware**: Laptop Gigabyte G5 (GPU RTX 3060, i5-11va, 16GB RAM)
- **Dataset**: Plastic-Paper-Metal (Roboflow) — 930 imágenes, 3 clases
- **Objetivo**: Modelo de detección con >90% de precisión
- **Split**: 80% train / 20% test

## Estructura del proyecto

```
proyecto-reciclaje/
├── venv/
├── datasets/
│   └── plastic-paper-metal/
│       ├── images/
│       ├── labels/
│       └── data.yaml
├── scripts/
│   ├── 01_setup_environment.py
│   ├── 02_verify_gpu.py
│   ├── 03_test_camera.py
│   ├── 04_load_dataset.py
│   └── 05_initialize_yolo.py
├── models/
├── requirements.txt
└── README.md
```

## Setup

```powershell
python scripts\01_setup_environment.py
.\venv\Scripts\Activate.ps1
```

Ver el resto de scripts en [scripts/](scripts/) para verificación de GPU, cámara, carga del dataset e inicialización de YOLO.
