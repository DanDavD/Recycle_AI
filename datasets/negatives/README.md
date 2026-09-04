# 🚫 Imágenes Negativas (Fondo y Objetos Ajenos)

Esta carpeta está destinada a albergar **imágenes negativas** (background images sin cajas ni etiquetas) para entrenar al modelo YOLOv8 a **suprimir todo falso positivo posible**.

## ¿Por qué son necesarias?
YOLO es un detector por propuesta de regiones. Si se entrena únicamente con objetos positivos (`metal`, `paper`, `plastic`), cualquier objeto no visto previamente (madera, tela, basura orgánica, manos, suelo, reflejos) tenderá a ser forzado a la clase con menor distancia de características visuales.

Al incluir entre un **10% y 15%** de imágenes de fondo sin cajas:
1. El modelo aprende explícitamente a asociar estas texturas con la probabilidad de "fondo" (sin objeto).
2. Se evitan aperturas falsas de compuertas en el basurero inteligente.
3. No se altera la definición del problema: `data.yaml` conserva intactas sus 3 clases objetivo (`metal`, `paper`, `plastic`).

## Fuentes recomendadas de negativos
Puedes colocar imágenes sueltas o subcarpetas temáticas dentro de esta carpeta:
- `wood/`: Fotos de madera, palitos, paletas de helado, corcho, virutas, tablas.
- `fabric/`: Telas, ropa, trapos, mascarillas, estambre.
- `organic/`: Cáscaras de fruta, restos de comida, servilletas usadas con grasa, hojas, plantas.
- `station_background/`: Fotos del basurero inteligente tomadas directamente con su cámara:
  - Tolva vacía.
  - Iluminación ambiental diurna y nocturna.
  - Manos humanas depositando objetos (sin el objeto).
  - Sombras y reflejos del entorno real.
- `other/`: Cerámica, piedras, cables, objetos electrónicos pequeños.

## Cómo utilizarlas en el pipeline
1. **Descarga automática (Roboflow Universe):**
   ```bash
   python scripts/04_load_dataset.py --download-negatives
   ```
2. **Extracción desde un dataset existente:**
   ```bash
   python scripts/04_load_dataset.py --extract-negatives <nombre_carpeta_o_dataset>
   ```
3. **Combinación con el dataset general:**
   ```bash
   python scripts/combine_datasets.py --negatives-ratio 0.10
   ```
   *`combine_datasets.py` leerá recursivamente todas las imágenes de esta carpeta y generará los archivos `.txt` vacíos correspondientes en `train/labels`, `valid/labels` y `test/labels`.*
