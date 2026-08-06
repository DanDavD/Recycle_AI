"""
Script 03: prueba la camara con OpenCV y corre deteccion en vivo con un YOLOv8
pre-entrenado (COCO, generico) para validar que camara + GPU + ultralytics
funcionan juntos, antes de entrenar el modelo custom de reciclaje.

Ejecutar con el venv activado:
    python scripts/03_test_camera.py

Controles:
    q -> salir
"""
import cv2
from ultralytics import YOLO

CAMERA_INDEX = 0
# Modelo pre-entrenado de YOLOv8 (80 clases de COCO), solo para probar el
# pipeline completo. Se descarga automaticamente la primera vez que se usa.
PRETRAINED_MODEL = "yolov8n.pt"


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la camara (indice {CAMERA_INDEX}).")
        print("Si tienes varias camaras, prueba cambiando CAMERA_INDEX a 1, 2, etc.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Camara abierta: {width}x{height}")

    print("[..] Cargando YOLOv8 pre-entrenado (se descarga la primera vez)...")
    model = YOLO(PRETRAINED_MODEL)
    print("[OK] Modelo cargado. Presiona 'q' en la ventana de video para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] No se pudo leer el frame de la camara.")
            break

        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("Recycle_AI - Test camara + YOLO (q para salir)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
