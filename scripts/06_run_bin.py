"""
Script 06: inferencia en vivo para el basurero inteligente.

Mira por la camara, detecta el material y decide UNA sola compuerta que abrir.

Resuelve los dos requisitos que no son de dataset sino de codigo:
  - Si hay varios objetos en el frame, elige uno solo (ver --select).
  - Cuando decide, siempre devuelve un material concreto, nunca algo ambiguo.

Requiere un modelo ya entrenado con scripts/05_train_model.py:
    python scripts/06_run_bin.py
    python scripts/06_run_bin.py --model models/plastic-paper-metal_yolov8n.pt
    python scripts/06_run_bin.py --select area --conf 0.35
    python scripts/06_run_bin.py --no-view          # sin ventana, para el basurero real

Controles (con ventana):
    q -> salir
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

load_dotenv(ROOT_DIR / ".env")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))

# Confianza minima para actuar. Por debajo de esto el basurero no abre nada y
# sigue mirando: es preferible esperar un frame mas que mandar un plastico al
# tacho del papel, porque contaminar la mezcla arruina el reciclaje del lote.
# Si prefieres que responda siempre sin importar la duda, baja esto a 0.01.
DEFAULT_CONF = 0.40

# Frames seguidos con el mismo material antes de abrir la compuerta. Evita que
# una deteccion parpadeante active el servo de ida y vuelta.
DEFAULT_STABLE_FRAMES = 3

# Segundos de espera despues de abrir, para no reaccionar al mismo objeto dos veces.
DEFAULT_COOLDOWN = 3.0

# Mapa material -> compuerta. Las claves se comparan en minusculas contra los
# nombres de clase del modelo (los de data.yaml), buscando coincidencia parcial,
# asi que funciona igual con "plastic", "Plastic" o "plastic-bottle".
COMPARTMENTS = {
    "plastic": 1,
    "plastico": 1,
    "paper": 2,
    "papel": 2,
    "carton": 2,
    "metal": 3,
    "lata": 3,
    "can": 3,
}


def resolve_compartment(class_name: str) -> int | None:
    """Traduce el nombre de clase del modelo al numero de compuerta."""
    nombre = class_name.lower()
    for clave, compuerta in COMPARTMENTS.items():
        if clave in nombre:
            return compuerta
    return None


def abrir_compuerta(material: str, compuerta: int) -> None:
    """Punto de conexion con el hardware del basurero.

    Aqui va el codigo del servo. Dos formas tipicas:

      * Raspberry Pi (GPIO directo):
            from gpiozero import Servo
            servo = Servo(17)
            servo.max(); time.sleep(1); servo.mid()

      * Arduino por USB (el PC hace la vision, el Arduino mueve los servos):
            import serial
            arduino = serial.Serial("COM3", 9600)
            arduino.write(f"{compuerta}\\n".encode())

    Por ahora solo lo reporta por consola para poder probar la logica sin el
    hardware montado.
    """
    print(f">>> ABRIR COMPUERTA {compuerta}  ({material})")


def pick_detection(boxes, strategy: str, frame_shape):
    """De todas las detecciones del frame devuelve UNA sola.

    Con varios objetos delante de la camara hay que quedarse con uno, porque
    solo se puede abrir una compuerta a la vez.

      conf   -> la deteccion mas confiable (default, la mas predecible)
      area   -> la caja mas grande, suele ser el objeto mas cercano a la camara
      center -> la mas cercana al centro del frame, el que va cayendo
    """
    if strategy == "conf":
        return max(boxes, key=lambda b: float(b.conf[0]))

    if strategy == "area":
        def area(b):
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            return (x2 - x1) * (y2 - y1)
        return max(boxes, key=area)

    # center
    h, w = frame_shape[:2]
    cx_frame, cy_frame = w / 2, h / 2

    def dist_al_centro(b):
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (cx - cx_frame) ** 2 + (cy - cy_frame) ** 2

    return min(boxes, key=dist_al_centro)


def find_model(explicit: str | None) -> Path:
    if explicit:
        ruta = Path(explicit)
        if not ruta.exists():
            sys.exit(f"No encuentro el modelo: {ruta}")
        return ruta

    entrenados = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not entrenados:
        sys.exit(
            f"No hay ningun modelo entrenado en {MODELS_DIR}\n"
            f"Entrena primero:  python scripts/05_train_model.py"
        )
    print(f"[OK] Usando el modelo mas reciente: {entrenados[0].name}")
    return entrenados[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia en vivo del basurero inteligente")
    parser.add_argument("--model", default=None, help="ruta al .pt (por defecto: el mas reciente de models/)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confianza minima para actuar")
    parser.add_argument(
        "--select",
        choices=["conf", "area", "center"],
        default="conf",
        help="como elegir un solo objeto cuando hay varios en el frame",
    )
    parser.add_argument("--stable", type=int, default=DEFAULT_STABLE_FRAMES,
                        help="frames seguidos con el mismo material antes de abrir")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN,
                        help="segundos de pausa despues de abrir una compuerta")
    parser.add_argument("--no-view", action="store_true", help="no abrir ventana de video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(str(find_model(args.model)))
    print(f"[OK] Clases del modelo: {model.names}")

    # Avisar de clases sin compuerta asignada antes de arrancar, no a media corrida.
    sin_mapear = [n for n in model.names.values() if resolve_compartment(n) is None]
    if sin_mapear:
        print(f"[!!] Sin compuerta asignada: {sin_mapear} -- agregalas a COMPARTMENTS en este script")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(
            f"[ERROR] No se pudo abrir la camara (indice {CAMERA_INDEX}).\n"
            f"Prueba con CAMERA_INDEX=1 en el .env"
        )
    print(f"[OK] Camara {CAMERA_INDEX} abierta. Ctrl+C para salir." if args.no_view
          else f"[OK] Camara {CAMERA_INDEX} abierta. Presiona 'q' en la ventana para salir.")

    ultimo_material: str | None = None
    frames_seguidos = 0
    listo_en = 0.0  # timestamp a partir del cual se puede volver a actuar

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] No se pudo leer el frame de la camara.")
                break

            results = model.predict(frame, conf=args.conf, verbose=False)
            boxes = results[0].boxes

            material = None
            if len(boxes) > 0:
                box = pick_detection(boxes, args.select, frame.shape)
                material = model.names[int(box.cls[0])]
                confianza = float(box.conf[0])

            # Contar frames consecutivos del mismo material.
            if material is not None and material == ultimo_material:
                frames_seguidos += 1
            else:
                frames_seguidos = 1 if material is not None else 0
            ultimo_material = material

            if material is not None and frames_seguidos >= args.stable and time.time() >= listo_en:
                compuerta = resolve_compartment(material)
                if compuerta is None:
                    print(f"[!!] '{material}' no tiene compuerta asignada, no abro nada")
                else:
                    print(f"[OK] {material} ({confianza:.0%}) estable en {frames_seguidos} frames")
                    abrir_compuerta(material, compuerta)
                listo_en = time.time() + args.cooldown
                frames_seguidos = 0
                ultimo_material = None

            if not args.no_view:
                anotado = results[0].plot()
                estado = f"{material} ({confianza:.0%})" if material else "esperando objeto"
                if time.time() < listo_en:
                    estado = "cooldown"
                cv2.putText(anotado, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Recycle_AI - Basurero inteligente (q para salir)", anotado)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[OK] Detenido por el usuario")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
