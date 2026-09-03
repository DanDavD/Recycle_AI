"""
Script 07: Inferencia del basurero inteligente conectado a CleanCity (reciclaje-inteligente-web).

Reutiliza la lógica de visión, tracking y selección de objetos de scripts/06_run_bin.py
SIN MODIFICAR el archivo original. Cuando se confirma un residuo:
  1. Ejecuta la apertura de compuerta original (servos/hardware).
  2. Envía el evento de clasificación al Backend NestJS via HTTP POST (/clasificacion).
  3. Recibe el token QR firmado criptográficamente y genera el enlace para escanear en la PWA.

Uso:
    python scripts/07_run_bin_web.py
    python scripts/07_run_bin_web.py --model models/tu_modelo.pt --conf 0.45
    python scripts/07_run_bin_web.py --no-view
"""

import os
import sys
import importlib.util
import requests
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Cargar .env de forma nativa o mediante python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000/api/v1")
STATION_ID = os.environ.get("STATION_ID", "")
STATION_TOKEN = os.environ.get("STATION_TOKEN", "tk_prototipo_v1_recycle_ai_2026")

# Mapeo de clases del modelo YOLO al modelo de datos del Backend CleanCity
CATEGORY_MAP = {
    "plastic": "Plástico",
    "plastico": "Plástico",
    "paper": "Papel",
    "papel": "Papel",
    "carton": "Papel",
    "metal": "Metal",
    "lata": "Metal",
    "can": "Metal",
}


def obtener_station_id() -> str | None:
    """Llama al backend para obtener (o crear) la estación demo y devuelve su ID."""
    headers = {"X-Station-Token": STATION_TOKEN}
    try:
        url = f"{BACKEND_URL.rstrip('/')}/clasificacion/demo-station"
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            data = res.json()
            sid = data.get("stationId")
            print(f"[CLEANCITY] ✅ Estación demo lista: {data.get('stationName')} (ID: {sid[:8]}...)")
            return sid
        else:
            print(f"[CLEANCITY WARN] No se pudo obtener estación demo (HTTP {res.status_code}): {res.text}")
    except requests.exceptions.ConnectionError:
        print(f"[CLEANCITY] ⚠️  No se pudo conectar a {BACKEND_URL} al resolver estación demo.")
    except Exception as e:
        print(f"[CLEANCITY ERROR] {e}")
    return None


def reportar_al_backend(material: str, station_id: str, confianza: float = 0.95, peso: float = 0.25):
    """Envía la clasificación al backend de reciclaje-inteligente-web."""
    categoria = CATEGORY_MAP.get(material.lower(), material.capitalize())

    headers = {
        "Content-Type": "application/json",
        "X-Station-Token": STATION_TOKEN,
    }

    payload = {
        "categoria": categoria,
        "confianza": round(float(confianza), 2),
        "peso": peso,
        "stationId": station_id,
    }

    try:
        url = f"{BACKEND_URL.rstrip('/')}/clasificacion"
        res = requests.post(url, json=payload, headers=headers, timeout=4)

        if res.status_code == 201:
            data = res.json()
            qr = data.get("qr", {})
            codigo = qr.get("codigo", "N/A")
            puntos = qr.get("puntos", 0)
            print(f"\n[CLEANCITY BACKEND] ✅ Evento registrado — {categoria}")
            print(f"[CLEANCITY QR]       Código: {codigo} | Puntos: {puntos}")
            print(f"[CLEANCITY QR LINK]  https://api.qrserver.com/v1/create-qr-code/?size=300x300&data={codigo}\n")
            return data
        else:
            print(f"\n[CLEANCITY BACKEND WARN] Respuesta HTTP {res.status_code}: {res.text}\n")
    except requests.exceptions.ConnectionError:
        print(f"\n[CLEANCITY BACKEND] ⚠️  No se pudo conectar a {BACKEND_URL}")
    except Exception as e:
        print(f"\n[CLEANCITY BACKEND ERROR] {e}\n")

    return None


def main():
    # Cargar dinámicamente el script 06 original
    script_06_path = ROOT_DIR / "scripts" / "06_run_bin.py"
    if not script_06_path.exists():
        sys.exit(f"[ERROR] No se encuentra {script_06_path}")

    spec = importlib.util.spec_from_file_location("bin_module", str(script_06_path))
    if spec is None or spec.loader is None:
        sys.exit("[ERROR] No se pudo cargar el módulo de 06_run_bin.py")

    bin_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(bin_module)
    except ModuleNotFoundError as e:
        print(f"\n[ERROR DE ENTORNO VIRTUAL] No se encontró la dependencia: '{e.name}'")
        print("=========================================================================")
        print("Recycle_AI requiere ejecutarse dentro de su entorno virtual de Python.")
        print("Ejecuta en tu terminal:")
        print("   1. Si es la primera vez que configuras Recycle_AI:")
        print("      python scripts/01_setup_environment.py")
        print("   2. Luego activa el entorno virtual:")
        print("      source venv/bin/activate")
        print("   3. Y vuelve a correr:")
        print("      python scripts/07_run_bin_web.py")
        print("=========================================================================\n")
        sys.exit(1)

    print("=================================================================")
    print("  CleanCity EcoGridAI + Recycle_AI (Basurero Inteligente Conectado)")
    print("=================================================================")
    print(f"  Backend URL   : {BACKEND_URL}")
    print(f"  Station Token : {STATION_TOKEN[:8]}... (Configurado)")
    print("=================================================================\n")

    # Resolver el station ID desde el backend (crea la estación demo si no existe)
    station_id: str | None = STATION_ID or obtener_station_id()
    if station_id:
        print(f"[OK] Station ID activo: {station_id[:8]}...\n")
    else:
        print("[WARN] Sin Station ID — las clasificaciones no se enviarán al backend.\n")

    # Guardar referencia a la función abrir_compuerta original
    abrir_compuerta_original = bin_module.abrir_compuerta

    # Interceptar abrir_compuerta con el puente a CleanCity
    def abrir_compuerta_conectada(material: str, compuerta: int, *args, **kwargs) -> None:
        confianza = kwargs.get("confianza", args[0] if len(args) > 0 else 0.95)
        # 1. Ejecutar el comportamiento original del basurero (hardware/servos)
        abrir_compuerta_original(material, compuerta, *args, **kwargs)
        # 2. Enviar la telemetría con la certeza real y generar QR en el backend (solo si hay station_id)
        if station_id:
            reportar_al_backend(material, station_id, confianza=confianza)

    bin_module.abrir_compuerta = abrir_compuerta_conectada

    # Ejecutar el bucle principal de visión artificial
    bin_module.main()


if __name__ == "__main__":
    main()
