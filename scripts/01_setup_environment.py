"""
Script 01: crea el entorno virtual e instala todas las dependencias del proyecto.

Ejecutar con el Python del SISTEMA (fuera de cualquier venv), desde la raíz del proyecto:
    python scripts/01_setup_environment.py
"""
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT_DIR / "venv"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"

# Índice de PyTorch con soporte CUDA 12.1 (compatible con RTX 3060 / Ampere)
TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"

CORE_PACKAGES = [
    "ultralytics",
    "opencv-python",
    "numpy",
    "pandas",
    "requests",
]


def get_venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd: list) -> None:
    print(f"[..] Ejecutando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def create_virtual_environment() -> None:
    if VENV_DIR.exists():
        print(f"[OK] El entorno virtual ya existe en {VENV_DIR}")
        return
    print(f"[..] Creando entorno virtual en {VENV_DIR}")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    print("[OK] Entorno virtual creado")


def main() -> None:
    create_virtual_environment()
    python_bin = str(get_venv_python())

    run([python_bin, "-m", "pip", "install", "--upgrade", "pip"])

    # PyTorch con soporte CUDA (necesario para usar la RTX 3060 al entrenar)
    run([
        python_bin, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", TORCH_CUDA_INDEX_URL,
    ])

    # Resto de dependencias del proyecto
    run([python_bin, "-m", "pip", "install", *CORE_PACKAGES])

    # Congelar versiones exactas instaladas
    print(f"[..] Guardando dependencias en {REQUIREMENTS_FILE}")
    with open(REQUIREMENTS_FILE, "w", encoding="utf-8") as f:
        subprocess.run([python_bin, "-m", "pip", "freeze"], stdout=f, check=True)
    print("[OK] requirements.txt actualizado")

    print("\nListo. Activa el entorno virtual con:")
    if sys.platform == "win32":
        print(r"    .\venv\Scripts\Activate.ps1")
    else:
        print("    source venv/bin/activate")


if __name__ == "__main__":
    main()
