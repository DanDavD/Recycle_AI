"""
Script 02: verifica que PyTorch detecta la GPU (RTX 3060) vía CUDA.

Ejecutar con el venv activado:
    python scripts/02_verify_gpu.py
"""
import torch


def main() -> None:
    print(f"Version de PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[ERROR] PyTorch no detecta ninguna GPU.")
        print("Revisa que los drivers de NVIDIA esten actualizados (nvidia-smi)")
        print("y que instalaste torch con --index-url .../cu121 (no la version CPU).")
        return

    print(f"Version de CUDA (build de torch): {torch.version.cuda}")
    print(f"Cantidad de GPUs: {torch.cuda.device_count()}")
    print(f"GPU activa: {torch.cuda.get_device_name(0)}")

    # Prueba real: mover un tensor a la GPU y operar
    x = torch.rand(3, 3).cuda()
    y = x @ x
    print(f"\nTensor de prueba en {y.device}:\n{y}")
    print("\n[OK] La GPU esta lista para entrenar.")


if __name__ == "__main__":
    main()
