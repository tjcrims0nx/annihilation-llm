import importlib.util
import subprocess
import sys


def main():
    is_gpu = "--gpu" in sys.argv
    needs_reinstall = False
    needs_install = False

    try:
        import torch

        if is_gpu and not torch.cuda.is_available():
            print(
                "CRITICAL: GPU detected but PyTorch is CPU-only! Forcing CUDA reinstall...",
                flush=True,
            )
            needs_reinstall = True

        # Test if torchvision is available
        if importlib.util.find_spec("torchvision") is None:
            raise ImportError("torchvision not found")

        # Test if annihilate is available
        if importlib.util.find_spec("annihilate") is None:
            raise ImportError("annihilate not found")
    except ImportError as e:
        print(f"Missing dependency detected ({e}). Installing...", flush=True)
        needs_install = True

    if needs_reinstall:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            ".",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu121",
            "--reinstall",
        ]
        print(f"Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        print("CUDA PyTorch and dependencies reinstallation complete.", flush=True)
    elif needs_install:
        cmd = [sys.executable, "-m", "pip", "install", "."]
        if is_gpu:
            cmd.extend(["--extra-index-url", "https://download.pytorch.org/whl/cu121"])
        print(f"Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        print("Dependencies installation complete.", flush=True)
    else:
        print(
            "Environment verification passed! All dependencies correctly installed.",
            flush=True,
        )


if __name__ == "__main__":
    main()
