import importlib.util
import shutil
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
                "CRITICAL: GPU detected but PyTorch is CPU-only! Forcing CUDA reinstall... (NOTE: PyTorch extraction can take 15-20 minutes)",
                flush=True,
            )
            needs_reinstall = True

        # Test if annihilate is available
        if importlib.util.find_spec("annihilate") is None:
            raise ImportError("annihilate not found")
    except ImportError as e:
        print(f"Missing dependency detected ({e}). Installing...", flush=True)
        needs_install = True

    if needs_reinstall or needs_install:
        # Check if 'uv' is available on Windows
        has_uv = shutil.which("uv") is not None

        cmd = []
        if has_uv:
            print(
                "Detected 'uv' package manager. Using fast installation...", flush=True
            )
            cmd = [
                "uv",
                "sync",
                "--link-mode=copy",
            ]
        else:
            cmd = [sys.executable, "-m", "pip", "install", ".", "--no-cache-dir"]
            if is_gpu:
                cmd.extend(["--extra-index-url", "https://download.pytorch.org/whl/cu121"])
            if needs_reinstall:
                cmd.append("--force-reinstall")

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
