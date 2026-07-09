import importlib.util
import shutil
import subprocess
import sys
import os


def main():
    is_gpu = "--gpu" in sys.argv
    needs_install = False

    # Check if annihilate and torch are installed
    try:
        # Check torch installation without importing it in this process (which locks the DLL on Windows)
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            raise ImportError("torch not found")
        if importlib.util.find_spec("annihilate") is None:
            raise ImportError("annihilate not found")
    except ImportError as e:
        print(f"Missing dependency detected ({e}). Installing...", flush=True)
        needs_install = True

    if needs_install:
        has_uv = shutil.which("uv") is not None

        if has_uv:
            print(
                "Detected 'uv' package manager. Using fast installation...", flush=True
            )
            env = os.environ.copy()
            if "UV_EXCLUDE_NEWER" in env:
                del env["UV_EXCLUDE_NEWER"]

            cmd = ["uv", "sync", "--no-progress", "--link-mode=copy"]
            print(f"Running: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, env=env)
        else:
            cmd = [sys.executable, "-m", "pip", "install", ".", "--no-cache-dir"]
            if is_gpu:
                cmd.extend(
                    ["--extra-index-url", "https://download.pytorch.org/whl/cu121"]
                )
            print(f"Running: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)

        print("Dependencies installation complete.", flush=True)

    if is_gpu:
        # Use a subprocess to check CUDA availability without loading the DLL into this process
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True,
                text=True,
                check=True
            )
            has_cuda = result.stdout.strip() == "True"
            
            if not has_cuda:
                print(
                    "GPU detected but PyTorch is CPU-only. Installing CUDA version...",
                    flush=True,
                )
                has_uv = shutil.which("uv") is not None
                if has_uv:
                    cmd = [
                        "uv",
                        "pip",
                        "install",
                        "torch",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu121",
                        "--reinstall",
                        "--no-progress",
                    ]
                else:
                    cmd = [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu121",
                        "--force-reinstall",
                        "--no-cache-dir",
                    ]
                print(f"Running: {' '.join(cmd)}", flush=True)
                subprocess.run(cmd, check=True)
                print("CUDA PyTorch installation complete.", flush=True)
            else:
                print(
                    "Environment verification passed! All dependencies correctly installed.",
                    flush=True,
                )
        except subprocess.CalledProcessError:
            print("ERROR: Failed to verify torch CUDA status.", flush=True)
            sys.exit(1)
    else:
        print(
            "Environment verification passed! All dependencies correctly installed.",
            flush=True,
        )


if __name__ == "__main__":
    main()
