import importlib.util
import os
import shutil
import subprocess
import sys

# Settings the TUI passes on the command line every run now that OBLITERATUS is
# integrated: `--kernel-type`, `--use-cosmic-layer-selection`, `--use-ega`.
OBLITERATUS_SETTINGS = ("kernel_type", "use_cosmic_layer_selection", "use_ega")


def missing_obliteratus_settings() -> list[str]:
    """Names of the OBLITERATUS settings the installed `annihilate` does not have.

    An `annihilate` predating the OBLITERATUS integration imports fine, so the
    `find_spec` check below passes and this script reports the environment as
    good, and then the run dies on "unrecognized arguments" once the TUI spawns
    the engine, after a model has already been downloaded and loaded. Checking
    the fields turns that into a reinstall here instead.

    Probed out of process for the same reason as the CUDA check further down:
    nothing this script verifies should be left imported in this interpreter.
    """
    probe = (
        "from annihilate.config import Settings\n"
        f"names = {OBLITERATUS_SETTINGS!r}\n"
        "print(' '.join(n for n in names if n not in Settings.model_fields))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Importable per `find_spec` but not actually usable. Reported as all
        # settings missing: a broken engine and a stale one need the same remedy.
        return list(OBLITERATUS_SETTINGS)

    return result.stdout.split()


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

    # Only worth probing when the package is there to probe; otherwise the
    # install below is already going to happen.
    if not needs_install:
        missing = missing_obliteratus_settings()
        if missing:
            print(
                f"Installed annihilate predates OBLITERATUS (missing {', '.join(missing)}). "
                "Reinstalling...",
                flush=True,
            )
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
                    ["--extra-index-url", "https://download.pytorch.org/whl/cu126"]
                )
            print(f"Running: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)

        print("Dependencies installation complete.", flush=True)

        # A reinstall that still does not provide the settings means something is
        # shadowing this checkout: an older wheel carrying the same version
        # number, or a stray `annihilate/` earlier on sys.path. Stop here, since
        # the whole point of the check is to not discover it mid-run.
        still_missing = missing_obliteratus_settings()
        if still_missing:
            print(
                "ERROR: annihilate still does not provide "
                f"{', '.join(still_missing)} after reinstalling. Another "
                "annihilate is shadowing this checkout. Uninstall it with "
                "`pip uninstall annihilate-llm` and try again.",
                flush=True,
            )
            sys.exit(1)

    if is_gpu:
        # Use a subprocess to check CUDA availability without loading the DLL into this process
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import torch; print(torch.cuda.is_available())",
                ],
                capture_output=True,
                text=True,
                check=True,
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
                        "https://download.pytorch.org/whl/cu126",
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
                        "https://download.pytorch.org/whl/cu126",
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
