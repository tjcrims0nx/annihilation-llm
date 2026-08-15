import importlib.util
import os
import shutil
import subprocess
import sys

# Settings the TUI passes on the command line every run now that OBLITERATUS is
# integrated: `--kernel-type`, `--use-cosmic-layer-selection`, `--use-ega`.
OBLITERATUS_SETTINGS = ("kernel_type", "use_cosmic_layer_selection", "use_ega")


def in_virtual_env() -> bool:
    return sys.prefix != sys.base_prefix


def uv_sync_target_args() -> list[str]:
    """Arguments pointing `uv sync` at the interpreter running this script.

    uv resolves the project environment to `.venv` no matter which interpreter
    invoked it, but the TUI prefers `annihilation-env` when it exists (see
    `tui/src/subprocess.rs`). Left to its default, uv installs into `.venv`
    while this script — and the engine the TUI spawns next — goes on importing
    from `annihilation-env`, so a sync that reports "Installed 2 packages"
    changes nothing either of them can see.

    Outside a virtual environment there is no active target to name, so uv keeps
    its own default.
    """
    return ["--active"] if in_virtual_env() else []


def uv_pip_target_args() -> list[str]:
    """The same targeting for `uv pip install`, which has no `--active`.

    `--active` is a `uv sync` flag; passing it to `uv pip install` fails outright
    with "unexpected argument". `--python` is the equivalent there and is more
    explicit besides — it names the interpreter rather than relying on
    VIRTUAL_ENV being read the way we expect (same approach as
    `scripts/gguf_converter.py`).
    """
    return ["--python", sys.executable]


def uv_env() -> dict[str, str]:
    """Environment for uv calls, with the active virtual environment pinned."""
    env = os.environ.copy()
    env.pop("UV_EXCLUDE_NEWER", None)

    if in_virtual_env():
        # `--active` follows VIRTUAL_ENV, which the TUI does not necessarily set
        # (it runs the venv's python directly instead of activating it) and which
        # has been seen holding a bare relative path. `sys.prefix` is the
        # environment this interpreter actually imports from.
        env["VIRTUAL_ENV"] = sys.prefix

    return env


def check_obliteratus_settings() -> str | None:
    """Why the installed `annihilate` cannot serve OBLITERATUS, or None if it can.

    An `annihilate` predating the OBLITERATUS integration imports fine, so the
    `find_spec` check below passes and this script reports the environment as
    good — and then the run dies on "unrecognized arguments" once the TUI spawns
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
        # A failed import is a different problem from a stale engine and needs a
        # different fix, so report what actually went wrong rather than folding
        # it into "these fields are missing".
        output = (result.stderr or result.stdout).strip().splitlines()
        detail = output[-1] if output else "no output"
        return f"cannot import annihilate.config: {detail}"

    missing = result.stdout.split()
    if missing:
        return f"missing {', '.join(missing)}"

    return None


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
        problem = check_obliteratus_settings()
        if problem is not None:
            print(
                f"Installed annihilate cannot serve OBLITERATUS ({problem}). "
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

            cmd = [
                "uv",
                "sync",
                "--no-progress",
                "--link-mode=copy",
                *uv_sync_target_args(),
            ]
            print(f"Running: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, env=uv_env())
        else:
            cmd = [sys.executable, "-m", "pip", "install", ".", "--no-cache-dir"]
            if is_gpu:
                cmd.extend(
                    ["--extra-index-url", "https://download.pytorch.org/whl/cu126"]
                )
            print(f"Running: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)

        print("Dependencies installation complete.", flush=True)

        # Installing is not the same as installing somewhere this interpreter can
        # see, which is the failure this check exists to catch. Naming the
        # environment matters: the remedy for "landed in the wrong environment"
        # is nothing like the remedy for "older copy earlier on sys.path". Kept
        # ASCII like the rest of this script's output, which the TUI decodes as
        # UTF-8.
        problem = check_obliteratus_settings()
        if problem is not None:
            env_name = os.path.basename(sys.prefix)
            print(
                f"ERROR: annihilate in {sys.prefix} still cannot serve "
                f"OBLITERATUS after installing ({problem}).\n"
                f"  Running interpreter: {sys.executable}\n"
                f"  sys.prefix:          {sys.prefix}\n"
                f"  VIRTUAL_ENV:         {os.environ.get('VIRTUAL_ENV', '(unset)')}\n"
                f"The install may have targeted a different environment. "
                f"Make sure uv is installing into '{env_name}', not a "
                f"sibling like '.venv'.",
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
                        *uv_pip_target_args(),
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
                subprocess.run(cmd, check=True, env=uv_env())
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
