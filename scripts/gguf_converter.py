import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

# isinstance rather than hasattr: a type checker cannot narrow sys.stdout (typed
# TextIO) from a hasattr guard, so `reconfigure` resolves to object and the call
# fails `ty check`. The guard is still needed — stdout is not always a real
# TextIOWrapper, e.g. under pytest capture.
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")


def print_event(level: str, msg: str):
    print(json.dumps({"level": level, "message": msg}), flush=True)


def _find_file(root: Path, name: str) -> Path | None:
    """Recursively find a file by name under root."""
    for path in root.rglob(name):
        return path
    return None


def _download_verified(url: str, dest: Path, expected_sha256: str | None = None) -> str:
    """Download `url` to `dest`, returning the archive's SHA-256 hex digest.

    These archives are executed (the prebuilt `llama-quantize` binary) and
    imported (`convert_hf_to_gguf.py`), so they get treated as code:

    * The URL scheme is pinned to HTTPS. The release-asset URL is read out of a
      JSON response, so without this check a redirected or tampered API reply
      could point the download at `http://` or `file://`.
    * The digest is always computed and reported. When `expected_sha256` is
      known (GitHub publishes one per release asset, and the caller may pin one
      via the environment) a mismatch is fatal instead of merely logged.
    """
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme != "https":
        print_event("error", f"Refusing to download over {scheme or 'unknown'}: {url}")
        sys.exit(1)

    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            digest.update(chunk)
            f.write(chunk)

    actual = digest.hexdigest()

    if expected_sha256:
        if actual.lower() != expected_sha256.strip().lower():
            dest.unlink(missing_ok=True)
            print_event(
                "error",
                f"Checksum mismatch for {url}: expected {expected_sha256}, got {actual}. "
                "The download was discarded.",
            )
            sys.exit(1)
        print_event("info", f"Verified SHA-256 {actual} for {dest.name}.")
    else:
        print_event(
            "info",
            f"SHA-256 of {dest.name} is {actual} (no checksum published to verify against).",
        )

    return actual


def _bin_flavor() -> str:
    """The release-asset flavor fragment that identifies this machine.

    llama.cpp release assets are named `llama-<tag>-bin-<flavor>.<ext>`,
    e.g. `llama-b10472-bin-win-cpu-x64.zip` or
    `llama-b10472-bin-macos-arm64.tar.gz`.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        arch = "x64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        print_event(
            "error",
            f"No llama.cpp prebuilt binary is available for this architecture "
            f"({machine}). Place a `llama-quantize` build under "
            f"llama_cpp_bin/ manually to continue.",
        )
        sys.exit(1)

    if system == "windows":
        return f"win-cpu-{arch}"
    if system == "darwin":
        return f"macos-{arch}"
    if system == "linux":
        # The "ubuntu" build is the generic glibc Linux binary, the sensible
        # default for Linux distributions. musl-based systems (Alpine) may
        # need to build llama.cpp from source instead.
        return f"ubuntu-{arch}"

    print_event(
        "error",
        f"No llama.cpp prebuilt binary is available for this operating system "
        f"({system}). Place a `llama-quantize` build under "
        f"llama_cpp_bin/ manually to continue.",
    )
    sys.exit(1)


def _safe_extract_archive(archive: Path, dest: Path):
    """Extract a `.zip` or `.tar.gz` into `dest`, rejecting members that
    escape `dest`.

    `ZipFile.extractall` already sanitizes member paths, but it does so
    silently. Failing loudly means a malformed archive surfaces as an error
    rather than as files quietly landing somewhere unexpected. For tarballs
    the executable bit is restored on extracted binaries as well: GitHub
    archives preserve permissions, but a tar member arriving without it
    would make quantize fail with a permission error at conversion time.
    """
    dest_resolved = dest.resolve()

    def check_member(name: str):
        target = (dest_resolved / name).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            print_event(
                "error",
                f"Archive {archive.name} contains an unsafe path: {name}",
            )
            sys.exit(1)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zip_ref:
            for member in zip_ref.namelist():
                check_member(member)
            zip_ref.extractall(dest)
        return

    with tarfile.open(archive, "r:*") as tar_ref:
        for member in tar_ref.getmembers():
            check_member(member.name)
        try:
            # `filter="data"` is defence in depth beyond the path check above:
            # it additionally rejects device files and hard links. It exists
            # only in newer Python patch releases, so fall back gracefully.
            tar_ref.extractall(dest, filter="data")
        except TypeError:
            tar_ref.extractall(dest)
        for member in tar_ref.getmembers():
            if member.isfile():
                extracted = dest / member.name
                mode = extracted.stat().st_mode
                extracted.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def ensure_llama_cpp(bin_dir: Path):
    if not bin_dir.exists():
        bin_dir.mkdir(parents=True, exist_ok=True)

    quantize_name = "llama-quantize.exe" if os.name == "nt" else "llama-quantize"
    quantize_exe = _find_file(bin_dir, quantize_name)

    # Tag of the release the binaries came from, so the source archive below can
    # be pinned to the same revision instead of tracking a moving branch head.
    release_tag = None

    if quantize_exe is None:
        print_event("info", "Downloading llama.cpp prebuilt binaries...")
        # The same fetch-and-verify path on every platform; only the asset
        # flavor and the archive format differ.
        flavor = _bin_flavor()

        req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        url = None
        expected_digest = None
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
            release_tag = data.get("tag_name")
            for asset in data["assets"]:
                if (
                    "llama-" in asset["name"]
                    and f"-{flavor}" in asset["name"]
                    and "bin" in asset["name"]
                ):
                    url = asset["browser_download_url"]
                    # GitHub publishes asset digests as "sha256:<hex>".
                    digest = asset.get("digest") or ""
                    if digest.lower().startswith("sha256:"):
                        expected_digest = digest.split(":", 1)[1]
                    break
        except Exception as e:
            print_event("error", f"Failed to fetch latest release info: {e}")
            sys.exit(1)

        if not url:
            print_event(
                "error",
                f"Could not find a valid llama.cpp release asset for {flavor}.",
            )
            sys.exit(1)

        # An explicit pin always wins over whatever the API reports.
        expected_digest = os.environ.get("LLAMA_CPP_BIN_SHA256") or expected_digest

        print_event("info", f"Downloading from {url}...")
        archive_path = bin_dir / ("llama.zip" if os.name == "nt" else "llama.tar.gz")
        _download_verified(url, archive_path, expected_digest)
        _safe_extract_archive(archive_path, bin_dir)
        archive_path.unlink()

        quantize_exe = _find_file(bin_dir, quantize_name)
        if quantize_exe is None:
            print_event("error", f"Could not find {quantize_name} after extraction.")
            sys.exit(1)

    # Download the full llama.cpp source for convert_hf_to_gguf.py + conversion module.
    # Prefer the tag matching the binaries just downloaded; only fall back to the
    # branch head when no tag is known (i.e. the binaries were already cached).
    ref = f"refs/tags/{release_tag}" if release_tag else "refs/heads/master"
    src_dir_name = f"llama.cpp-{release_tag}" if release_tag else "llama.cpp-master"
    src_dir = bin_dir / src_dir_name
    convert_py = src_dir / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        # A previous run may have cached the source under a different ref.
        cached = _find_file(bin_dir, "convert_hf_to_gguf.py")
        if cached is not None:
            convert_py = cached
        else:
            print_event(
                "info",
                f"Downloading llama.cpp source ({ref}) for conversion scripts...",
            )
            url = f"https://github.com/ggml-org/llama.cpp/archive/{ref}.zip"
            src_zip = bin_dir / "llama_src.zip"
            _download_verified(url, src_zip, os.environ.get("LLAMA_CPP_SRC_SHA256"))
            _safe_extract_archive(src_zip, bin_dir)
            src_zip.unlink()

            convert_py = _find_file(bin_dir, "convert_hf_to_gguf.py")
            if convert_py is None:
                print_event(
                    "error", "Could not find convert_hf_to_gguf.py after extraction."
                )
                sys.exit(1)

    # Ensure gguf package is installed. It is an optional extra rather than a
    # required dependency (nothing under src/annihilate imports it), so a plain
    # `uv sync` or `pip install annihilate-llm` leaves it out and this fallback
    # is what makes conversion work anyway. Install the extra to skip it:
    # `pip install annihilate-llm[gguf]`, or `uv sync --extra gguf`.
    #
    # Probe by spec rather than importing, since an import statement here would
    # not resolve for type checking.
    if importlib.util.find_spec("gguf") is None:
        print_event("info", "Installing gguf python package...")
        _install_package("gguf")

    return quantize_exe, convert_py


def _install_package(package: str) -> None:
    """Installs a package into the running interpreter's environment.

    Environments created by `uv venv` (which is what `uv sync` builds, and what
    the TUI picks first) do not ship pip, so `-m pip install` fails there with
    "No module named pip". Try pip when it is actually importable, and otherwise
    fall back to `uv pip install --python <this interpreter>`, which targets this
    environment rather than whatever uv would infer on its own.
    """

    attempts: list[list[str]] = []

    if importlib.util.find_spec("pip") is not None:
        attempts.append([sys.executable, "-m", "pip", "install", package])

    uv = shutil.which("uv")
    if uv is not None:
        attempts.append([uv, "pip", "install", "--python", sys.executable, package])

    if not attempts:
        print_event(
            "error",
            f"Cannot install {package}: this environment has no pip, and uv was "
            f"not found on PATH. Install it manually with "
            f"`uv pip install {package}`.",
        )
        sys.exit(1)

    errors: list[str] = []

    for command in attempts:
        try:
            subprocess.check_call(command)
            return
        except (subprocess.CalledProcessError, OSError) as e:
            errors.append(f"{' '.join(command)} -> {e}")

    # Report every attempt; a bare "returned non-zero exit status 1" gives no
    # indication of which installer was tried or why it failed.
    details = "; ".join(errors)
    print_event("error", f"Failed to install {package}. Tried: {details}")
    sys.exit(1)


def load_optimal_trial_and_merge(
    model_name: str, target_trial_id: int | None = None
) -> str:
    # Needs to be able to import src
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.annihilate.utils import checkpoint_name_for_model

    # Must match how main.py names the file, or no checkpoint is ever found.
    sanitized = checkpoint_name_for_model(model_name)
    checkpoint_path = (
        Path(__file__).parent.parent / "checkpoints" / f"{sanitized}.jsonl"
    )

    if not checkpoint_path.exists():
        print_event(
            "error",
            f"Error: {model_name} is not a directory, and no checkpoint was found.",
        )
        sys.exit(1)

    print_event(
        "info",
        f"Found completed runs for {model_name}. Reconstructing optimal trial...",
    )

    import torch
    import torch.nn.functional as F

    from src.annihilate.export import read_trial_attributes, settings_from_checkpoint
    from src.annihilate.model import AbliterationParameters, Model
    from src.annihilate.system import empty_cache
    from src.annihilate.utils import load_prompts

    # Reconstructing a trial under different settings silently produces a
    # different model, so reuse the settings the study was run with (including
    # the pinned model_commit revision).
    settings = settings_from_checkpoint(str(checkpoint_path), model_name)

    # Ensure batch_size is valid
    if settings.batch_size == 0:
        settings.batch_size = 16

    # Ensure seed is valid (needed for torch.manual_seed in abliterate)
    if settings.seed is None:
        settings.seed = 42

    trials = read_trial_attributes(str(checkpoint_path))

    best_trial_params = None
    best_direction_index = None
    best_kl = float("inf")
    best_refusals = float("inf")
    best_trial_id = -1

    if target_trial_id is not None and target_trial_id in trials:
        attrs = trials[target_trial_id]
        best_refusals = attrs.get("refusals", float("inf"))
        best_kl = attrs.get("kl_divergence", float("inf"))
        best_trial_params = attrs.get("parameters")
        best_direction_index = attrs.get("direction_index")
        best_trial_id = target_trial_id
    else:
        for trial_id, attrs in trials.items():
            refusals = attrs.get("refusals", float("inf"))
            kl = attrs.get("kl_divergence", float("inf"))

            if refusals < best_refusals or (refusals == best_refusals and kl < best_kl):
                best_refusals = refusals
                best_kl = kl
                best_trial_params = attrs.get("parameters")
                best_direction_index = attrs.get("direction_index")
                best_trial_id = trial_id

    if not best_trial_params:
        print_event(
            "error", f"Could not find any successful trials in {checkpoint_path}"
        )
        sys.exit(1)

    print_event(
        "info",
        f"Selected Trial {best_trial_id} ({best_refusals} refusals, KL Div: {best_kl:.4f}). Loading base model...",
    )

    model = Model(settings)

    print_event("info", "Calculating refusal directions...")
    if settings.batch_size == 0:
        settings.batch_size = 16

    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)

    good_means = model.get_residuals_mean(good_prompts)
    bad_means = model.get_residuals_mean(bad_prompts)

    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if settings.orthogonalize_direction:
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
        refusal_directions = (
            refusal_directions - projection_vector.unsqueeze(1) * good_directions
        )
        refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

    print_event("info", "Applying abliteration parameters...")
    parameters = {k: AbliterationParameters(**v) for k, v in best_trial_params.items()}
    model.abliterate(refusal_directions, best_direction_index, parameters)

    print_event("info", "Merging model...")
    merged = model.get_merged_model()

    out_dir = Path(__file__).parent / f"gguf_tmp_export_{sanitized}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print_event("info", "Saving merged model to temporary folder for conversion...")
    merged.save_pretrained(out_dir, max_shard_size="10GB")
    model.tokenizer.save_pretrained(out_dir)
    processor = getattr(model, "processor", None)
    if processor is not None:
        processor.save_pretrained(out_dir)

    del merged
    del model
    empty_cache()

    return str(out_dir)


def _stream_output(proc: subprocess.Popen) -> int:
    """Relay a child's merged stdout/stderr as debug events, then return its code.

    `Popen.stdout` is only non-None when `stdout=PIPE` was requested; guarding
    keeps a mis-configured call from raising `TypeError: 'NoneType' is not
    iterable` and masking the child's real failure.
    """
    if proc.stdout is not None:
        with proc.stdout:
            for line in proc.stdout:
                print_event("debug", line.strip())
    return proc.wait()


def convert_to_gguf(
    model_path: str,
    quant_type: str,
    output_path: str,
    target_trial_id: int | None = None,
):
    bin_dir = Path(__file__).parent / "llama_cpp_bin"
    quantize_exe, convert_py = ensure_llama_cpp(bin_dir)

    temp_merged_dir = None
    if not os.path.exists(model_path):
        temp_merged_dir = load_optimal_trial_and_merge(
            model_path, target_trial_id=target_trial_id
        )
        model_path = temp_merged_dir

    try:
        # 1. Convert HF model to F16 GGUF.
        if quant_type.upper() == "F16":
            f16_gguf = output_path
        else:
            stem = Path(output_path).stem
            f16_gguf = str(Path(output_path).parent / f"{stem}-temp-F16.gguf")

        env = os.environ.copy()
        env["NO_LOCAL_GGUF"] = "1"

        cmd = [
            sys.executable,
            str(convert_py),
            model_path,
            "--outfile",
            f16_gguf,
            "--outtype",
            "f16",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(convert_py.parent),
            env=env,
        )
        returncode = _stream_output(proc)

        if returncode != 0 or not os.path.exists(f16_gguf):
            print_event("error", "Conversion to F16 GGUF failed.")
            sys.exit(1)

        if quant_type.upper() == "F16":
            if f16_gguf != output_path:
                shutil.move(f16_gguf, output_path)
            print_event("info", f"GGUF conversion complete: {output_path}")
            return

        # 2. Quantize from F16 to the target type
        print_event("info", f"Quantizing to {quant_type}...")
        cmd = [str(quantize_exe), f16_gguf, output_path, quant_type]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        returncode = _stream_output(proc)

        if returncode != 0:
            print_event("error", "Quantization failed.")
            sys.exit(1)

        # Cleanup intermediate F16 file
        os.remove(f16_gguf)
        print_event("info", f"GGUF quantization complete: {output_path}")

    finally:
        if temp_merged_dir and os.path.exists(temp_merged_dir):
            shutil.rmtree(temp_merged_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--quant-type", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trial", "--trial-id", type=int, default=None)
    args = parser.parse_args()

    try:
        convert_to_gguf(
            args.model_path, args.quant_type, args.output, target_trial_id=args.trial
        )
    except Exception as e:
        print_event("error", str(e))
        sys.exit(1)
