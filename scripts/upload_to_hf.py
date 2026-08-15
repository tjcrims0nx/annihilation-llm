"""
Upload annihilated model (Safetensors, Configs, Tokenizer, and GGUF) to HuggingFace Hub.
Usage:
  python scripts/upload_to_hf.py [--trial TRIAL_ID] [--repo REPO_ID]
"""

import argparse
import io
import shutil
import sys
from pathlib import Path

# See the note in scripts/gguf_converter.py: hasattr does not narrow sys.stdout
# for the type checker, isinstance does.
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi

# Import annihilate components
from annihilate.export import read_trial_attributes, settings_from_checkpoint
from annihilate.model import AbliterationParameters, Model
from annihilate.system import empty_cache
from annihilate.utils import checkpoint_name_for_model, load_prompts

DEFAULT_REPO_ID = "Grimxlock/openbmb-MiniCPM5-1B-F16-Annihilated"
MODEL_NAME = "openbmb/MiniCPM5-1B"
ROOT_DIR = Path(__file__).parent.parent.resolve()


def main():
    parser = argparse.ArgumentParser(description="Upload annihilated model to HF")
    parser.add_argument(
        "--model-name", type=str, default=MODEL_NAME, help="Base model name or ID"
    )
    parser.add_argument(
        "--trial", type=int, default=179, help="Trial ID to export (e.g. 179, 196)"
    )
    parser.add_argument(
        "--repo", type=str, default=DEFAULT_REPO_ID, help="HuggingFace Repo ID"
    )
    args = parser.parse_args()

    repo_id = args.repo
    target_trial_id = args.trial
    model_name = args.model_name

    print(f"=== ANNIHILATION-LLM: Upload to HuggingFace ({repo_id}) ===")

    api = HfApi()
    user_info = api.whoami()
    print(f"Authenticated as: {user_info['name']}")

    sanitized = checkpoint_name_for_model(model_name)
    checkpoint_path = ROOT_DIR / "checkpoints" / f"{sanitized}.jsonl"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file {checkpoint_path} not found.")
        sys.exit(1)

    print(f"Reading checkpoint: {checkpoint_path}")
    settings = settings_from_checkpoint(str(checkpoint_path), model_name)

    if settings.batch_size == 0:
        settings.batch_size = 16

    if settings.seed is None:
        settings.seed = 42

    trials = read_trial_attributes(str(checkpoint_path))

    if target_trial_id in trials:
        best_trial_id = target_trial_id
        attrs = trials[target_trial_id]
        best_refusals = attrs.get("refusals", 0)
        best_kl = attrs.get("kl_divergence", 0.0)
        best_trial_params = attrs.get("parameters")
        best_direction_index = attrs.get("direction_index")
    else:
        print(f"Trial ID {target_trial_id} not found, finding lowest refusal trial...")
        best_trial_params = None
        best_direction_index = None
        best_kl = float("inf")
        best_refusals = float("inf")
        best_trial_id = -1

        for trial_id, attrs in trials.items():
            refusals = attrs.get("refusals", float("inf"))
            kl = attrs.get("kl_divergence", float("inf"))

            if refusals < best_refusals or (refusals == best_refusals and kl < best_kl):
                best_refusals = refusals
                best_kl = kl
                best_trial_params = attrs.get("parameters")
                best_direction_index = attrs.get("direction_index")
                best_trial_id = trial_id

    print(
        f"Selected Trial {best_trial_id} (Refusals: {best_refusals}, KL Div: {best_kl:.4f}). Loading base model..."
    )

    model = Model(settings)

    print("Calculating refusal directions...")
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

    if best_trial_params is None:
        print("Error: No parameters found for trial.")
        sys.exit(1)

    parameters = {k: AbliterationParameters(**v) for k, v in best_trial_params.items()}
    model.abliterate(refusal_directions, best_direction_index, parameters)

    print("Merging LoRA adapters into base model...")
    merged = model.get_merged_model()

    export_dir = ROOT_DIR / "exports_upload_tmp"
    if export_dir.exists():
        shutil.rmtree(export_dir, ignore_errors=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    print("Saving merged model & tokenizer...")
    merged.save_pretrained(export_dir, max_shard_size="10GB")
    model.tokenizer.save_pretrained(export_dir)
    processor = getattr(model, "processor", None)
    if processor is not None:
        processor.save_pretrained(export_dir)

    del merged
    del model
    empty_cache()

    # 2. Find and copy GGUF file
    exports_dir = ROOT_DIR / "exports"
    gguf_candidates = list(exports_dir.glob("*.gguf")) if exports_dir.exists() else []

    if gguf_candidates:
        for gguf_file in gguf_candidates:
            target_gguf = export_dir / gguf_file.name
            print(
                f"Copying GGUF file {gguf_file.name} ({gguf_file.stat().st_size / (1024**3):.2f} GiB) -> {target_gguf}..."
            )
            shutil.copy2(gguf_file, target_gguf)
    else:
        print(f"Warning: No GGUF files found in {exports_dir}.")

    # 3. Create Model Card README.md
    readme_content = f"""---
license: apache-2.0
base_model: {MODEL_NAME}
library_name: transformers
tags:
- abliterated
- uncensored
- annihilation-llm
- gguf
---

# {repo_id}

This repository contains the **annihilated / abliterated** version of [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) generated using [Annihilation LLM](https://github.com/tjcrims0nx/annihilation-llm).

## Files Included:
- **`model.safetensors`**: Merged PyTorch / Safetensors weights (BFloat16 / Float16).
- **`*.gguf`**: F16 GGUF model for llama.cpp / LM Studio / Ollama.
- **Tokenizer & Config files**: Full HuggingFace compatibility files.

## Optimal Trial Details:
- **Trial ID**: {best_trial_id}
- **Refusals**: {best_refusals} / 100 prompts
- **KL Divergence**: {best_kl:.4f}
"""
    readme_path = export_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")

    # 4. Upload to HuggingFace Hub
    print(f"Uploading all files to HuggingFace Hub repo: {repo_id}...")
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="model",
    )

    print(
        f"[SUCCESS] Upload complete! Check your model at: https://huggingface.co/{repo_id}"
    )

    # Cleanup temp directory
    shutil.rmtree(export_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
