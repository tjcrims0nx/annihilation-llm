"""Export a specific trial from a study checkpoint as a merged model.

The optimizer stores each trial's abliteration parameters in the Optuna journal
(a JSONL file under ``checkpoints/``). This module reconstructs one of those
trials without rerunning the search: it recomputes the refusal directions from
the configured prompt datasets, applies the stored parameters, and writes the
merged model to disk.
"""

import argparse
import json
import sys
from pathlib import Path


# Provide rich JSON output for TUI parsing
def print_event(level: str, msg: str):
    print(json.dumps({"level": level, "message": msg}), flush=True)


# Optuna's JournalFileBackend writes one operation per line; op_code 8 is the
# "set trial user attribute" record that carries the abliteration parameters,
# and op_code 2 is the study-level attribute that carries the settings blob.
_OP_SET_TRIAL_USER_ATTR = 8
_OP_SET_STUDY_USER_ATTR = 2


def read_study_settings(checkpoint_path: str) -> str | None:
    """Returns the serialized settings a study was run with, if recorded."""

    settings_json = None

    with open(checkpoint_path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("op_code") != _OP_SET_STUDY_USER_ATTR:
                continue

            stored = data.get("user_attr", {}).get("settings")
            if stored is not None:
                # A resumed study writes the blob again; keep the newest.
                settings_json = stored

    return settings_json


def settings_from_checkpoint(checkpoint_path: str, model_name: str):
    """Rebuilds the Settings a study was run with.

    Reconstructing a trial with anything other than the original settings
    silently produces a different model, because settings such as
    ``orthogonalize_direction``, ``row_normalization`` and the prompt datasets
    all feed into the refusal directions and the merge.

    The model recorded in the checkpoint takes precedence over ``model_name``,
    which callers derive from the filename: that mapping replaces ``/`` with
    ``--`` and so cannot be inverted reliably. A disagreement is reported
    rather than applied silently.
    """
    from .config import Settings

    # Settings parses sys.argv via CliSettingsSource, which would choke on the
    # calling script's own arguments, so hide them while constructing it.
    original_argv = sys.argv.copy()
    sys.argv = sys.argv[:1]
    try:
        settings_json = read_study_settings(checkpoint_path)

        if settings_json is None:
            print_event(
                "warning",
                "Checkpoint records no settings; falling back to the current configuration.",
            )
            return Settings(model=model_name)

        settings = Settings.model_validate_json(settings_json)

        if settings.model != model_name:
            print_event(
                "warning",
                f"Checkpoint was recorded for {settings.model}, not {model_name}. "
                f"Using {settings.model}.",
            )

        # Fields marked exclude=True are absent from the stored blob, so
        # model_validate_json resets them to their defaults. Restore them from
        # the current configuration rather than silently losing them.
        defaults = Settings(model=settings.model)
    finally:
        sys.argv = original_argv

    for name, field in Settings.model_fields.items():
        if field.exclude:
            setattr(settings, name, getattr(defaults, name))

    return settings


def model_name_from_checkpoint(checkpoint_path: str) -> str:
    """Recovers the model ID from a checkpoint filename.

    Checkpoints are named after the model with path separators replaced by
    ``--`` (see ``sanitize_model_name`` in the TUI), so the mapping back is
    only unambiguous for Hugging Face-style ``owner/name`` IDs.
    """
    return Path(checkpoint_path).stem.replace("--", "/")


def read_trial_attributes(checkpoint_path: str) -> dict[int, dict]:
    """Collects the user attributes recorded for each trial in a checkpoint."""

    trials: dict[int, dict] = {}

    with open(checkpoint_path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("op_code") != _OP_SET_TRIAL_USER_ATTR:
                continue

            trial_id = data.get("trial_id")
            if trial_id is None:
                continue

            # Attributes for a trial are spread over several records.
            trials.setdefault(trial_id, {}).update(data.get("user_attr", {}))

    return trials


def export_model(checkpoint_path: str, trial_id: int, output_dir: str):
    print_event("info", f"Exporting trial {trial_id} from {checkpoint_path}...")

    if not Path(checkpoint_path).is_file():
        print_event("error", f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    import torch
    import torch.nn.functional as F

    from .model import AbliterationParameters, Model
    from .system import empty_cache
    from .utils import load_prompts

    trials = read_trial_attributes(checkpoint_path)
    attributes = trials.get(trial_id)

    if attributes is None or not attributes.get("parameters"):
        print_event(
            "error",
            f"Could not find parameters for trial {trial_id} in {checkpoint_path}",
        )
        sys.exit(1)

    trial_parameters = attributes["parameters"]
    direction_index = attributes.get("direction_index")

    model_name = model_name_from_checkpoint(checkpoint_path)

    # Reuse the settings the study was actually run with, so the reconstructed
    # model matches the trial (including the pinned model_commit revision).
    settings = settings_from_checkpoint(checkpoint_path, model_name)

    # A batch size of 0 means "determine automatically", which only happens in
    # the main optimization loop. Residual analysis below needs a real value.
    if settings.batch_size == 0:
        settings.batch_size = 16

    # abliterate() seeds torch, so the seed must be concrete.
    if settings.seed is None:
        settings.seed = 42

    print_event("info", f"Loading base model {model_name}...")
    model = Model(settings)

    try:
        print_event("info", "Calculating refusal directions...")
        good_prompts = load_prompts(settings, settings.good_prompts)
        bad_prompts = load_prompts(settings, settings.bad_prompts)

        good_means = model.get_residuals_mean(good_prompts)
        bad_means = model.get_residuals_mean(bad_prompts)

        refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

        if settings.orthogonalize_direction:
            good_directions = F.normalize(good_means, p=2, dim=1)
            projection = torch.sum(refusal_directions * good_directions, dim=1)
            refusal_directions = (
                refusal_directions - projection.unsqueeze(1) * good_directions
            )
            refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

        print_event("info", "Applying abliteration parameters...")
        parameters = {
            component: AbliterationParameters(**values)
            for component, values in trial_parameters.items()
        }
        model.abliterate(refusal_directions, direction_index, parameters)

        print_event("info", "Merging model...")
        merged = model.get_merged_model()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print_event("info", f"Saving merged model to {output_path}...")
        merged.save_pretrained(output_path, max_shard_size=settings.max_shard_size)
        model.tokenizer.save_pretrained(output_path)
        if model.processor is not None:
            model.processor.save_pretrained(output_path)

        del merged
    finally:
        del model
        empty_cache()

    print_event("info", f"Export complete: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export a trial from a study checkpoint as a merged model."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the .jsonl checkpoint."
    )
    parser.add_argument("--trial-id", required=True, type=int, help="Trial to export.")
    parser.add_argument(
        "--output", required=True, help="Directory to write the model to."
    )
    args = parser.parse_args()

    try:
        export_model(args.checkpoint, args.trial_id, args.output)
    except Exception as error:
        print_event("error", str(error))
        sys.exit(1)


if __name__ == "__main__":
    main()
