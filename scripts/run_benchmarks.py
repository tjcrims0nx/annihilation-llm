"""Benchmark runner for the Annihilation TUI.

Loads the base model, finds the best trial from checkpoints,
applies abliteration, and then runs lm-eval benchmarks.

Usage: python -u scripts/run_benchmarks.py <model_name> [benchmark1,benchmark2,...]
"""

import json
import sys
from pathlib import Path

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import lm_eval
import torch
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM

from src.annihilate.config import Settings
from src.annihilate.export import read_trial_attributes, settings_from_checkpoint
from src.annihilate.model import AbliterationParameters, Model
from src.annihilate.utils import checkpoint_name_for_model, load_prompts


def print_event(event_type: str, content: str):
    print(json.dumps({"type": event_type, "content": content}), flush=True)


def load_optimal_trial_and_merge(
    model_name: str, target_trial_id: int | None = None
) -> Model:
    """Load the base model and reconstruct the selected/best abliterated version from checkpoints."""
    original_argv = sys.argv.copy()
    sys.argv = sys.argv[:1]
    base_settings = Settings(model=model_name)
    sys.argv = original_argv

    sanitized = checkpoint_name_for_model(model_name)
    checkpoint_path = (
        Path(__file__).parent.parent
        / base_settings.study_checkpoint_dir
        / f"{sanitized}.jsonl"
    )

    if not checkpoint_path.exists():
        print_event(
            "error",
            f"No checkpoint found at {checkpoint_path}. Run annihilation first.",
        )
        sys.exit(1)

    settings = settings_from_checkpoint(str(checkpoint_path), model_name)

    if settings.batch_size == 0:
        settings.batch_size = 16

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
        print_event("error", "No successful trials found in checkpoint.")
        sys.exit(1)

    print_event(
        "status",
        f"Loading model {model_name} (Trial {best_trial_id}, KL Div: {best_kl:.4f})...",
    )
    model = Model(settings)

    print_event("status", "Calculating refusal directions...")
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

    print_event("status", "Applying abliteration parameters...")
    parameters = {k: AbliterationParameters(**v) for k, v in best_trial_params.items()}
    model.abliterate(refusal_directions, best_direction_index, parameters)

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmarks for Annihilation TUI")
    parser.add_argument("model_name", nargs="?", default="openbmb/MiniCPM5-1B")
    parser.add_argument("benchmarks", nargs="?", default="hellaswag,arc_easy")
    parser.add_argument("--trial", "--trial-id", type=int, default=None)
    args, _ = parser.parse_known_args()

    model_name = args.model_name
    benchmarks = (
        args.benchmarks.split(",") if args.benchmarks else ["hellaswag", "arc_easy"]
    )
    target_trial_id = args.trial

    try:
        model = load_optimal_trial_and_merge(
            model_name, target_trial_id=target_trial_id
        )

        # Initialize lm-eval wrapper
        print_event("status", "Initializing evaluation harness...")
        hflm = HFLM(
            pretrained=model.model,
            tokenizer=model.tokenizer,  # ty:ignore[invalid-argument-type]
            batch_size="auto",
        )

        for benchmark in benchmarks:
            print_event("status", f"Running benchmark {benchmark}...")

            results = lm_eval.simple_evaluate(
                model=hflm,
                tasks=[benchmark],
            )

            # simple_evaluate returns None on non-primary ranks, and omits the
            # task key entirely if it produced no metrics.
            benchmark_results = (results or {}).get("results", {}).get(benchmark)
            if not benchmark_results:
                print_event("error", f"Benchmark {benchmark} returned no results.")
                continue

            for metric, value in benchmark_results.items():
                if metric != "alias":
                    if isinstance(value, float):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = f"{value}"

                    # Send result event to TUI
                    print(
                        json.dumps(
                            {
                                "type": "result",
                                "benchmark": benchmark,
                                "metric": metric,
                                "value": value_str,
                            }
                        ),
                        flush=True,
                    )

        print_event("done", "Benchmarks completed successfully.")

    except Exception:
        import traceback

        print_event("error", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
