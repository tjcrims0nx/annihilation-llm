"""Chat server for the Annihilation TUI.

Loads the base model, finds the best trial from checkpoints,
applies abliteration, and then serves an interactive chat loop
over stdin/stdout using JSON messages.

Usage: python -u scripts/chat_server.py <model_name>
"""

import json
import sys
from pathlib import Path
from threading import Thread

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import TextIteratorStreamer

from src.annihilate.config import Settings
from src.annihilate.export import read_trial_attributes, settings_from_checkpoint
from src.annihilate.model import AbliterationParameters, Model
from src.annihilate.utils import checkpoint_name_for_model, load_prompts


def load_optimal_trial_and_merge(model_name: str) -> Model:
    """Load the base model and reconstruct the best abliterated version from checkpoints."""
    original_argv = sys.argv.copy()
    sys.argv = sys.argv[:1]
    base_settings = Settings(model=model_name)
    sys.argv = original_argv

    # Must match how main.py names the file, or no checkpoint is ever found.
    sanitized = checkpoint_name_for_model(model_name)
    checkpoint_path = (
        Path(__file__).parent.parent
        / base_settings.study_checkpoint_dir
        / f"{sanitized}.jsonl"
    )

    if not checkpoint_path.exists():
        print(
            json.dumps(
                {
                    "type": "error",
                    "content": f"No checkpoint found at {checkpoint_path}. Run annihilation first.",
                }
            ),
            flush=True,
        )
        sys.exit(1)

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

    for trial_id, attrs in trials.items():
        refusals = attrs.get("refusals", float("inf"))
        kl = attrs.get("kl_divergence", float("inf"))

        if refusals < best_refusals or (refusals == best_refusals and kl < best_kl):
            best_refusals = refusals
            best_kl = kl
            best_trial_params = attrs.get("parameters")
            best_direction_index = attrs.get("direction_index")

    if not best_trial_params:
        print(
            json.dumps(
                {
                    "type": "error",
                    "content": "No successful trials found in checkpoint.",
                }
            ),
            flush=True,
        )
        sys.exit(1)

    print(json.dumps({"type": "status", "content": "Loading model..."}), flush=True)
    model = Model(settings)

    print(
        json.dumps({"type": "status", "content": "Calculating refusal directions..."}),
        flush=True,
    )
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

    print(
        json.dumps(
            {"type": "status", "content": "Applying abliteration parameters..."}
        ),
        flush=True,
    )
    parameters = {k: AbliterationParameters(**v) for k, v in best_trial_params.items()}
    model.abliterate(refusal_directions, best_direction_index, parameters)

    return model


def main():
    if len(sys.argv) < 2:
        print(
            json.dumps({"type": "error", "content": "Missing model name argument"}),
            flush=True,
        )
        sys.exit(1)

    model_name = sys.argv[1]

    try:
        model = load_optimal_trial_and_merge(model_name)
        print(json.dumps({"type": "ready"}), flush=True)

        # Read chat history from stdin line by line
        for line in sys.stdin:
            if not line.strip():
                continue

            try:
                chat = json.loads(line)

                # Use the model's tokenizer to apply chat template
                prompt_text = model.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = model.tokenizer(prompt_text, return_tensors="pt").to(
                    model.model.device
                )

                streamer = TextIteratorStreamer(
                    model.tokenizer, skip_prompt=True, skip_special_tokens=True
                )
                generation_kwargs = dict(
                    **inputs, streamer=streamer, max_new_tokens=500
                )

                # If generate() raises, it never signals the end of the stream,
                # so iterating the streamer below would block forever. Always
                # close the stream and carry the error back to the main thread.
                generation_error: list[BaseException] = []

                def generate():
                    try:
                        model.model.generate(**generation_kwargs)
                    except BaseException as error:  # noqa: BLE001
                        generation_error.append(error)
                        streamer.end()

                thread = Thread(target=generate)
                thread.start()

                try:
                    for text in streamer:
                        if text:
                            print(
                                json.dumps({"type": "token", "content": text}),
                                flush=True,
                            )
                finally:
                    thread.join()

                if generation_error:
                    raise generation_error[0]

                print(json.dumps({"type": "done"}), flush=True)
            except Exception:
                import traceback

                print(
                    json.dumps({"type": "error", "content": traceback.format_exc()}),
                    flush=True,
                )

    except Exception:
        import traceback

        print(
            json.dumps({"type": "error", "content": traceback.format_exc()}), flush=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
