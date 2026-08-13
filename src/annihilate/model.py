# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
# Copyright (C) 2025-2026  grimxlock + contributors (Annihilate fork)

import math
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Type, cast

import torch
import torch.linalg as LA
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,
)

from .config import KernelType, QuantizationMethod, RowNormalization, Settings
from .system import empty_cache
from .utils import Prompt, batchify, format_exception, print

# How many times more aligned with the refusal direction than chance an expert must
# be before EGA applies the full intervention. Below this it is scaled down
# proportionally, and an expert at chance alignment is left untouched.
#
# Expressed as a ratio to chance rather than an absolute one so it means the same
# thing on every model; see `_ega_scale`. The value reproduces the behaviour of the
# previous fixed multiplier at hidden size 4096, which is the width it happened to
# be calibrated for.
EGA_FULL_INTERVENTION_RATIO = 3.2


def _ega_scale(lora_A: Tensor, W: Tensor) -> float:
    """Scale factor for Expert-Granular Abliteration on one MoE expert.

    Returns 1.0 for an expert strongly aligned with the refusal direction and
    approaches 0.0 for one no better aligned than chance, so benign experts keep
    their knowledge.

    The raw ratio ``||v^T W|| / ||W||_F`` cannot be compared across models: for a
    direction no more aligned than chance it sits at ``1/sqrt(d_out)``, so a fixed
    multiplier scales an equally benign expert differently at every hidden size
    (0.62 at 1024 but 0.23 at 8192, measured). Dividing by that baseline makes the
    result dimensionless, with 1.0 meaning "no better than chance".
    """
    w_norm = torch.norm(W)
    if w_norm <= 0:
        return 1.0

    chance = 1.0 / math.sqrt(W.shape[0])
    alignment = (torch.norm(lora_A) / w_norm).item() / chance
    return min(1.0, alignment / EGA_FULL_INTERVENTION_RATIO)


class UnsupportedModelFormatError(ValueError):
    pass


def get_model_class(
    model: str,
) -> Type[AutoModelForImageTextToText] | Type[AutoModelForCausalLM]:
    configs = PretrainedConfig.get_config_dict(model)

    if any([("vision_config" in config) for config in configs]):
        return AutoModelForImageTextToText
    else:
        return AutoModelForCausalLM


# Pre-quantized formats we can load, mapped to the package that provides the
# kernels. Transformers dispatches on `quant_method` in the model's own
# config.json and raises if the backend is missing, so the value of detecting it
# up front is naming the missing package instead of failing deep inside a load.
QUANTIZATION_BACKENDS = {
    "compressed-tensors": "compressed-tensors",
    "fp8": "compressed-tensors",
    "bitsandbytes": "bitsandbytes",
    "gptq": "gptqmodel",
    "awq": "autoawq",
    "aqlm": "aqlm",
    "hqq": "hqq",
    "quanto": "optimum-quanto",
    "eetq": "eetq",
    "fbgemm_fp8": "fbgemm-gpu",
    "torchao": "torchao",
}


@dataclass
class ModelFormat:
    """What a model reference is, determined before any weights are loaded."""

    # Value of `quantization_config.quant_method`, or None for an unquantized model.
    quantization: str | None
    # Package providing the kernels, when we know which one it is.
    backend: str | None
    # True if the architecture is implemented in the repo rather than in
    # transformers, which means loading it executes code from that repo.
    remote_code: bool
    multimodal: bool
    architecture: str | None

    @property
    def is_prequantized(self) -> bool:
        return self.quantization is not None


def detect_model_format(model: str, **config_kwargs: Any) -> ModelFormat:
    """Inspect a model's config to learn what loading it will require.

    Reads config.json only. This is deliberately separate from loading so the
    program can report an unavailable backend, or a repo that would execute its
    own code, before spending minutes on a download.
    """
    configs = PretrainedConfig.get_config_dict(model, **config_kwargs)

    quantization = None
    for config in configs:
        quantization_config = config.get("quantization_config")
        if isinstance(quantization_config, dict):
            method = quantization_config.get("quant_method")
            if method is not None:
                quantization = str(method).lower()
                break

    architectures = next(
        (config["architectures"] for config in configs if config.get("architectures")),
        None,
    )

    return ModelFormat(
        quantization=quantization,
        backend=QUANTIZATION_BACKENDS.get(quantization) if quantization else None,
        remote_code=any(config.get("auto_map") for config in configs),
        multimodal=any("vision_config" in config for config in configs),
        architecture=architectures[0] if architectures else None,
    )


def _is_backend_installed(backend: str) -> bool:
    # Distribution name, not import name: these differ often enough
    # (compressed-tensors/compressed_tensors, autoawq/awq) that checking the
    # metadata is more reliable than guessing the module.
    try:
        version(backend)
    except PackageNotFoundError:
        return False
    return True


def is_gguf_reference(model: str) -> bool:
    normalized = model.lower()
    return normalized.endswith(".gguf") or normalized.endswith("-gguf")


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    # Set for multimodal models, None for text-only ones.
    processor: ProcessorMixin | None
    peft_config: LoraConfig
    dtype: torch.dtype

    def __init__(self, settings: Settings):
        self.settings = settings
        self.needs_reload = False
        self.trusted_models = set()

        self.revision_kwargs = {}
        if settings.model_commit is not None:
            self.revision_kwargs["revision"] = settings.model_commit

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        if is_gguf_reference(settings.model):
            raise UnsupportedModelFormatError(
                "GGUF models are not supported by Annihilate. "
                "Use the original Transformers/Hugging Face model repository "
                "with safetensors or PyTorch weights instead. GGUF files are "
                "for llama.cpp-style inference and cannot be abliterated with "
                "the PEFT/LoRA workflow."
            )

        self.format = detect_model_format(settings.model, **self.revision_kwargs)
        self._report_format()

        self.tokenizer = AutoTokenizer.from_pretrained(  # ty: ignore[invalid-assignment]
            settings.model,
            trust_remote_code=True
            if (settings.trust_remote_code or settings.model in self.trusted_models)
            else None,
            **self.revision_kwargs,
        )

        # Multimodal models have a processor we'll want to save.
        self.processor = None
        if get_model_class(settings.model) == AutoModelForImageTextToText:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    settings.model,
                    **self.revision_kwargs,
                )
            except Exception as e:
                print(
                    f"[yellow]Warning: Failed to load processor for {settings.model}: {e}[/]"
                )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fallback for tokenizers that don't declare a chat_template.
        if getattr(self.tokenizer, "chat_template", None) is None:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n' }}"
                "{% endif %}"
            )

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        self.model = None
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]...")

            try:
                quantization_config = self._get_quantization_config(dtype)

                extra_kwargs = {}
                # Only include quantization_config if it's not None
                # (some models like gpt-oss have issues with explicit None).
                if quantization_config is not None:
                    extra_kwargs["quantization_config"] = quantization_config

                self.model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=True
                    if (
                        settings.trust_remote_code
                        or settings.model in self.trusted_models
                    )
                    else None,
                    **self.revision_kwargs,
                    **extra_kwargs,
                )

                self.dtype = self.model.dtype

                # If we reach this point and the model requires trust_remote_code,
                # the user must have agreed when prompted to execute remote code,
                # because from_pretrained raises an exception otherwise.
                self.trusted_models.add(settings.model)

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(
                    [
                        Prompt(
                            system=settings.system_prompt,
                            user="What is 1+1?",
                        )
                    ],
                    max_new_tokens=1,
                )
            except Exception as error:
                self.model = None
                empty_cache()

                formatted = format_exception(error)
                if "\n" in formatted:
                    print(f"* [red]Failed:\n{formatted}[/]")
                else:
                    print(f"* [red]Failed ({formatted})[/]")

                continue

            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("* Quantized to 4-bit precision")

            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        self._apply_lora()

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")

        layer_count = len(self.get_layers())

        all_components: dict[str, int] = {}
        # How many layers each component was found on, which is not implied by the
        # module count: MoE layers contribute one module per expert.
        component_layers: dict[str, int] = {}
        for layer_index in range(layer_count):
            for component, modules in self.get_layer_modules(layer_index).items():
                all_components[component] = all_components.get(component, 0) + len(
                    modules
                )
                component_layers[component] = component_layers.get(component, 0) + 1

        print("* Abliterable components:")
        for component, count in all_components.items():
            covered = component_layers[component]
            coverage = (
                ""
                if covered == layer_count
                else f" (on {covered}/{layer_count} layers)"
            )
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total{coverage}")

        # Abliteration applies to whatever was targeted, so a component that is
        # absent - or present on only some layers - still yields a run that reports
        # success while leaving part of the refusal behaviour untouched. Say so here
        # rather than letting it surface later as an unexplained weak decensor.
        missing = [
            family
            for family, prefix in (("attention", "attn."), ("MLP", "mlp."))
            if not any(component.startswith(prefix) for component in all_components)
        ]
        partial = sorted(
            component
            for component, covered in component_layers.items()
            if covered < layer_count
        )

        if missing or partial:
            print()
            if missing:
                print(
                    f"[yellow]WARNING: No {' or '.join(missing)} modules could be "
                    "targeted in this model.[/]"
                )
            if partial:
                print(
                    "[yellow]WARNING: Some layers are not fully targeted: "
                    f"[bold]{', '.join(partial)}[/] missing from part of the model.[/]"
                )
            print(
                "[yellow]This architecture is only partially supported. Abliteration "
                "will still run, but the untargeted parts are left unchanged, so "
                "validate the output quality. Adding architecture-specific targeting "
                "to get_layer_modules() would cover the rest.[/]"
            )

    def _apply_lora(self):
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PreTrainedModel)

        # Always use LoRA adapters for abliteration (faster reload, no weight modification).
        # Collect actual leaf module names from the model for LoRA targeting.
        # This is more robust than splitting component keys (e.g. "attn.o_proj" -> "o_proj")
        # because hybrid models like Qwen3.5 MoE have modules with different names
        # across layers (e.g. "o_proj" on attention layers, "out_proj" on linear attention layers).
        target_modules_set: set[str] = set()

        module_id_to_full_name = {
            id(module): module_name
            for module_name, module in self.model.named_modules()
        }

        for layer_index in range(len(self.get_layers())):
            for modules in self.get_layer_modules(layer_index).values():
                for module in modules:
                    full_name = module_id_to_full_name.get(id(module))
                    if full_name is not None:
                        target_modules_set.add(full_name)

        target_modules = sorted(target_modules_set)

        if self.settings.row_normalization != RowNormalization.FULL:
            # Rank 1 is sufficient for directional ablation without renormalization.
            lora_rank = 1
        else:
            # Row magnitude preservation introduces nonlinear effects.
            lora_rank = self.settings.full_normalization_lora_rank

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct,
            # as VL models are typically just causal LMs with an added image encoder.
            task_type="CAUSAL_LM",
        )

        # self.peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        display_targets = sorted({name.rsplit(".", 1)[-1] for name in target_modules})
        print(
            f"* LoRA adapters initialized (target types: {', '.join(display_targets)})"
        )

    def _report_format(self) -> None:
        """Report what the model is, and fail early on anything we can't load."""
        fmt = self.format

        described = fmt.architecture or "unknown architecture"
        if fmt.multimodal:
            described += ", multimodal"
        if fmt.remote_code:
            described += ", custom code"
        print(f"* Detected [bold]{described}[/]")

        if fmt.remote_code and not self.settings.trust_remote_code:
            # from_pretrained raises on its own, but only after the download.
            print(
                "[yellow]WARNING: This model implements its architecture in its "
                "own repository. Loading it executes that code. Pass "
                "--trust-remote-code if you have reviewed it.[/]"
            )

        if not fmt.is_prequantized:
            return

        print(f"* Pre-quantized model: [bold]{fmt.quantization}[/]")

        if fmt.backend is None:
            print(
                f"[yellow]WARNING: Unrecognized quantization method "
                f"'{fmt.quantization}'. Loading will fail if transformers cannot "
                "dispatch it. Abliteration itself is format-agnostic, so a "
                "loadable model should still work.[/]"
            )
        elif not _is_backend_installed(fmt.backend):
            raise UnsupportedModelFormatError(
                f"{self.settings.model} is quantized with "
                f"'{fmt.quantization}', which needs the {fmt.backend} package. "
                f'Install it with "pip install {fmt.backend}", or use an '
                "unquantized version of this model."
            )

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        """
        Creates quantization config based on settings.

        Args:
            dtype: The dtype string (e.g., "auto", "bfloat16")

        Returns:
            BitsAndBytesConfig or None
        """
        if self.format.is_prequantized:
            # The model carries its own quantization_config. Passing a second one
            # either errors or silently re-quantizes already-quantized weights,
            # and the model needs no help from us to load.
            if self.settings.quantization == QuantizationMethod.BNB_4BIT:
                print(
                    f"[yellow]WARNING: Ignoring --quantization bnb_4bit because "
                    f"this model is already quantized ({self.format.quantization}).[/]"
                )
            return None

        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # BitsAndBytesConfig expects a torch.dtype, not a string.
            if dtype == "auto":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, dtype)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None

    def get_merged_model(self) -> PreTrainedModel:
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PeftModel)

        # Only for quantization *we* applied. Reloading without a
        # quantization_config recovers full precision precisely because the
        # config came from settings; a pre-quantized repo declares it in its own
        # config.json, so this path would just reload quantized weights. PEFT
        # dequantizes those during merge instead - see below.
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # Quantized models need special handling - we must reload the base model
            # in full precision to merge the LoRA adapters

            # Get the adapter state dict before we do anything
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=True
                if (
                    self.settings.trust_remote_code
                    or self.settings.model in self.trusted_models
                )
                else None,
                **self.revision_kwargs,
            )

            # Apply LoRA adapters to the CPU model
            print("* Applying LoRA adapters...")
            peft_model = get_peft_model(base_model, self.peft_config)

            # Copy the trained adapter weights
            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            # Merge and unload
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            return merged_model
        else:
            # Non-quantized model - can merge directly
            if self.format.is_prequantized:
                # Merging materializes each targeted layer at full precision, so
                # the export is no longer quantized and is substantially larger
                # than the repo it came from. Say so before it lands on disk.
                print(
                    f"[yellow]NOTE: Merging dequantizes this "
                    f"{self.format.quantization} model. The exported weights will "
                    "be full precision and larger than the original. Export as an "
                    "adapter instead to keep the quantized base.[/]"
                )
            print("* Merging LoRA adapters into base model...")
            merged_model = self.model.merge_and_unload()
            # merge_and_unload() modifies self.model in-place, destroying LoRA adapters.
            # Mark for full reload if user switches trials later.
            self.needs_reload = True
            return merged_model

    def reset_model(self):
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.
        """

        # If a prior model load was interrupted/cancelled mid-process, self.model will be None.
        current_model = None
        if self.model is not None:
            current_model = getattr(self.model.config, "name_or_path", None)

        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation).
            for name, module in self.model.named_modules():
                if "lora_B" in name and hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
            return

        # Purge existing model object from memory to make space.
        self.model = None
        empty_cache()

        quantization_config = self._get_quantization_config(
            str(self.dtype).split(".")[-1]
        )

        # Build kwargs, only include quantization_config if it's not None.
        extra_kwargs = {}
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config

        self.model = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            dtype=self.dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=True
            if (
                self.settings.trust_remote_code
                or self.settings.model in self.trusted_models
            )
            else None,
            **self.revision_kwargs,
            **extra_kwargs,
        )

        self._apply_lora()

        self.needs_reload = False

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers

        # Text-only models.
        return model.model.layers

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        layer = self.get_layers()[layer_index]

        modules = {}

        def try_add(component: str, module: Any):
            # Only add if it's a proper nn.Module (PEFT can wrap these with LoRA).
            if isinstance(module, Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)
            elif isinstance(module, Tensor):
                # The attribute exists at the path we expect but is a raw weight
                # rather than a module, which means the architecture moved. Raise
                # something `probe` will not swallow: silently skipping it would
                # abliterate a partially targeted model and report success.
                raise UnsupportedModelFormatError(
                    f"Unexpected Tensor at {component} in layer {layer_index} - "
                    "expected nn.Module. This model's architecture is not "
                    "supported by the current tensor targeting."
                )

        @contextmanager
        def probe():
            """Try one architecture-specific attribute path.

            A missing path is how this function tests for an architecture, so
            `AttributeError`/`IndexError`/`TypeError` are expected and ignored.
            `UnsupportedModelFormatError` is not: it means the path was found but
            held something unusable, which must reach the caller.

            The previous version suppressed every `Exception`, which included the
            `AssertionError` raised for that case, so the check meant to catch
            architecture changes could never fire.
            """
            try:
                yield
            except (AttributeError, IndexError, TypeError):
                pass

        any_layer: Any = layer

        # Standard self-attention out-projection (most models).
        with probe():
            try_add("attn.o_proj", any_layer.self_attn.o_proj)

        # Qwen3.5 MoE hybrid layers use GatedDeltaNet (linear attention) instead of
        # standard self-attention, so self_attn.o_proj doesn't exist on those layers.
        with probe():
            try_add("attn.o_proj", any_layer.linear_attn.out_proj)

        # Most dense models.
        with probe():
            try_add("mlp.down_proj", any_layer.mlp.down_proj)

        # Some MoE models (e.g. Qwen3).
        with probe():
            for expert in any_layer.mlp.experts:
                try_add("mlp.down_proj", expert.down_proj)

        # Phi-3.5-MoE (and possibly others).
        with probe():
            for expert in any_layer.block_sparse_moe.experts:
                try_add("mlp.down_proj", expert.w2)

        # LFM dense operator blocks.
        with probe():
            try_add("attn.o_proj", any_layer.conv.out_proj)

        with probe():
            try_add("mlp.down_proj", any_layer.feed_forward.w2)

        # LFM transformer blocks.
        with probe():
            try_add("attn.o_proj", any_layer.self_attn.out_proj)

        with probe():
            for expert in any_layer.feed_forward.experts:
                try_add("mlp.down_proj", expert.w2)

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with probe():
            try_add("mlp.down_proj", any_layer.shared_mlp.output_linear)

        # Granite MoE Hybrid - MoE layers with experts.
        with probe():
            for expert in any_layer.moe.experts:
                try_add("mlp.down_proj", expert.output_linear)

        # We need at least one module across all components for abliteration to work.
        total_modules = sum(len(mods) for mods in modules.values())
        if total_modules == 0:
            raise UnsupportedModelFormatError(
                f"No abliterable modules found in layer {layer_index}. This "
                "model's architecture needs architecture-specific tensor "
                "targeting added to get_layer_modules()."
            )

        return modules

    def get_abliterable_components(self) -> list[str]:
        components: set[str] = set()

        # Scan all layers because hybrid models (e.g. Qwen3.5 MoE) have different
        # components on different layers (some have self_attn, others linear_attn).
        for layer_index in range(len(self.get_layers())):
            components.update(self.get_layer_modules(layer_index).keys())

        return sorted(components)

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        if direction_index is None:
            refusal_direction = None
        else:
            # The index must be shifted by 1 because the first element
            # of refusal_directions is the direction for the embeddings.
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                params = parameters[component]

                # Type inference fails here for some reason.
                distance = abs(layer_index - params.max_weight_position)

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > params.min_weight_distance:
                    continue

                # Interpolate between max_weight and min_weight over min_weight_distance.
                if self.settings.kernel_type == KernelType.GAUSSIAN:
                    # Gaussian bell-curve interpolation
                    distance_norm = distance / max(
                        1e-5, params.min_weight_distance / 2.0
                    )
                    weight = params.min_weight + (
                        params.max_weight - params.min_weight
                    ) * math.exp(-0.5 * distance_norm**2)
                else:
                    # Linear interpolation
                    weight = params.max_weight + (
                        distance / params.min_weight_distance
                    ) * (params.min_weight - params.max_weight)

                # A weight of 0 disables this component's ablation. reset_model() has
                # already left the adapter at identity, so abort before the otherwise
                # wasteful decomposition (which would also be operating on a zero matrix).
                if weight == 0:
                    continue

                if refusal_direction is None:
                    # The index must be shifted by 1 because the first element
                    # of refusal_directions is the direction for the embeddings.
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                for module in modules:
                    # FIXME: This cast is potentially invalid, because the program logic
                    #        does not guarantee that the module is of type Linear, and in fact
                    #        the retrieved modules might not conform to the interface assumed
                    #        below (though they do in practice). However, this is difficult
                    #        to fix cleanly, because get_layer_modules is called twice on
                    #        different model configurations, and PEFT employs different
                    #        module types depending on the chosen quantization.
                    module = cast(Linear, module)

                    # LoRA abliteration: delta W = -lambda * v * (v^T W)
                    # lora_B = -lambda * v
                    # lora_A = v^T W

                    # Use the FP32 refusal direction directly (no downcast/upcast)
                    # and move to the correct device.
                    v = layer_refusal_direction.to(module.weight.device)

                    # Get W (dequantize if necessary).
                    #
                    # FIXME: This cast is valid only under the assumption that the original
                    #        module wrapped by the LoRA adapter has a weight attribute.
                    #        See the comment above for why this is currently not guaranteed.
                    base_weight = cast(Tensor, module.base_layer.weight)
                    quant_state = getattr(base_weight, "quant_state", None)

                    if quant_state is None:
                        W = base_weight.to(torch.float32)
                    else:
                        # 4-bit quantization.
                        # This cast is always valid. Type inference fails here because the
                        # bnb.functional module is not found by ty for some reason.
                        import bitsandbytes as bnb

                        W = cast(
                            Tensor,
                            bnb.functional.dequantize_4bit(  # ty: ignore[possibly-missing-submodule]
                                base_weight.data,
                                quant_state,
                            ).to(torch.float32),
                        )

                    # Flatten weight matrix to (out_features, in_features).
                    W = W.view(W.shape[0], -1)

                    if self.settings.row_normalization != RowNormalization.NONE:
                        # Keep a reference to the original weight matrix so we can subtract it later.
                        W_org = W
                        # Get the row norms.
                        W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                        # Normalize the weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)

                    # Calculate lora_A = v^T W
                    # v is (d_out,), W is (d_out, d_in)
                    # v @ W -> (d_in,)
                    lora_A = (v @ W).view(1, -1)

                    current_weight = weight
                    if getattr(self.settings, "use_ega", False) and len(modules) > 1:
                        # MoE Expert-Granular Abliteration: modify experts that carry
                        # the refusal direction, leave benign knowledge experts alone.
                        current_weight = current_weight * _ega_scale(lora_A, W)

                    # Calculate lora_B = -current_weight * v
                    # v is (d_out,)
                    lora_B = (-current_weight * v).view(-1, 1)

                    if self.settings.row_normalization == RowNormalization.PRE:
                        # Make the LoRA adapter apply to the original weight matrix.
                        lora_B = W_row_norms * lora_B
                    elif self.settings.row_normalization == RowNormalization.FULL:
                        # Approximates https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
                        W = W + lora_B @ lora_A
                        # Normalize the adjusted weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)
                        # Restore the original row norms of the weight matrix.
                        W = W * W_row_norms
                        # Subtract the original matrix to turn W into a delta.
                        W = W - W_org
                        # Use a low-rank SVD to get an approximation of the matrix.
                        r = self.peft_config.r
                        # svd_lowrank is randomized:
                        # https://github.com/pytorch/pytorch/blob/20919052303c0b5ba87f8bf7e19237dc33ab09d3/torch/_lowrank.py#L108-L109
                        # Reseed immediately before the call so restoring a trial is independent of RNG history.
                        torch.manual_seed(self.settings.seed)
                        U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
                        # Truncate it to the part we want to store in the LoRA adapter.
                        # Note: svd_lowrank actually returns V, so transpose it to get Vh.
                        U = U[:, :r]
                        S = S[:r]
                        Vh = Vh[:, :r].T
                        # Transfer it into the LoRA adapter components. Split the singular values
                        # evenly between the two components to keep their norms balanced and avoid
                        # potential issues with numerical stability.
                        sqrt_S = torch.sqrt(S)
                        lora_B = U @ torch.diag(sqrt_S)
                        lora_A = torch.diag(sqrt_S) @ Vh

                    # Assign to adapters. The adapter name is "default", because that's
                    # what PEFT uses when no name is explicitly specified, as above.
                    # These casts are therefore valid.
                    weight_A = cast(Tensor, module.lora_A["default"].weight)
                    weight_B = cast(Tensor, module.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]

        # This cast is valid because list[str] is the return type
        # for batched operation with tokenize=False.
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        if self.settings.response_prefix:
            # Append the common response prefix to the prompts so that evaluation happens
            # at the point where responses start to differ for different prompts.
            chat_prompts = [
                prompt + self.settings.response_prefix for prompt in chat_prompts
            ]

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        return self.tokenizer.batch_decode(
            # Extract the newly generated part.
            # This cast is valid because the input_ids property is a Tensor
            # if the tokenizer is invoked with return_tensors="pt", as above.
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(
                batch,
                skip_special_tokens=skip_special_tokens,
            ):
                responses.append(response)

        return responses

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # KV cache is unnecessary here because we only need the hidden states
            # for the first generated token.
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Hidden states for the first (only) generated token.
        # This cast is valid because we passed output_hidden_states=True above.
        hidden_states = cast(tuple[tuple[FloatTensor]], outputs.hidden_states)[0]

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(
            # layer_hidden_states has shape (prompt, position, component),
            # so this extracts the hidden states at the end of each prompt,
            # and stacks them up over the layers.
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        residuals = residuals.to(torch.float32)

        if 0 <= self.settings.winsorization_quantile < 1:
            # Apply symmetric winsorization to each layer of the per-prompt residuals.
            abs_residuals = torch.abs(residuals)
            # Get the (prompt, layer, 1) quantiles of the (prompt, layer, component) residuals.
            thresholds = torch.quantile(
                abs_residuals,
                self.settings.winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)

        if self.settings.offload_outputs_to_cpu:
            residuals = residuals.cpu()
            empty_cache()

        return residuals

    def get_residuals_batched(self, prompts: list[Prompt]) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))

        return torch.cat(residuals, dim=0)

    def get_residuals_mean(self, prompts: list[Prompt]) -> Tensor:
        if not prompts:
            raise ValueError("prompts must not be empty")

        running_sum = None
        total_count = 0

        for batch in batchify(prompts, self.settings.batch_size):
            batch_residuals = self.get_residuals(batch)

            # Accumulate in high precision on CPU to reduce peak VRAM usage.
            batch_sum = batch_residuals.sum(dim=0, dtype=torch.float64).cpu()

            if running_sum is None:
                running_sum = batch_sum
            else:
                running_sum += batch_sum

            total_count += batch_residuals.shape[0]

        assert running_sum is not None

        return (running_sum / total_count).to(torch.float32)

    # We work with logprobs rather than probabilities for numerical stability
    # when computing the KL divergence.
    def get_logprobs(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the (log) probability distributions
        # over the vocabulary at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # Use raw logits, not processed generation scores; processors can insert
        # -inf for suppressed tokens, which can make KL divergence evaluate to NaN.
        # This cast is valid because we passed output_logits=True above.
        logits = cast(tuple[FloatTensor], outputs.logits)[0]

        # The returned tensor has shape (prompt, token).
        logprobs = F.log_softmax(logits, dim=-1)

        if self.settings.offload_outputs_to_cpu:
            del outputs, logits
            logprobs = logprobs.cpu()
            empty_cache()

        return logprobs

    def get_logprobs_batched(self, prompts: list[Prompt]) -> Tensor:
        logprobs = []

        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch))

        return torch.cat(logprobs, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]], streamer=None) -> str:
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        if streamer is None:
            streamer = TextStreamer(
                # The TextStreamer constructor annotates this parameter with the AutoTokenizer
                # type, which makes no sense because AutoTokenizer is a factory class,
                # not a base class that tokenizers inherit from.
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        # This cast is valid because str is the return type
        # when passing a sequence of token IDs.
        return cast(
            str,
            self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
        )
