#!/usr/bin/env python3
"""
Quantize a HuggingFace transformers model to FP8 (compressed-tensors format).

Mirrors the CLI surface of convert_hf_to_gguf.py so command lines are broadly
interchangeable. Arguments that have no meaning for an FP8 pass are accepted
and ignored (documented in each --help entry).

Output is a standard HF transformers directory (config.json + safetensors +
tokenizer.*). It can be loaded with:

    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained("<outdir>")

or served natively by vLLM / SGLang / TGI.

Requires:
    pip install "llmcompressor<0.12" "transformers<5"
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("convert-hf-to-fp8")


# Monkey-patch shutil.copy so every copied file gets owner-write. llmcompressor's
# copy_python_files_from_model_cache (helpers.py:74) uses shutil.copy on cached
# HF `.py` modules whose source mode is often 0464 or 0444 (read-only for owner).
# When llmcompressor then re-invokes the same copy path during save_pretrained,
# open('wb') on the read-only destination raises PermissionError. Patching here
# ensures the destination is always writable regardless of source mode. Must run
# before any llmcompressor import; llmcompressor calls `shutil.copy(...)` by
# module-attribute lookup, so replacing shutil.copy takes effect immediately.
_orig_shutil_copy = shutil.copy


def _copy_writable(src, dst, *args, **kwargs):
    result = _orig_shutil_copy(src, dst, *args, **kwargs)
    try:
        st = os.stat(result)
        os.chmod(result, st.st_mode | 0o200)
    except OSError:
        pass
    return result


shutil.copy = _copy_writable


# vision encoders and multimodal projectors are activation-sensitive; keeping
# them in the original precision preserves quality with negligible size cost.
VISION_IGNORE_PATTERNS = [
    "re:.*vision_tower.*",
    "re:.*vision_model.*",
    "re:.*visual\\..*",
    "re:.*multi_modal_projector.*",
    "re:.*mm_projector.*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace transformers model to FP8 (compressed-tensors)"
    )
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="only export config + tokenizer (no weights)",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="output directory; default: <model-basename>-FP8",
    )
    parser.add_argument(
        "--outtype", type=str,
        choices=["fp8", "fp8_dynamic", "auto"], default="auto",
        help=(
            "FP8 scheme: 'fp8_dynamic' uses static per-channel weights + dynamic "
            "per-token activations (no calibration needed); 'fp8' uses static "
            "per-tensor activations (requires calibration; not implemented here). "
            "'auto' selects fp8_dynamic."
        ),
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="ignored (safetensors is endian-agnostic)",
    )
    parser.add_argument(
        "model", type=str, nargs="?",
        help="directory containing model files or HuggingFace repository ID",
    )
    parser.add_argument(
        "--use-temp-file", action="store_true",
        help="ignored (no bespoke temp-file path here)",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="ignored (weights are always fully materialized for FP8 quantization)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="name used to derive the default output directory",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--split-max-tensors", type=int, default=0,
        help="ignored (safetensors sharding uses --split-max-size)",
    )
    parser.add_argument(
        "--split-max-size", type=str, default="0",
        help="max shard size for saved safetensors, e.g. 5G. '0' = library default (5GB).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the plan without running the quantization",
    )
    parser.add_argument(
        "--no-tensor-first-split", action="store_true",
        help="ignored",
    )
    parser.add_argument(
        "--metadata", type=Path,
        help="ignored (GGUF-only)",
    )
    parser.add_argument(
        "--print-supported-models", action="store_true",
        help="print the supported model families and exit",
    )
    parser.add_argument(
        "--remote", action="store_true",
        help=(
            "treat the model argument as a HuggingFace repo id. In practice this "
            "flag is not required: local paths and repo ids are both accepted."
        ),
    )
    parser.add_argument(
        "--mmproj", action="store_true",
        help="ignored (vision components are auto-detected and kept in original precision)",
    )
    parser.add_argument(
        "--mistral-format", action="store_true",
        help="not supported (this tool consumes HF transformers format only)",
    )
    parser.add_argument(
        "--disable-mistral-community-chat-template", action="store_true",
        help="ignored",
    )
    parser.add_argument(
        "--sentence-transformers-dense-modules", action="store_true",
        help="ignored",
    )
    parser.add_argument(
        "--fuse-gate-up-exps", action="store_true",
        help="ignored (llm-compressor handles MoE gate/up layers automatically)",
    )
    parser.add_argument(
        "--offload-folder", type=Path, default=None,
        help=(
            "directory used by accelerate for disk offload when the model does "
            "not fit in RAM+VRAM. Default: <outfile>/.offload. Needs enough free "
            "space to hold the parts of the model that don't fit in memory."
        ),
    )
    parser.add_argument(
        "--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
        help=(
            "device used to run the quantization pass. 'cpu' avoids VRAM entirely "
            "(FP8_DYNAMIC is a data-free / weight-only pass, so no forward runs) "
            "and is the reliable fallback when GPU quantization OOMs. Requires "
            "roughly 2 x (model size in bytes) of RAM. 'auto' = cuda if available."
        ),
    )

    args = parser.parse_args()
    if not args.print_supported_models and args.model is None:
        parser.error("the following arguments are required: model")
    return args


def resolve_scheme(outtype: str) -> str:
    if outtype in ("auto", "fp8_dynamic"):
        return "FP8_DYNAMIC"
    if outtype == "fp8":
        return "FP8"
    raise ValueError(f"unknown --outtype value: {outtype}")


def default_output_path(model_path: str, model_name: str | None) -> Path:
    base = model_name or Path(model_path.rstrip("/")).name
    return Path(f"{base}-FP8")


def translate_shard_size(spec: str) -> str | None:
    """Translate the GGUF split-size grammar ('5G', '500M', '1024K', or a raw
    byte count) into what transformers.save_pretrained expects ('5GB', '500MB',
    ...). Returns None to mean 'use the library default'."""
    if not spec or spec == "0":
        return None
    m = re.fullmatch(r"(\d+)\s*([KMG]?)", spec.strip())
    if not m:
        raise ValueError(f"invalid --split-max-size: {spec!r}")
    n, unit = m.group(1), m.group(2)
    if unit == "":
        return n
    return f"{n}{unit}B"


def looks_multimodal(config_dict: dict) -> bool:
    return any(
        k in config_dict
        for k in ("vision_config", "vision_tower_config", "vision_model_config",
                  "audio_config", "speech_config")
    )


def print_supported_models() -> None:
    print(
        "llm-compressor's FP8_DYNAMIC pass works on any HuggingFace transformers\n"
        "model whose weights live in torch.nn.Linear modules. In practice this\n"
        "covers:\n"
        "  - Llama 1/2/3/3.1/3.2/4 and derivatives (Vicuna, WizardLM, ...)\n"
        "  - Mistral 7B/Small/Nemo/Large, Mixtral, Codestral\n"
        "  - Qwen 1/2/2.5/3 (dense + MoE), Qwen-VL\n"
        "  - Gemma 1/2/3\n"
        "  - Phi 2/3/3.5/4\n"
        "  - DeepSeek V2/V3, DeepSeek-Coder, DeepSeek-VL\n"
        "  - Falcon, MPT, StarCoder, InternLM, Yi, Baichuan, CommandR\n"
        "  - Vision-language variants of the above (vision tower is kept in the\n"
        "    original precision, only the LLM Linears are quantized)\n"
    )


def _load_model(model_ref: str, is_multimodal: bool, offload_folder: Path, device: str):
    """Pick a suitable Auto* class. For image-text models the CausalLM head is
    wrapped by a conditional-generation class; instantiating via
    AutoModelForCausalLM would strip the vision tower.

    device 'cpu' places the whole model on CPU (no VRAM, no offload thrashing).
    device 'cuda'/'auto' let accelerate spread across GPUs, spilling to
    offload_folder when RAM+VRAM aren't enough."""
    from transformers import AutoModelForCausalLM

    kw = dict(torch_dtype="auto", trust_remote_code=True)
    if device == "cpu":
        kw["device_map"] = {"": "cpu"}
    else:
        offload_folder.mkdir(parents=True, exist_ok=True)
        kw["device_map"] = "auto"
        kw["offload_folder"] = str(offload_folder)

    if is_multimodal:
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(model_ref, **kw)
        except (ImportError, ValueError):
            pass
        try:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq.from_pretrained(model_ref, **kw)
        except (ImportError, ValueError):
            pass
    return AutoModelForCausalLM.from_pretrained(model_ref, **kw)


def _run_vocab_only(args: argparse.Namespace) -> int:
    from transformers import AutoConfig, AutoTokenizer

    out_dir = args.outfile or default_output_path(args.model, args.model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    AutoConfig.from_pretrained(args.model, trust_remote_code=True).save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(args.model, trust_remote_code=True).save_pretrained(out_dir)
    logger.info("wrote config + tokenizer to %s", out_dir)
    return 0


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.print_supported_models:
        print_supported_models()
        return 0

    if args.mistral_format:
        logger.error(
            "--mistral-format is not supported: this tool consumes HuggingFace "
            "transformers format only. Convert the model to HF format first, or "
            "quantize the mistralai/... HF sibling repository."
        )
        return 2

    if args.vocab_only:
        return _run_vocab_only(args)

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        logger.error('Install with: pip install "llmcompressor<0.12" "transformers<5"')
        return 3

    # transformers v5 writes rope_scaling in a form that transformers v4
    # loaders reject; refuse to run so we can't silently produce a checkpoint
    # that our downstream v4 environments would misload.
    import transformers
    tfx_major = int(transformers.__version__.split(".", 1)[0])
    if tfx_major >= 5:
        logger.error(
            "transformers %s detected. This tool targets transformers v4 output "
            "for downstream compatibility. Reinstall with: "
            'pip install "llmcompressor<0.12" "transformers<5"',
            transformers.__version__,
        )
        return 4

    scheme = resolve_scheme(args.outtype)
    if scheme == "FP8" and not args.dry_run:
        logger.error(
            "--outtype fp8 (static activations) requires a calibration dataset "
            "and is not implemented here. Use --outtype fp8_dynamic (default)."
        )
        return 2

    model_ref = args.model
    out_dir = args.outfile or default_output_path(model_ref, args.model_name)
    shard_size = translate_shard_size(args.split_max_size)
    offload_folder = args.offload_folder or (out_dir / ".offload")

    config = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
    is_multimodal = looks_multimodal(config.to_dict())

    ignore = ["lm_head"]
    if is_multimodal:
        ignore.extend(VISION_IGNORE_PATTERNS)

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    logger.info("model:            %s", model_ref)
    logger.info("output directory: %s", out_dir)
    logger.info("scheme:           %s", scheme)
    logger.info("multimodal:       %s", is_multimodal)
    logger.info("ignored layers:   %s", ignore)
    logger.info("max shard size:   %s", shard_size or "library default")
    logger.info("device:           %s", device)
    if device != "cpu":
        logger.info("offload folder:   %s", offload_folder)

    if args.dry_run:
        logger.info("--dry-run: nothing written.")
        return 0

    logger.info("loading model weights (this can take a while)...")
    model = _load_model(model_ref, is_multimodal, offload_folder, device)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=ignore,
    )

    logger.info("running one-shot FP8 quantization...")
    oneshot(model=model, recipe=recipe)

    out_dir.mkdir(parents=True, exist_ok=True)

    # llmcompressor copies custom `.py` modules (trust_remote_code models) from
    # the HF on-disk cache using shutil.copy, which preserves the source's 0444
    # mode. Then it re-copies them during save_pretrained, and open('wb') fails
    # on the read-only destination. Pre-emptively make any pre-existing files
    # writable so the second copy can overwrite them.
    _make_tree_writable(out_dir)

    save_kwargs: dict = {"save_compressed": True}
    if shard_size is not None:
        save_kwargs["max_shard_size"] = shard_size

    logger.info("saving quantized model to %s", out_dir)
    model.save_pretrained(out_dir, **save_kwargs)

    try:
        AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True).save_pretrained(out_dir)
    except Exception as e:
        logger.warning("could not save tokenizer: %s", e)

    if is_multimodal:
        try:
            AutoProcessor.from_pretrained(model_ref, trust_remote_code=True).save_pretrained(out_dir)
        except Exception as e:
            logger.warning("could not save processor: %s", e)

    # Restore normal mode on the .py files copied from cache so a subsequent
    # rerun into the same directory doesn't hit the read-only wall again.
    _make_tree_writable(out_dir)

    # accelerate created the offload dir but may have written nothing there
    # (or already migrated everything out). Remove it if empty so it doesn't
    # clutter the model card.
    if offload_folder.exists() and offload_folder.is_dir() and not any(offload_folder.iterdir()):
        try:
            offload_folder.rmdir()
        except OSError:
            pass

    logger.info("done.")
    return 0


def _make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for f in root.rglob("*"):
        if f.is_file() or f.is_dir():
            try:
                mode = f.stat().st_mode
                f.chmod(mode | 0o200)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
