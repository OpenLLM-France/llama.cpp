#!/usr/bin/env python3
"""
Quantize a HuggingFace transformers model to NVFP4 (compressed-tensors format).

NVFP4 is NVIDIA's 4-bit float format (E2M1 elements, block size 16, FP8-E4M3
block scales + a per-tensor FP32 global scale). It runs natively on Blackwell
Tensor Cores (GB10, GB200, RTX 50xx).

Two schemes are offered (--outtype):

    nvfp4      w4a4  — weights AND activations in NVFP4. Maximum speed/size on
                       Blackwell. REQUIRES a calibration dataset (a short
                       forward pass computes the static activation scales).
                       This matches unistra-dnum/Luciole-1B-Instruct-1.1-NVFP4.

    nvfp4a16   w4a16 — weights in NVFP4, activations in 16-bit. DATA-FREE (no
                       calibration), like the FP8_DYNAMIC pass. Smaller model +
                       slightly higher quality than w4a4, but does not use
                       Blackwell's 4-bit activation path (less speedup).

The calibration set (nvfp4 only) is a mix of an English and a French chat
dataset, so the activation scales reflect Luciole's bilingual usage. Both are
overridable; French Alpaca-style rows are converted to chat messages.

Output is a standard HF transformers directory, servable by vLLM / SGLang on a
Blackwell GPU. Shared conversion plumbing (model loading, ignore patterns,
Mamba handling, writable-tree fixups) is reused from convert_hf_to_fp8.py.

Requires:
    pip install "llmcompressor<0.12" "transformers<5" datasets
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Reuse the FP8 converter's shared helpers. Importing it also installs its
# shutil.copy monkey-patch (needed so llmcompressor can overwrite the read-only
# .py modules it copies for trust_remote_code models). main() is guarded, so
# the import has no side effect beyond that patch.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import convert_hf_to_fp8 as fp8  # noqa: E402

logger = logging.getLogger("convert-hf-to-fp4")

# Calibration defaults (nvfp4 / w4a4 only). "id:split" — split optional.
DEFAULT_CALIB_EN = "HuggingFaceH4/ultrachat_200k:train_sft"  # multi-turn `messages`
DEFAULT_CALIB_FR = "jpacifico/French-Alpaca-dataset-Instruct-110K:train"  # alpaca fields
DEFAULT_CALIB_SAMPLES = 512
DEFAULT_CALIB_FRACTION_FR = 0.5
DEFAULT_MAX_SEQ_LENGTH = 2048


def resolve_scheme(outtype: str) -> str:
    if outtype in ("auto", "nvfp4"):
        return "NVFP4"
    if outtype == "nvfp4a16":
        return "NVFP4A16"
    raise ValueError(f"unknown --outtype value: {outtype}")


def default_output_path(model_path: str, model_name: str | None) -> Path:
    base = model_name or Path(model_path.rstrip("/")).name
    return Path(f"{base}-FP4")


def _row_to_messages(row: dict) -> list[dict] | None:
    """Normalise one dataset row into a chat `messages` list, or None to skip.

    Handles the two default layouts (and a couple of common variants):
      * ultrachat_200k : {'messages': [{role, content}, ...]}
      * French-Alpaca  : {'instruction', 'input', 'output'}
    """
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and "content" in msgs[0]:
        return msgs
    # Alpaca-style instruction/(input)/output -> a single user/assistant turn.
    instruction = row.get("instruction")
    if instruction:
        user = instruction
        extra = row.get("input")
        if extra:
            user = f"{instruction}\n{extra}"
        out = row.get("output") or row.get("response") or ""
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": out}]
    # Bare text fields -> single user turn.
    for key in ("text", "content", "prompt"):
        if row.get(key):
            return [{"role": "user", "content": row[key]}]
    return None


def _load_split(spec: str, default_split: str, n: int, seed: int):
    """Load `n` rows from a 'dataset_id[:split]' spec, shuffled."""
    from datasets import load_dataset

    if ":" in spec:
        ds_id, split = spec.split(":", 1)
    else:
        ds_id, split = spec, default_split
    logger.info("calibration: loading %d sample(s) from %s [%s]", n, ds_id, split)
    ds = load_dataset(ds_id, split=split)
    ds = ds.shuffle(seed=seed)
    if n < len(ds):
        ds = ds.select(range(n))
    return ds


def build_calibration_dataset(tokenizer, args):
    """Build a tokenized, chat-templated calibration set mixing EN + FR.

    Each row is rendered through the model's own chat template and tokenized to
    at most --max-seq-length, so the activation statistics match real prompts.
    """
    from datasets import Dataset

    total = args.calibration_samples
    n_fr = round(total * args.calibration_fraction_fr)
    n_en = total - n_fr

    rows = []
    for spec, default_split, n in (
        (args.calib_en, "train_sft", n_en),
        (args.calib_fr, "train", n_fr),
    ):
        if n <= 0:
            continue
        for row in _load_split(spec, default_split, n, args.seed):
            messages = _row_to_messages(row)
            if not messages:
                continue
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as e:  # e.g. a role the template rejects
                logger.debug("skipping a calibration row: %s", e)
                continue
            rows.append({"text": text})

    if not rows:
        raise RuntimeError(
            "calibration set is empty — check --calib-en / --calib-fr specs and "
            "that the model has a chat template."
        )

    ds = Dataset.from_list(rows).shuffle(seed=args.seed)

    def _tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=False,  # the chat template already added them
        )

    ds = ds.map(_tokenize, remove_columns=ds.column_names)
    logger.info("calibration: %d rendered example(s) (EN target %d, FR target %d)",
                len(ds), n_en, n_fr)
    return ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace transformers model to NVFP4 (compressed-tensors)"
    )
    # --- shared surface with convert_hf_to_fp8.py (kept identical) -----------
    parser.add_argument("--vocab-only", action="store_true",
                        help="only export config + tokenizer (no weights)")
    parser.add_argument("--outfile", type=Path,
                        help="output directory; default: <model-basename>-FP4")
    parser.add_argument("--outtype", type=str,
                        choices=["nvfp4", "nvfp4a16", "auto"], default="auto",
                        help="FP4 scheme: 'nvfp4' = w4a4 (needs calibration, matches the "
                             "reference); 'nvfp4a16' = weight-only w4a16 (data-free). "
                             "'auto' selects nvfp4.")
    parser.add_argument("model", type=str, nargs="?",
                        help="directory containing model files or HuggingFace repository ID")
    parser.add_argument("--model-name", type=str, default=None,
                        help="name used to derive the default output directory")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("--split-max-size", type=str, default="0",
                        help="max shard size for saved safetensors, e.g. 5G. '0' = library default.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without running the quantization")
    parser.add_argument("--print-supported-models", action="store_true",
                        help="print the supported model families and exit")
    parser.add_argument("--offload-folder", type=Path, default=None,
                        help="accelerate disk-offload dir (default: <outfile>/.offload)")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                        help="device for the quantization pass. w4a4 runs a calibration "
                             "forward pass, so 'cuda' is strongly recommended (CPU works but "
                             "is very slow). 'auto' = cuda if available.")
    parser.add_argument("--skip-mamba-fp8", "--skip-mamba", dest="skip_mamba", action="store_true",
                        help="for hybrid Mamba-Transformer models, keep every Linear inside the "
                             "Mamba mixer at the original precision (see convert_hf_to_fp8.py).")
    # accept-and-ignore flags so FP8/GGUF command lines stay interchangeable
    for ignored in ("--bigendian", "--use-temp-file", "--no-lazy", "--no-tensor-first-split",
                    "--remote", "--mmproj", "--mistral-format",
                    "--disable-mistral-community-chat-template",
                    "--sentence-transformers-dense-modules", "--fuse-gate-up-exps"):
        parser.add_argument(ignored, action="store_true", help="ignored")
    parser.add_argument("--split-max-tensors", type=int, default=0, help="ignored")
    parser.add_argument("--metadata", type=Path, help="ignored (GGUF-only)")

    # --- FP4-specific: calibration (nvfp4 / w4a4 only) -----------------------
    parser.add_argument("--calib-en", type=str, default=DEFAULT_CALIB_EN,
                        help=f"English calibration set 'id[:split]' (default: {DEFAULT_CALIB_EN})")
    parser.add_argument("--calib-fr", type=str, default=DEFAULT_CALIB_FR,
                        help=f"French calibration set 'id[:split]' (default: {DEFAULT_CALIB_FR})")
    parser.add_argument("--calibration-samples", type=int, default=DEFAULT_CALIB_SAMPLES,
                        help=f"total calibration samples (default: {DEFAULT_CALIB_SAMPLES})")
    parser.add_argument("--calibration-fraction-fr", type=float, default=DEFAULT_CALIB_FRACTION_FR,
                        help=f"fraction of calibration samples taken from French "
                             f"(default: {DEFAULT_CALIB_FRACTION_FR})")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                        help=f"max token length per calibration sample (default: {DEFAULT_MAX_SEQ_LENGTH})")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed for calibration sampling")

    args = parser.parse_args()
    if not args.print_supported_models and args.model is None:
        parser.error("the following arguments are required: model")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.print_supported_models:
        fp8.print_supported_models()
        return 0

    if args.vocab_only:
        # Reuse the FP8 vocab-only path but write to our -FP4 default dir.
        args.outfile = args.outfile or default_output_path(args.model, args.model_name)
        return fp8._run_vocab_only(args)

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        logger.error('Install with: pip install "llmcompressor<0.12" "transformers<5" datasets')
        return 3

    import transformers
    if int(transformers.__version__.split(".", 1)[0]) >= 5:
        logger.error(
            "transformers %s detected. This tool targets transformers v4 output for "
            'downstream compatibility. Reinstall with: pip install "llmcompressor<0.12" "transformers<5"',
            transformers.__version__,
        )
        return 4

    scheme = resolve_scheme(args.outtype)
    needs_calibration = scheme == "NVFP4"

    model_ref = args.model
    out_dir = args.outfile or default_output_path(model_ref, args.model_name)
    shard_size = fp8.translate_shard_size(args.split_max_size)
    offload_folder = args.offload_folder or (out_dir / ".offload")

    config = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
    is_multimodal = fp8.looks_multimodal(config.to_dict())
    model_type = (getattr(config, "model_type", "") or "").lower()
    is_mamba_hybrid = model_type in fp8.MAMBA_HYBRID_MODEL_TYPES

    ignore = ["lm_head"]
    if is_multimodal:
        ignore.extend(fp8.VISION_IGNORE_PATTERNS)
    if args.skip_mamba:
        if not is_mamba_hybrid:
            logger.warning("--skip-mamba passed but model_type=%r is not a known Mamba-hybrid; "
                           "extra Mamba ignores skipped.", model_type)
        else:
            mamba_patterns = fp8._mamba_hybrid_ignore_patterns(config)
            ignore.extend(mamba_patterns)
            logger.info("--skip-mamba: keeping %d Mamba layer(s) at original precision",
                        len(mamba_patterns))

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if needs_calibration and device == "cpu":
        logger.warning("w4a4 (nvfp4) runs a calibration forward pass; on CPU this is very slow. "
                       "Use --device cuda on the Blackwell box, or --outtype nvfp4a16 (data-free).")

    logger.info("model:            %s", model_ref)
    logger.info("output directory: %s", out_dir)
    logger.info("scheme:           %s (%s)", scheme,
                "w4a4, calibrated" if needs_calibration else "w4a16, data-free")
    logger.info("multimodal:       %s", is_multimodal)
    logger.info("ignored layers:   %s", ignore)
    logger.info("max shard size:   %s", shard_size or "library default")
    logger.info("device:           %s", device)
    if needs_calibration:
        logger.info("calibration:      %d samples (%.0f%% FR), EN=%s FR=%s, max_seq=%d",
                    args.calibration_samples, 100 * args.calibration_fraction_fr,
                    args.calib_en, args.calib_fr, args.max_seq_length)

    if args.dry_run:
        logger.info("--dry-run: nothing written.")
        return 0

    logger.info("loading model weights (this can take a while)...")
    model = fp8._load_model(model_ref, is_multimodal, offload_folder, device)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)

    recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)

    oneshot_kwargs: dict = {"model": model, "recipe": recipe}
    if needs_calibration:
        logger.info("building bilingual calibration set...")
        oneshot_kwargs["dataset"] = build_calibration_dataset(tokenizer, args)
        oneshot_kwargs["processor"] = tokenizer  # for collation of the tokenized set
        oneshot_kwargs["max_seq_length"] = args.max_seq_length
        oneshot_kwargs["num_calibration_samples"] = args.calibration_samples

    logger.info("running one-shot %s quantization...", scheme)
    oneshot(**oneshot_kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    fp8._make_tree_writable(out_dir)

    save_kwargs: dict = {"save_compressed": True}
    if shard_size is not None:
        save_kwargs["max_shard_size"] = shard_size
    logger.info("saving quantized model to %s", out_dir)
    model.save_pretrained(out_dir, **save_kwargs)

    try:
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        logger.warning("could not save tokenizer: %s", e)
    if is_multimodal:
        try:
            AutoProcessor.from_pretrained(model_ref, trust_remote_code=True).save_pretrained(out_dir)
        except Exception as e:
            logger.warning("could not save processor: %s", e)

    fp8._make_tree_writable(out_dir)
    if offload_folder.exists() and offload_folder.is_dir() and not any(offload_folder.iterdir()):
        try:
            offload_folder.rmdir()
        except OSError:
            pass

    logger.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
