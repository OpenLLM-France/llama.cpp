#!/usr/bin/env python3
"""
Check that an FP8 conversion (produced by convert_hf_to_fp8.py) is functionally
equivalent to its source model.

Usage:
    python test_fp8.py <original_model_dir> <fp8_model_dir> [--vllm]
                       [--max-new-tokens N] [--num-prompts N]
                       [--min-agreement 0.85] [--seed 42]

Method:
    - Load both models via transformers.AutoModelForCausalLM (transformers
      auto-dequantizes the compressed-tensors FP8 checkpoint on load).
    - For a fixed set of chat-templated prompts, generate greedy continuations.
    - Compare first-N tokens between original and FP8 output. FP8_DYNAMIC on a
      well-behaved model is expected to match the source for the first ~30
      greedy tokens on most prompts; small early divergences are treated as
      warnings, and the overall pass criterion is per-prompt token-agreement
      >= --min-agreement (default 0.85).
    - With --vllm, additionally run vLLM on the FP8 model and check it matches
      the transformers FP8 outputs.

Exit codes:
    0  all checks passed
    1  agreement below threshold on at least one prompt
    2  missing dependency / setup error
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("test-fp8")


DEFAULT_PROMPTS = [
    "Qu'est-ce qu'une éclipse solaire ? Réponds brièvement.",
    "List three uses of the number pi in engineering.",
    "Écris une phrase courte pour expliquer la photosynthèse.",
    "In one sentence: why is the sky blue?",
    "Cite un philosophe des Lumières et son idée principale.",
]


@dataclass
class Generation:
    prompt: str
    text: str
    token_ids: list[int]


def _make_mamba_cpu_safe() -> None:
    """Route mamba_ssm's gated-RMSNorm through its own pure-torch reference on CPU.

    Nemotron-H's `torch_forward` (the CPU-friendly branch) still routes its
    gated RMSNorm through mamba_ssm's Triton kernel, which raises
    'Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)' on
    CPU tensors. Dispatch: CUDA inputs → original Triton kernel; CPU inputs →
    `rms_norm_ref`, the reference implementation shipped in the same module.

    Called at the top of `load_hf_model`, before any modeling module imports
    `rmsnorm_fn`, so the local binding picks up the wrapper. Idempotent."""
    import inspect
    try:
        from mamba_ssm.ops.triton import layernorm_gated
    except ImportError:
        return
    if getattr(layernorm_gated, "_cpu_safe_patched", False):
        return

    _orig_rmsnorm_fn = layernorm_gated.rmsnorm_fn
    _rms_norm_ref = layernorm_gated.rms_norm_ref
    # Older mamba_ssm rmsnorm_fn is 7 params (no is_rms_norm — hard-coded True
    # since this is the RMS variant); newer takes 8. Detect once so we only
    # forward args the installed version accepts.
    _orig_supports_is_rms_norm = (
        "is_rms_norm" in inspect.signature(_orig_rmsnorm_fn).parameters
    )

    def rmsnorm_fn_dispatch(x, weight, bias=None, z=None, eps=1e-6,
                            group_size=None, norm_before_gate=True,
                            is_rms_norm=True):
        if x.is_cuda:
            if _orig_supports_is_rms_norm:
                return _orig_rmsnorm_fn(x, weight, bias, z, eps, group_size,
                                        norm_before_gate, is_rms_norm)
            return _orig_rmsnorm_fn(x, weight, bias, z, eps, group_size,
                                    norm_before_gate)
        return _rms_norm_ref(x, weight, bias, z, eps, group_size,
                             norm_before_gate, is_rms_norm)

    layernorm_gated.rmsnorm_fn = rmsnorm_fn_dispatch
    layernorm_gated._cpu_safe_patched = True


def _make_cuda_stream_cpu_safe() -> None:
    """Make `torch.cuda.default_stream` accept CPU devices.

    Nemotron-H's custom modeling wraps its Mamba path in
    `with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):`
    unconditionally. On CPU, `default_stream("cpu")` raises
    `ValueError: Expected a cuda device`. `torch.cuda.stream(None)` is already
    a no-op context, so we only need to make `default_stream` return None on
    non-CUDA devices. Idempotent."""
    import torch
    if getattr(torch.cuda, "_cpu_safe_patched", False):
        return
    _orig_default_stream = torch.cuda.default_stream

    def default_stream(device=None):
        if device is not None:
            try:
                if torch.device(device).type != "cuda":
                    return None
            except (TypeError, ValueError):
                pass
        return _orig_default_stream(device)

    torch.cuda.default_stream = default_stream
    torch.cuda._cpu_safe_patched = True


def load_hf_model(model_dir: Path, device: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _make_cuda_stream_cpu_safe()
    _make_mamba_cpu_safe()
    logger.info("loading %s (device=%s) ...", model_dir, device)
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    kw = dict(dtype="auto", trust_remote_code=True)
    kw["device_map"] = {"": "cpu"} if device == "cpu" else "auto"
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kw)
    model.eval()
    return model, tok


def apply_template(tok, user_msg: str) -> str:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    logger.warning("no chat_template found on tokenizer; using raw prompt")
    return user_msg


def generate_hf(model, tok, prompt_text: str, max_new_tokens: int) -> Generation:
    import torch
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )
    new_ids = out[0, input_len:].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    return Generation(prompt=prompt_text, text=text, token_ids=new_ids)


def generate_vllm(model_dir: Path, prompts: list[str], max_new_tokens: int,
                  gpu_memory_utilization: float = 0.5,
                  max_model_len: int = 4096) -> list[Generation]:
    from vllm import LLM, SamplingParams
    logger.info("loading %s in vLLM (gpu_memory_utilization=%.2f, max_model_len=%d) ...",
                model_dir, gpu_memory_utilization, max_model_len)
    llm = LLM(
        model=str(model_dir),
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens),
    )
    gens: list[Generation] = []
    for prompt, out in zip(prompts, outputs):
        completion = out.outputs[0]
        gens.append(Generation(
            prompt=prompt,
            text=completion.text,
            token_ids=list(completion.token_ids),
        ))
    return gens


def token_agreement(a: list[int], b: list[int]) -> tuple[float, int]:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0
    matches = 0
    diverged_at = n
    for i in range(n):
        if a[i] == b[i]:
            matches += 1
        else:
            diverged_at = i
            break
    return matches / n, diverged_at


def compare(label_a: str, gens_a: list[Generation], label_b: str, gens_b: list[Generation],
            min_agreement: float) -> bool:
    print()
    print(f"=== {label_a}  vs  {label_b} ===")
    all_pass = True
    for ga, gb in zip(gens_a, gens_b):
        agree, first_diff = token_agreement(ga.token_ids, gb.token_ids)
        status = "PASS" if agree >= min_agreement else "FAIL"
        if agree < min_agreement:
            all_pass = False
        prompt_preview = ga.prompt.strip().replace("\n", " ")[:60]
        print(f"[{status}] agreement={agree:.2%}  first_diff@{first_diff}  prompt={prompt_preview!r}")
        if agree < 1.0:
            print(f"       {label_a}: {ga.text[:120]!r}")
            print(f"       {label_b}: {gb.text[:120]!r}")
    return all_pass


def _fit_vllm_util(desired: float, safety: float = 0.9) -> float:
    """Shrink gpu_memory_utilization to what is actually free on device 0.

    vLLM's `gpu_memory_utilization` is a fraction of *total* device memory,
    not of free memory, and it refuses to start if it can't reserve that
    much. On unified-memory boxes (DGX Spark) the "free" fraction of the
    120 GB pool can be well below the requested 0.5 when other processes
    (or the OS page cache) are holding memory. Cap the fraction at
    (free/total)*safety so vLLM always has a small headroom over what it
    is guaranteed to obtain. Returns the desired value untouched when
    CUDA is unavailable or the query fails."""
    try:
        import torch
        if not torch.cuda.is_available():
            return desired
        free, total = torch.cuda.mem_get_info(0)
    except Exception:
        return desired
    if total <= 0:
        return desired
    max_util = (free / total) * safety
    if max_util >= desired:
        return desired
    logger.warning(
        "shrinking vLLM gpu_memory_utilization %.2f -> %.2f "
        "(free=%.1f GiB, total=%.1f GiB, safety=%.2f)",
        desired, max_util, free / 2**30, total / 2**30, safety,
    )
    return max_util


def _release_cuda_memory() -> None:
    """Drop the caching allocator's block pool back to the driver.

    torch.cuda.empty_cache() alone is not enough — it releases *cached* blocks
    but the allocator may still hold reserved chunks. gc.collect() first drops
    any lingering Python-level references; then empty_cache + reset_peak +
    ipc_collect actually returns memory to the driver so a subsequent
    subprocess (like vLLM's engine core) sees it as free."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _wait_cuda_memory_stable(max_wait_s: float = 5.0,
                             poll_interval_s: float = 0.2) -> None:
    """Poll `mem_get_info` until free memory stops growing, or timeout.

    torch.cuda.empty_cache() releases blocks *asynchronously*; the CUDA driver
    may take several hundred milliseconds to fully unmap. If vLLM spawns its
    engine subprocess while the parent is still draining, vLLM's first memory
    snapshot is lower than the memory available during its later profiling
    call, and it fails with:
        AssertionError: Error in memory profiling.
        Initial free memory X, current free memory Y.
        (Y > X → the parent kept releasing.)
    Wait for two consecutive stable readings before letting vLLM start."""
    import time
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    prev = -1
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        free, _ = torch.cuda.mem_get_info(0)
        if free == prev:
            return
        prev = free
        time.sleep(poll_interval_s)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify FP8-converted model against its source")
    p.add_argument("original", type=Path, help="folder of the original (BF16/FP16) HF model")
    p.add_argument("fp8", type=Path, help="folder of the FP8-converted HF model")
    p.add_argument("--vllm", action="store_true", help="also run inference with vLLM on the FP8 model")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="device used to load the two HF models. Use 'cpu' to bypass "
                        "Triton/mamba-ssm compatibility issues on Blackwell — much "
                        "slower but sidesteps the fast path entirely.")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5,
                   help="fraction of GPU memory vLLM is allowed to reserve at startup. "
                        "Default 0.5 is conservative and safe when the transformers run "
                        "in the same process still holds cache. Raise to 0.9 on a dedicated GPU. "
                        "Automatically shrunk to fit actual free memory when needed.")
    p.add_argument("--vllm-max-model-len", type=int, default=4096,
                   help="max context length passed to vLLM. Smaller = less KV cache reserved. "
                        "Default 4096 is plenty for the short-generation checks this test does; "
                        "raise it only if a specific prompt/generation is longer.")
    p.add_argument("--max-new-tokens", type=int, default=30, help="tokens generated per prompt (default: 30)")
    p.add_argument("--num-prompts", type=int, default=3, help="number of prompts sampled from the pool (default: 3)")
    p.add_argument("--min-agreement", type=float, default=0.85,
                   help="minimum per-prompt token-agreement for PASS (default: 0.85)")
    p.add_argument("--seed", type=int, default=42, help="random seed for prompt selection (default: 42)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for p in (args.original, args.fp8):
        if not p.is_dir():
            logger.error("not a directory: %s", p)
            return 2

    set_seed(args.seed)
    prompts = random.sample(DEFAULT_PROMPTS, k=min(args.num_prompts, len(DEFAULT_PROMPTS)))
    logger.info("selected %d prompt(s)", len(prompts))
    for i, p in enumerate(prompts):
        logger.info("  [%d] %s", i, p)

    try:
        orig_model, orig_tok = load_hf_model(args.original, device=args.device)
    except Exception as e:
        logger.error("failed to load original model: %s", e)
        return 2

    orig_prompts = [apply_template(orig_tok, p) for p in prompts]
    logger.info("generating with original model ...")
    orig_gens = [generate_hf(orig_model, orig_tok, tp, args.max_new_tokens) for tp in orig_prompts]

    del orig_model
    _release_cuda_memory()

    try:
        fp8_model, fp8_tok = load_hf_model(args.fp8, device=args.device)
    except Exception as e:
        logger.error("failed to load FP8 model: %s", e)
        return 2

    fp8_prompts = [apply_template(fp8_tok, p) for p in prompts]
    logger.info("generating with FP8 model (transformers) ...")
    fp8_gens = [generate_hf(fp8_model, fp8_tok, tp, args.max_new_tokens) for tp in fp8_prompts]

    del fp8_model
    _release_cuda_memory()

    ok = compare("HF-original", orig_gens, "HF-FP8", fp8_gens, args.min_agreement)

    if args.vllm:
        _release_cuda_memory()
        _wait_cuda_memory_stable()
        util = _fit_vllm_util(args.vllm_gpu_memory_utilization)
        # A vLLM engine that reserves less than ~5% of a large unified pool
        # cannot hold even a small model + kv-cache, so bail out with a clear
        # message rather than letting vLLM emit an obscure allocator failure.
        if util < 0.05:
            logger.error(
                "only ~%.1f%% of GPU memory is currently free — vLLM cannot "
                "start with usable headroom. Free memory (kill other CUDA "
                "processes; drop OS page cache) and retry.",
                util * 100,
            )
            return 2
        try:
            vllm_gens = generate_vllm(
                args.fp8, fp8_prompts, args.max_new_tokens,
                gpu_memory_utilization=util,
                max_model_len=args.vllm_max_model_len,
            )
        except ImportError:
            logger.error("vLLM is not installed. `pip install vllm` and retry.")
            return 2
        except Exception as e:
            logger.error("vLLM run failed: %s", e)
            return 2
        ok = compare("HF-FP8", fp8_gens, "vLLM-FP8", vllm_gens, args.min_agreement) and ok

    print()
    print("=== SUMMARY ===")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
