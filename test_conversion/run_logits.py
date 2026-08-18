"""
Logit-level comparison: for each test case, compute the next-token
top-K log-probability distribution from
  (a) the transformers model (forward pass on the rendered prompt)
  (b) Ollama serving the GGUF (logprobs API on the same prompt)
and report top-1 agreement, top-5 overlap, and KL divergence.

This catches subtle numerical regressions in the GGUF (quantization,
conversion bugs, wrong activation, etc.) that the binary tool-call
behavioural test would not notice.

Per-case metrics:

  top1_match           bool  — same most-likely next token (most important)
  top1_lp_diff         float — |TF top-1 logprob − Ollama top-1 logprob|.
                               Concrete confidence delta on the chosen token.
                               fp16-vs-fp16: typically < 0.1.
                               Q4_K_M: typically < 0.5.
  top5_overlap         int   — how many of TF's top-5 are in Ollama's top-5 (0..5).
  mean_lp_diff_top3    float — primary aggregate metric: mean |Δlp| over TF's
                               top-3 tokens (aligned by token ID). Top-3 covers
                               the bulk of the probability mass; excluding the
                               4-5 tail tokens avoids the high noise that fp16
                               softmax has on low-probability logits.
  mean_lp_diff_top5    float — same but over top-5; reported for completeness.
                               Naturally noisier; use top-3 for judgement.
  tf_top5_missing      int   — count of TF's top-5 tokens not in Ollama's
                               top-K. High counts mean Ollama wasn't even close
                               on those tokens (significant divergence).
  kl_div_renorm        float — secondary: KL on renormalized common-top-K.
                               Can be inflated; ignore unless other signals
                               also flag.

Cases whose rendered prompt does NOT end at a generation point (i.e.
add_generation_prompt was False — last message was assistant text/tool_calls,
no `<|im_start|>assistant\\n` suffix) are SKIPPED: there is no canonical
"next token" to predict there.

Usage:
    python run_logits.py <hf_model_dir> <modelfile_path> <output_json>
                         --transformers-output <transformers.json>
                         [--model-name NAME]
                         [--ollama-url URL]
                         [--top-k K]
                         [--device cuda|cpu]
                         [--dtype fp16|fp32]
"""

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

try:
    import requests  # noqa: F401  (imported via run_ollama as well, but be explicit)
except ImportError:
    sys.exit("ERROR: this script needs the 'requests' package (pip install requests).")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    sys.exit(f"ERROR: this script needs torch + transformers ({e}).")

sys.path.insert(0, str(Path(__file__).parent))
from run_ollama import check_ollama_alive, ollama_create, ollama_delete, _post  # noqa: E402


def transformers_topk(model, tokenizer, prompt, top_k):
    """Forward-pass the model and return top-K (token_id, token, logprob)."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    vals, idxs = torch.topk(logprobs, top_k)
    return [
        {
            "token_id": int(i),
            "token": tokenizer.convert_ids_to_tokens(int(i)),
            "logprob": float(lp),
        }
        for i, lp in zip(idxs.tolist(), vals.tolist())
    ]


def ollama_topk(url, model_name, prompt, top_k):
    """Get Ollama's top-K next-token logprobs (raw=true so it doesn't apply the chat template)."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0, "seed": 0},
        "logprobs": True,
        "top_logprobs": top_k,
    }
    resp = _post(f"{url}/api/generate", payload)
    lps = resp.get("logprobs") or []
    if not lps:
        return None
    first = lps[0]
    return [
        {"token": t["token"], "logprob": t["logprob"], "bytes": t.get("bytes")}
        for t in first.get("top_logprobs", [])
    ]


_SPM_SPACE = "▁"  # ▁ — SentencePiece's word-boundary marker


def ollama_token_to_id(tokenizer, ol_entry, vocab):
    """Map an Ollama-reported token to the transformers vocab ID.

    Critical: we look the token up DIRECTLY in the vocab dict, not via
    tokenizer.encode(). The encoder normalizes (e.g. always converts a
    leading literal-space `' Bonjour'` to `▁Bonjour`), which would COLLIDE
    distinct GGUF vocab entries (`' Bonjour'`/▁Bonjour id=34362 vs
    `'Bonjour'` id=21327) and cause ol_by_id[id] to be set to the WRONG
    logprob (whichever distinct token appears later in the top-K list).
    """
    s = ol_entry["token"]

    # 1. Try the SentencePiece form: leading literal-space → ▁ prefix.
    spm_form = (_SPM_SPACE + s[1:]) if s.startswith(" ") else s
    if spm_form in vocab:
        return vocab[spm_form]

    # 2. Try the raw string (for non-space-prefixed tokens like 'Bonjour').
    if s in vocab:
        return vocab[s]

    # 3. Last resort: lossy re-tokenize. May collide; logged via caller.
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


def compare_topk(tf_top, ol_top, tokenizer):
    """Compare next-token top-K distributions and return per-case metrics."""
    if not tf_top or not ol_top:
        return None

    # Annotate Ollama entries with transformers vocab IDs (direct vocab lookup).
    vocab = tokenizer.get_vocab()
    ol_with_ids = [
        {**t, "token_id": ollama_token_to_id(tokenizer, t, vocab)} for t in ol_top
    ]

    tf1 = tf_top[0]
    ol1 = ol_with_ids[0]
    top1_match = tf1["token_id"] == ol1["token_id"]

    tf_top5_ids = {t["token_id"] for t in tf_top[:5]}
    ol_top5_ids = {t["token_id"] for t in ol_with_ids[:5] if t["token_id"] is not None}
    top5_overlap = len(tf_top5_ids & ol_top5_ids)

    # PRIMARY: absolute logprob differences on TF's top-N tokens (aligned
    # by token ID via Ollama's top-K). Reported on top-1 (concrete) and
    # top-3 (aggregate). Top-5 also computed for completeness but is
    # naturally noisy because fp16 softmax precision is lowest in the tail.
    ol_by_id = {t["token_id"]: t["logprob"] for t in ol_with_ids if t["token_id"] is not None}

    def diffs_over(n):
        d = []
        miss = 0
        for t in tf_top[:n]:
            ol_lp = ol_by_id.get(t["token_id"])
            if ol_lp is None:
                miss += 1
            else:
                d.append(abs(t["logprob"] - ol_lp))
        return d, miss

    d3, _ = diffs_over(3)
    d5, missing5 = diffs_over(5)
    mean3 = (sum(d3) / len(d3)) if d3 else None
    mean5 = (sum(d5) / len(d5)) if d5 else None

    # Top-1 logprob diff specifically (most interpretable; same token assumed).
    top1_lp_diff = None
    if top1_match:
        ol_top1_lp = ol_by_id.get(tf1["token_id"])
        if ol_top1_lp is not None:
            top1_lp_diff = abs(tf1["logprob"] - ol_top1_lp)

    # SECONDARY METRIC: KL on renormalized common-top-K (kept for reference;
    # can be inflated when overlap is small).
    tf_by_id = {t["token_id"]: t["logprob"] for t in tf_top}
    common = set(tf_by_id) & set(ol_by_id)
    kl = None
    if common:
        tf_p = {i: math.exp(tf_by_id[i]) for i in common}
        ol_p = {i: math.exp(ol_by_id[i]) for i in common}
        s_tf = sum(tf_p.values())
        s_ol = sum(ol_p.values())
        if s_tf > 0 and s_ol > 0:
            kl = 0.0
            for i in common:
                p = tf_p[i] / s_tf
                q = ol_p[i] / s_ol
                if p > 1e-12 and q > 1e-12:
                    kl += p * math.log(p / q)

    return {
        "top1_match": top1_match,
        "tf_top1": {"id": tf1["token_id"], "tok": tf1["token"], "lp": round(tf1["logprob"], 4)},
        "ol_top1": {"id": ol1["token_id"], "tok": ol1["token"], "lp": round(ol1["logprob"], 4)},
        "top1_lp_diff": top1_lp_diff,
        "top5_overlap": top5_overlap,
        "tf_top5_missing_in_ollama_topk": missing5,
        "mean_lp_diff_top3": mean3,
        "mean_lp_diff_top5": mean5,
        "kl_div_renorm": kl,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_model_dir", help="Path to HuggingFace transformers model directory")
    parser.add_argument("modelfile_path", help="Path to the GGUF Modelfile (for ollama create)")
    parser.add_argument("output_json")
    parser.add_argument("--transformers-output", required=True,
                        help="Path to transformers.json (provides the rendered prompts)")
    parser.add_argument("--model-name", default="test-chat-template-tmp")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default=None, choices=["fp16", "fp32", "bf16"],
                        help="Defaults to fp16 on cuda, fp32 on cpu")
    args = parser.parse_args()

    version = check_ollama_alive(args.ollama_url)
    print(f"[logits] Ollama reachable (version {version})")

    tf_path = Path(args.transformers_output)
    if not tf_path.exists():
        sys.exit(f"ERROR: transformers output not found at {tf_path}")
    transformers_data = json.loads(tf_path.read_text())

    if args.dtype is None:
        args.dtype = "fp16" if args.device == "cuda" else "fp32"
    torch_dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    print(f"[logits] Loading transformers model from {args.hf_model_dir} ({args.device}, {args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_dir, torch_dtype=torch_dtype, trust_remote_code=True,
    ).to(args.device)
    model.eval()

    ollama_create(args.model_name, args.modelfile_path)
    try:
        results = []
        for entry in transformers_data:
            name = entry["name"]
            prompt = entry.get("rendered_prompt")
            if not prompt:
                continue
            if entry.get("add_generation_prompt") is False:
                # No canonical next-token prediction: the conversation ends
                # on the assistant's own message (closed by <|im_end|>).
                # Skip — there's nothing meaningful to compare.
                print(f"[logits]   {name}  SKIP (no add_generation_prompt)")
                results.append({"name": name, "skipped": "no add_generation_prompt"})
                continue
            print(f"[logits]   {name}")
            try:
                tf_top = transformers_topk(model, tokenizer, prompt, args.top_k)
                ol_top = ollama_topk(args.ollama_url, args.model_name, prompt, args.top_k)
                cmp = compare_topk(tf_top, ol_top, tokenizer)
                results.append({
                    "name": name,
                    "comparison": cmp,
                    "tf_top5": tf_top[:5],
                    "ol_top5": (ol_top or [])[:5],
                })
                if cmp:
                    fmt = lambda v: (f"{v:.4f}" if v is not None else "n/a")
                    print(f"    top1_match={cmp['top1_match']}  "
                          f"|Δlp_top1|={fmt(cmp['top1_lp_diff'])}  "
                          f"mean|Δlp|_top3={fmt(cmp['mean_lp_diff_top3'])}  "
                          f"top5_overlap={cmp['top5_overlap']}/5  "
                          f"missing={cmp['tf_top5_missing_in_ollama_topk']}/5")
            except Exception as e:
                traceback.print_exc()
                results.append({"name": name, "error": f"{type(e).__name__}: {e}"})
    finally:
        ollama_delete(args.model_name)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[logits] Wrote {len(results)} results to {out}")


if __name__ == "__main__":
    main()
