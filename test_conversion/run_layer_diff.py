"""
Layer-by-layer activation dump for both backends, on a single anchor prompt.

Outputs two files in <work_dir>:

  tf_layers.npz      — numpy archive with one array per intermediate tensor:
                       "tokens"                        : input token ids (1, T)
                       "hidden-i"  for i in 0..N        : per-layer output of the i-th block
                                                          (transformers .hidden_states)
                       "attn_norm-i"  for i in 0..N-1   : output of input_layernorm
                       "self_attn-i"  for i in 0..N-1   : output of the attention block (without residual)
                       "post_norm-i"  for i in 0..N-1   : output of post_attention_layernorm
                       "mlp-i"        for i in 0..N-1   : output of MLP (without residual)
                       "final_norm"                    : after model.model.norm
                       "logits"                        : final LM head output

  gguf_layers.bin    — binary dump from llama-eval-callback (env-gated).
                       Records of: u32 name_len, name, u32 dtype, i64 ne[4], u64 nbytes, data.

The companion compare_layers.py loads both and computes per-layer divergence.

Usage:
    python run_layer_diff.py <hf_model_dir> <gguf_path>
                             --transformers-output <transformers.json>
                             --case <case_name>
                             --work-dir <dir>
                             [--device cuda|cpu]
                             [--dtype fp16|fp32|bf16]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LLAMA_BIN_DIR = Path("/home/jlouradour/src.nowsl/llama.cpp/build/bin")
EVAL_CALLBACK_BIN = LLAMA_BIN_DIR / "llama-eval-callback"

# Tensor name regex passed to the patched llama.cpp dumper.
# Keep aligned with the names cb()'d by src/models/nemotron.cpp.
GGUF_DUMP_REGEX = r'^(attn_norm|ffn_inp|ffn_norm|ffn_out|l_out|result_norm|result_output)-?[0-9]*$'


def transformers_dump(model_dir: Path, prompt: str, device: str, dtype: torch.dtype, out_path: Path):
    print(f"[tf] Loading model from {model_dir} ({device}, {dtype})")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()

    # Match the way our test renders prompts: tokenize the raw rendered prompt
    # without adding extra special tokens — the prompt already contains them.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs["input_ids"]
    print(f"[tf] Input tokens: {input_ids.shape}, last-pos id = {int(input_ids[0,-1])}")

    captures = {}

    def hook_for(key):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captures[key] = t.detach().cpu().float().numpy()
        return fn

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.input_layernorm.register_forward_hook(hook_for(f"attn_norm-{i}")))
        handles.append(layer.self_attn.register_forward_hook(hook_for(f"self_attn-{i}")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(hook_for(f"post_norm-{i}")))
        handles.append(layer.mlp.register_forward_hook(hook_for(f"mlp-{i}")))
    handles.append(model.model.norm.register_forward_hook(hook_for("final_norm")))

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    for h in handles:
        h.remove()

    # Per-layer hidden states (hidden[i] is the output of layer i; hidden[0] = embeddings)
    for i, h in enumerate(out.hidden_states):
        captures[f"hidden-{i}"] = h.detach().cpu().float().numpy()
    captures["logits"] = out.logits.detach().cpu().float().numpy()
    captures["tokens"] = input_ids.cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **captures)
    print(f"[tf] Saved {len(captures)} arrays to {out_path}")

    # Free memory: model is no longer needed
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def gguf_dump(gguf_path: Path, prompt: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LLAMA_BIN_DIR) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["LLAMA_DUMP_TENSORS_FILE"] = str(out_path)
    env["LLAMA_DUMP_TENSORS_REGEX"] = GGUF_DUMP_REGEX
    # Atomic tokenization of <|im_start|> etc., so the token count matches
    # what HF transformers produces on a fully-rendered chat-template prompt.
    env["LLAMA_TOKENIZE_PARSE_SPECIAL"] = "1"

    cmd = [str(EVAL_CALLBACK_BIN),
           "-m", str(gguf_path),
           "-p", prompt,
           "-n", "1"]
    print(f"[gguf] Running {EVAL_CALLBACK_BIN.name} with regex={GGUF_DUMP_REGEX!r}")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-1000:])
        print(res.stderr[-1000:])
        sys.exit(f"[gguf] llama-eval-callback failed (exit {res.returncode})")
    if not out_path.exists() or out_path.stat().st_size == 0:
        sys.exit(f"[gguf] dump file empty: {out_path}")
    print(f"[gguf] Dump size: {out_path.stat().st_size/1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hf_model_dir")
    parser.add_argument("gguf_path")
    parser.add_argument("--transformers-output", required=True,
                        help="Path to existing transformers.json (provides rendered_prompt per case)")
    parser.add_argument("--case", required=True,
                        help="Test case name from test_cases.py to use as the anchor prompt")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default=None, choices=["fp16", "fp32", "bf16"],
                        help="Defaults to fp16 on cuda, fp32 on cpu")
    args = parser.parse_args()

    hf_dir = Path(args.hf_model_dir).resolve()
    gguf_path = Path(args.gguf_path).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    tf_json = json.loads(Path(args.transformers_output).read_text())
    case = next((c for c in tf_json if c["name"] == args.case), None)
    if case is None:
        sys.exit(f"case {args.case!r} not found in {args.transformers_output}")
    prompt = case["rendered_prompt"]

    if args.dtype is None:
        args.dtype = "fp16" if args.device == "cuda" else "fp32"
    torch_dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    tf_out = work_dir / "tf_layers.npz"
    gg_out = work_dir / "gguf_layers.bin"

    transformers_dump(hf_dir, prompt, args.device, torch_dtype, tf_out)
    gguf_dump(gguf_path, prompt, gg_out)

    # Also save the prompt + tokens so compare_layers can sanity check.
    meta = {
        "case": args.case,
        "prompt": prompt,
        "device": args.device,
        "dtype": args.dtype,
        "hf_model_dir": str(hf_dir),
        "gguf_path": str(gguf_path),
    }
    (work_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved meta to {work_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
