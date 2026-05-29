"""
Compare per-layer activations from tf_layers.npz and gguf_layers.bin
and produce a divergence plot.

For each pair (transformers tensor, GGUF tensor) referring to the same
position in the graph, we compute:

  max_abs_diff   max |x_tf - x_gg|  in fp32
  mean_abs_diff  mean over all elements
  rel_max        max_abs_diff / max(|x_tf|, |x_gg|, 1e-8)
  cosine         1 - cos(x_tf, x_gg)  (smaller = closer)
  l2_rel         ||x_tf - x_gg|| / ||x_tf||

Mappings used (nemotron architecture):
  GGUF                 transformers
  l_out-i              hidden-(i+1)               per-layer output (post-residual)
  attn_norm-i          attn_norm-i                input_layernorm output
  ffn_inp-i            (attn-residual sum)        not directly hookable in transformers;
                                                  approximated as hidden-i + self_attn-i
  ffn_norm-i           post_norm-i                post_attention_layernorm output
  ffn_out-i            (mlp + residual)           approximated as ffn_inp + mlp output
  result_norm          final_norm                 output of model.model.norm
  result_output        logits                     final LM head

The first three columns of the report use l_out / final_norm / logits, which
are the most reliable (no hook approximation). The fine-grained per-op
analysis uses the rest.

Output:
  <work_dir>/layer_diff_report.txt      text report
  <work_dir>/layer_diff_overview.png    log-scale per-layer divergence plot
  <work_dir>/layer_diff_by_op.png       per-op-type breakdown

Usage:
    python compare_layers.py <work_dir>
"""

import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# GGML type ids → numpy dtype (for the ones we'll encounter on activation tensors).
GGML_TYPE = {
    0:  ("f32",  np.float32),
    1:  ("f16",  np.float16),
    24: ("bf16", None),    # handled specially
    26: ("i32",  np.int32),
    30: ("i64",  np.int64),
}


def parse_gguf_dump(path: Path):
    """Yield (name, np_array_fp32) from the binary dump."""
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    n = 0
    while i < len(data):
        if i + 4 > len(data):
            break
        name_len = struct.unpack_from("<I", data, i)[0]; i += 4
        if name_len == 0 or name_len > 1024:
            break
        name = data[i:i+name_len].decode("utf-8", errors="replace"); i += name_len
        dtype = struct.unpack_from("<I", data, i)[0]; i += 4
        ne = struct.unpack_from("<4q", data, i); i += 32
        nbytes = struct.unpack_from("<Q", data, i)[0]; i += 8
        raw = data[i:i+nbytes]; i += nbytes
        info = GGML_TYPE.get(dtype)
        if info is None:
            # skip unknown
            n += 1
            continue
        _, np_dtype = info
        if np_dtype is not None:
            arr = np.frombuffer(raw, dtype=np_dtype).astype(np.float32)
        else:
            # bf16: upper 16 bits of fp32
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32).copy()
        # reshape per ggml's column-major-ish shape: ne is contiguous-first;
        # for our use cases tensors are 1D (vocab) or [embed, tokens, 1, 1].
        # Strip trailing singletons.
        shape = tuple(int(s) for s in ne if s > 1) or (1,)
        try:
            arr = arr.reshape(shape[::-1])  # ggml stores ne in element-stride order
        except ValueError:
            # fallback: leave as flat
            pass
        yield name, arr
        n += 1


def compute_diff_metrics(x_tf, x_gg):
    """Both inputs flattened to fp32 and same total element count."""
    if x_tf.size != x_gg.size:
        return None
    a = x_tf.astype(np.float32).reshape(-1)
    b = x_gg.astype(np.float32).reshape(-1)
    diff = a - b
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    denom = float(max(np.abs(a).max(), np.abs(b).max(), 1e-8))
    rel_max = max_abs / denom
    # cosine-distance
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    cos = 1.0 - float(a @ b) / (na * nb + 1e-12)
    l2_rel = float(np.linalg.norm(diff)) / (na + 1e-12)
    return dict(max_abs=max_abs, mean_abs=mean_abs, rel_max=rel_max, cosine=cos, l2_rel=l2_rel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir")
    parser.add_argument("--top-tail-only", action="store_true",
                        help="For multi-token tensors, only compare the LAST token position "
                             "(matches what next-token prediction uses).")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    tf_path = work_dir / "tf_layers.npz"
    gg_path = work_dir / "gguf_layers.bin"
    meta = json.loads((work_dir / "meta.json").read_text())

    print(f"Comparing layer activations for case {meta['case']!r}")
    print(f"  HF dir: {meta['hf_model_dir']}")
    print(f"  GGUF:   {meta['gguf_path']}")
    print()

    tf = dict(np.load(tf_path))
    gg = dict(parse_gguf_dump(gg_path))
    print(f"transformers: {len(tf)} arrays")
    print(f"gguf:         {len(gg)} arrays")

    n_layers = max(int(k.split("-")[1]) for k in tf if k.startswith("hidden-")) + 1 - 1
    # hidden_states has N+1 entries (0..N); N = num_layers
    print(f"layers: {n_layers}")

    T = int(tf["tokens"].shape[-1])
    print(f"tokens: {T}")

    # Mapping rules: each entry is (label, op_type, tf_array_or_callable, gg_key)
    # tf entry can be a string (npz key) or a callable taking the tf dict and
    # returning an array — to combine multiple hook outputs.
    def add_resid(i):
        """ffn_inp = hidden[i] + self_attn[i]  (post-attention, pre-MLP, with residual)"""
        return lambda tfd: tfd[f"hidden-{i}"] + tfd[f"self_attn-{i}"]

    mappings = []
    for i in range(n_layers):
        mappings.append((f"L{i:02d} attn_norm", "norm",      f"attn_norm-{i}",  f"attn_norm-{i}"))
        mappings.append((f"L{i:02d} ffn_inp",   "post_attn", add_resid(i),      f"ffn_inp-{i}"))
        mappings.append((f"L{i:02d} ffn_norm",  "norm",      f"post_norm-{i}",  f"ffn_norm-{i}"))
        mappings.append((f"L{i:02d} l_out",     "block_out", f"hidden-{i+1}",   f"l_out-{i}"))
    mappings.append(("final_norm",  "norm",       "final_norm",  "result_norm"))
    mappings.append(("logits",      "head",       "logits",      "result_output"))

    rows = []
    for label, op, tk, gk in mappings:
        # Resolve transformers tensor (string key or callable on the dict)
        if callable(tk):
            try:
                x_tf = tk(tf)
            except KeyError as e:
                rows.append({"label": label, "op": op, "skip": f"missing tf key {e}"})
                continue
        else:
            if tk not in tf:
                rows.append({"label": label, "op": op, "skip": f"missing tf={tk}"})
                continue
            x_tf = tf[tk]
        if gk not in gg:
            rows.append({"label": label, "op": op, "skip": f"missing gg={gk}"})
            continue
        x_gg = gg[gk]
        if args.top_tail_only:
            # Pick last token slice
            x_tf = x_tf.reshape(-1, x_tf.shape[-1])[-1] if x_tf.ndim >= 2 else x_tf
            x_gg = x_gg.reshape(-1, x_gg.shape[-1])[-1] if x_gg.ndim >= 2 else x_gg
        m = compute_diff_metrics(x_tf, x_gg)
        if m is None:
            rows.append({"label": label, "op": op, "skip": f"shape mismatch tf={x_tf.shape} gg={x_gg.shape}"})
            continue
        rows.append({"label": label, "op": op, **m, "shape": tuple(x_tf.shape)})

    # ─── Text report ───
    out_txt = work_dir / "layer_diff_report.txt"
    lines = []
    lines.append(f"Layer-by-layer activation comparison — case {meta['case']!r}")
    lines.append(f"HF dtype: {meta['dtype']} on {meta['device']}")
    lines.append("")
    lines.append(f"{'label':<25}  {'op':<10}  {'shape':<22}  {'max|Δ|':<10}  {'mean|Δ|':<10}  {'rel_max':<9}  {'cos_d':<10}  {'l2_rel':<10}")
    lines.append("-" * 120)
    for r in rows:
        if "skip" in r:
            lines.append(f"{r['label']:<25}  {r['op']:<10}  SKIP: {r['skip']}")
            continue
        shape = str(r["shape"])
        lines.append(f"{r['label']:<25}  {r['op']:<10}  {shape:<22}  "
                     f"{r['max_abs']:<10.4g}  {r['mean_abs']:<10.4g}  "
                     f"{r['rel_max']:<9.3g}  {r['cosine']:<10.4g}  {r['l2_rel']:<10.4g}")
    report = "\n".join(lines)
    out_txt.write_text(report + "\n")
    print()
    print(report)
    print()
    print(f"Wrote {out_txt}")

    # ─── Plots ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots (pip install matplotlib)")
        return

    valid = [r for r in rows if "skip" not in r]
    layer_rows = [r for r in valid if r["label"].startswith("L")]

    # Plot 1: overview, divergence vs layer index, separate lines per op
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    by_op = defaultdict(list)
    for r in layer_rows:
        # label is "L00 attn_norm" / "L00 ffn_norm" / "L00 l_out"
        op_label = r["label"].split()[1]
        layer_idx = int(r["label"][1:3])
        by_op[op_label].append((layer_idx, r))

    for ax, metric in zip(axes, ["max_abs", "l2_rel"]):
        for op_label, lst in by_op.items():
            lst.sort()
            xs = [li for li, _ in lst]
            ys = [r[metric] for _, r in lst]
            ax.plot(xs, ys, marker="o", label=op_label)
        # add final_norm and logits as scatter at x = N
        for r in valid:
            if r["label"] == "final_norm":
                ax.scatter([n_layers], [r[metric]], marker="*", s=200, label="final_norm")
            if r["label"] == "logits":
                ax.scatter([n_layers + 0.3], [r[metric]], marker="X", s=150, label="logits")
        ax.set_xlabel("layer index")
        ax.set_ylabel(metric)
        ax.set_yscale("log")
        ax.set_title(f"divergence per layer  ({metric})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"transformers vs GGUF — case {meta['case']!r}")
    fig.tight_layout()
    p1 = work_dir / "layer_diff_overview.png"
    fig.savefig(p1, dpi=110, bbox_inches="tight")
    print(f"Wrote {p1}")

    # Plot 2: cumulative growth of l_out divergence to highlight where drift accumulates
    fig, ax = plt.subplots(figsize=(10, 5))
    l_out_rows = sorted(by_op.get("l_out", []))
    if l_out_rows:
        xs = [li for li, _ in l_out_rows]
        for metric in ["max_abs", "l2_rel", "cosine"]:
            ys = [r[metric] for _, r in l_out_rows]
            ax.plot(xs, ys, marker="o", label=metric)
        ax.set_xlabel("layer index")
        ax.set_ylabel("metric")
        ax.set_yscale("log")
        ax.set_title(f"l_out divergence over depth — case {meta['case']!r}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    p2 = work_dir / "layer_diff_l_out.png"
    fig.savefig(p2, dpi=110, bbox_inches="tight")
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
