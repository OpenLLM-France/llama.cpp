# test_conversion

Validate that an HF transformers model has been converted to GGUF
faithfully — both the tokenizer/chat-template AND the model weights — by
comparing the **transformers reference** against the **GGUF served by
Ollama**. Used to catch tokenizer drift, broken chat templates,
quantization regressions, conversion bugs in `convert_hf_to_gguf.py`,
etc., before publishing a release.

The suite has two parts:

1. **`test_main.py`** — main 5-step pipeline (tokenizer, chat
   template, behaviour, logits). Run this for every release.
2. **`run_layer_diff.py` + `compare_layers.py`** — deeper layer-by-layer
   activation comparison. Run when (1) flags a logit-level regression
   and you need to localize which op causes it.

## Prerequisites

### Python

```bash
pip install transformers torch requests numpy matplotlib safetensors
```

### llama.cpp built (with our patches)

The layer-diff tool uses two env-gated patches in `llama.cpp`:

- `common/debug.cpp` — binary tensor dump triggered by
  `LLAMA_DUMP_TENSORS_FILE` and `LLAMA_DUMP_TENSORS_REGEX`. Pure no-op
  when those vars are unset.
- `examples/eval-callback/eval-callback.cpp` — atomic tokenization of
  control tokens (`<|im_start|>` etc.) when `LLAMA_TOKENIZE_PARSE_SPECIAL=1`.
  Default behaviour unchanged.

Build:

```bash
cd /path/to/llama.cpp
cmake -B build
cmake --build build --target llama-eval-callback -j$(nproc)
```

### Ollama server running

```bash
ollama serve     # in another terminal
```

The main pipeline talks to it on `http://localhost:11434`.

---

## Step 0 — Convert HF → GGUF

Starting from a HuggingFace transformers checkpoint directory:

```bash
HF_MODEL=/path/to/Luciole-1B-SFT-1.2          # contains config.json, model.safetensors, tokenizer.*
GGUF_DIR=/path/to/Luciole-1B-SFT-1.2-gguf      # output directory

mkdir -p "$GGUF_DIR"
python /path/to/llama.cpp/convert_hf_to_gguf.py \
    "$HF_MODEL" \
    --outfile "$GGUF_DIR/Luciole-1B-SFT-f16.gguf" \
    --outtype f16
```

For a quantized variant (smaller, faster, with some precision loss):

```bash
/path/to/llama.cpp/build/bin/llama-quantize \
    "$GGUF_DIR/Luciole-1B-SFT-f16.gguf" \
    "$GGUF_DIR/Luciole-1B-SFT-q4_k_m.gguf" \
    Q4_K_M
```

## Step 1 — Write the Ollama `Modelfile`

In `$GGUF_DIR/Modelfile`:

```
FROM ./Luciole-1B-SFT-f16.gguf        # or your quantized variant
PARAMETER seed 1234
PARAMETER num_ctx 32000
PARAMETER temperature 0.6
SYSTEM "You are a helpful AI assistant named Luciole, trained by LINAGORA and OpenLLM France."
TEMPLATE """
…your Go-template version of the jinja chat template, including {{- range .Tools }}{{ . }}{{- end }} for tool support…
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
…
```

Two pitfalls Ollama 0.24 hits silently:

- `FROM` must be **relative** to the Modelfile directory. Absolute paths
  fail with `no Modelfile or safetensors files found`.
- Ollama detects tool-calling capability from the template body. For the
  `nemotron` architecture only the literal `{{ . }}` form inside
  `{{ range .Tools }}` is recognized — `{{ .Function }}` or
  `{{ json . }}` will silently disable tool support (Ollama returns
  `does not support tools` on any tool request).

---

## Step 2 — Run the main pipeline

```bash
cd /path/to/llama.cpp/test_conversion
python test_main.py "$HF_MODEL" "$GGUF_DIR"
```

This runs five steps; each writes a JSON into
`results/<hf_basename>__vs__<gguf_basename>/` and is skipped on rerun if
its JSON already exists. Pass `--force` to recompute, or delete the
JSON manually for a partial rerun. Slow steps can be turned off with
`--no-behavior` and `--no-logits`.

### What each step checks

| step | script | output | what it tests |
|------|--------|--------|---------------|
| 1 | `run_transformers.py` | `transformers.json` | renders each test case with `tokenizer.apply_chat_template(...)` and tokenizes — the reference for everything else. |
| 2 | `run_ollama.py` | `ollama.json` | per case, asks Ollama for `prompt_eval_count` two ways: (a) `/api/chat` (Ollama applies its Modelfile template + GGUF tokenizer); (b) `/api/generate raw=true` fed the transformers-rendered prompt (GGUF tokenizer only). |
| 3 | `run_behavior.py` | `behavior.json` | for cases with `expected_behavior`, sends the prompt to the model via `/api/generate raw=true`, parses the generated text for `<tool_call>{…}</tool_call>`, verifies tool name + required args. Bypasses Ollama's `/api/chat` tool parser, which is unreliable on `nemotron`-arch models in 0.24. |
| 4 | `run_logits.py` | `logits.json` | for the same prompts, runs transformers forward pass and Ollama with `logprobs=true`, compares next-token top-K distributions. Catches quantization / conversion regressions invisible to a binary tool-call test. |
| 5 | `compare.py` | stdout + exit code | unified report. |

### Reading the report

`compare.py` prints three sections.

**Token-count comparison** — per case, two columns:
- `tokenizer` = transformers `apply_chat_template(tokenize=True)` length vs
  Ollama's `prompt_eval_count` on the same rendered prompt via raw mode.
- `chat template` = same but Ollama applies its own template.

Some mismatches are flagged `[WARN]` (with note) instead of `[FAIL]`:
- **tokenizer +1 in tool cases**: known llama.cpp SentencePiece quirk
  — a single space following a special/added token gets segmented as a
  spurious `▁▁` (two-space) piece. See the *Known issues* section
  below.
- **chat template −N in tool cases**: Ollama renders each tool via
  Go's `json.Marshal` (compact JSON, no spaces); jinja's `tojson` uses
  pretty JSON (with spaces). Same data, only whitespace differs.

**Behavioural check** — for each `expected_behavior` case, did the
model emit a valid `<tool_call>` block with the right name + args?

**Logit comparison** — for each case (skipping ones without a generation
prompt), how close are the next-token distributions?

- `top1` ✓/✗: same most-likely next token (matched by vocab id)
- `|Δlp_top1|`: absolute logprob diff on the chosen token
  (fp16-vs-fp16: typically < 0.1; Q4_K_M: < 0.5 normal)
- `mean|Δlp|_top3`: mean of `|Δlp|` over TF's top-3 tokens
- `top5_overlap` / `miss`: how many of TF's top-5 are even in Ollama's top-K

The aggregate thresholds for FAIL/WARN are documented at the top of
`compare.py`.

### Exit code

- `0` — PASS, or PASS with known acceptable warnings.
- `1` — at least one `[FAIL]` somewhere.

---

## Step 3 — Layer-by-layer diagnostic (optional)

When the logit step flags an unexpected regression on a specific case,
this localizes which layer (and which op type within the layer) is
introducing the divergence.

```bash
python run_layer_diff.py "$HF_MODEL" "$GGUF_DIR/Luciole-1B-SFT-f16.gguf" \
    --transformers-output results/<hf>__vs__<gguf>/transformers.json \
    --case 02_system_user \
    --work-dir results/<hf>__vs__<gguf>/layer_diff_02_system_user

python compare_layers.py results/<hf>__vs__<gguf>/layer_diff_02_system_user --top-tail-only
```

Outputs:
- `tf_layers.npz` — transformers per-layer hidden states + intermediate
  hook outputs (input_layernorm, self_attn, post_attention_layernorm,
  mlp, final_norm, logits).
- `gguf_layers.bin` — llama.cpp per-layer activations
  (`attn_norm-i`, `ffn_inp-i`, `ffn_norm-i`, `l_out-i`, `result_norm`,
  `result_output`).
- `layer_diff_report.txt` — per-pair max/mean abs diff, relative max,
  cosine distance, l2 relative error.
- `layer_diff_overview.png` — log-scale divergence vs layer index,
  one series per op type.
- `layer_diff_l_out.png` — focused view of per-layer block output drift.

`--top-tail-only` restricts comparison to the **last token position** —
this is what matters for next-token prediction and avoids confusion at
the last layer where llama.cpp uses `inp_out_ids` to compute only the
last position.

### How to read the layer-diff

If `L00 attn_norm` matches to floating-point precision but `L00 ffn_inp`
diverges, the **attention block** is to blame. If `L00 attn_norm` already
diverges, the **input embedding or tokenization** is to blame (or the
input LN itself). And so on along the column of op types.

---

## Known issues / current findings (Luciole 1B SFT 1.2)

- **Tokenizer +1 in tool cases** — llama.cpp's SentencePiece-style
  tokenizer emits a spurious `▁▁` (double-space) piece when a special
  token is followed by exactly one literal space then text. Affects the
  fixed instruction string `function name and arguments within
  <tool_call></tool_call> XML tags:` in the system prompt of every
  tool-using conversation. Reported upstream; harmless (decoded string
  unchanged), but the model sees one out-of-distribution token per
  request. Flagged `[WARN]`.

- **Chat template `−N` in tool cases** — Ollama renders each tool with
  Go's `json.Marshal` (compact); jinja uses pretty JSON. ~21 tokens
  saved per tool definition. Cosmetic; the model parses both
  identically. Flagged `[WARN]`.

- **Logit drift at layer 0, attention block** — even with f16 GGUF
  matching f16 transformers, the attention output already diverges
  significantly at L00 (cos_d ≈ 0.05 on case 02). Most likely
  PyTorch SDPA vs llama.cpp attention kernel: different reduction
  orders in fp16 give different accumulation. Drifts to ~0.3 cos_d by
  the last layer. Top-1 token usually still matches.

- **`convert_hf_to_gguf.py` precision pitfall** — for the Nemotron
  LayerNorm1p hack, `data_torch + 1` must be done in fp32, otherwise
  the bf16 source values round before storage. Use
  `data_torch.float() + 1`. Other entries in the converter with the
  same `+ 1` pattern (Gemma, Nemotron-H, line ~8887 MTP block, lines
  5731+ for some mamba variant) should be audited similarly.

---

## File layout

```
test_conversion/
├── README.md                  # this file
├── test_main.py               # main orchestrator (steps 1–5)
├── test_cases.py              # canonical test conversations
├── run_transformers.py        # step 1
├── run_ollama.py              # step 2
├── run_behavior.py            # step 3
├── run_logits.py              # step 4
├── compare.py                 # step 5 — unified report
├── run_layer_diff.py          # layer-diff tool
├── compare_layers.py          # layer-diff report + plots
├── test.sh                    # convenience wrapper (if present)
└── results/                   # outputs land here, one subfolder per <hf>__vs__<gguf>
    └── <hf>__vs__<gguf>/
        ├── transformers.json
        ├── ollama.json
        ├── behavior.json
        ├── logits.json
        └── layer_diff_<case>/
            ├── tf_layers.npz
            ├── gguf_layers.bin
            ├── meta.json
            ├── layer_diff_report.txt
            └── *.png
```

## Hard-coded paths to update

`run_layer_diff.py` has the llama.cpp build path baked in at the top:

```python
LLAMA_BIN_DIR = Path("/home/jlouradour/src.nowsl/llama.cpp/build/bin")
```

Change this if your build directory is elsewhere.
