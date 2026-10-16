"""
Main orchestrator: compare the chat template + tokenizer of a transformers
model against an Ollama (GGUF + Modelfile) deployment.

Pipeline:
    1. run_transformers.py  ->  <work_dir>/transformers.json
    2. run_ollama.py        ->  <work_dir>/ollama.json
    3. run_behavior.py      ->  <work_dir>/behavior.json   (only cases with expected_behavior)
    4. run_logits.py        ->  <work_dir>/logits.json     (per-case next-token logit comparison)
    5. compare.py           ->  prints per-test report, exit 1 on failure

Each step is skipped if its output JSON already exists; delete the file (or
the whole <work_dir>) to force recomputation. Or pass --force. Slow optional
steps can be turned off with --no-behavior and --no-logits.

Requirements:
    - transformers + torch    (Python; transformers always; torch only for --logits)
    - requests                (Python)
    - ollama                  (must be running:  ollama serve)

Usage:
    python test_main.py <hf_model_dir> <gguf_dir>
                        [--work-dir DIR]
                        [--ollama-model-name NAME]
                        [--ollama-url URL]
                        [--num-predict N]
                        [--logits-top-k K]
                        [--logits-device cuda|cpu]
                        [--no-behavior]
                        [--no-logits]
                        [--force]

Where:
    <hf_model_dir>  is a HuggingFace transformers model directory
                    (must contain tokenizer files + chat_template).
    <gguf_dir>      is a directory containing both:
                      - a 'Modelfile' file
                      - the .gguf file referenced by the Modelfile (FROM ./...)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(label, cmd, output_file, force):
    if not force and output_file.exists():
        print(f"=== {label}: SKIP (using cached {output_file}) ===\n")
        return
    print(f"=== {label} ===")
    print("$ " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(f"!!! {label} failed (exit {rc})")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hf_model_dir", help="Path to the HuggingFace transformers model directory")
    parser.add_argument("gguf_dir", help="Directory containing the Modelfile and the .gguf file")
    parser.add_argument("--work-dir", default=None,
                        help="Where to store intermediate JSON files "
                             "(default: ./results/<hf_basename>__vs__<gguf_basename>/)")
    parser.add_argument("--ollama-model-name", default="test-chat-template-tmp",
                        help="Temporary ollama model name (created and removed by the script). "
                             "No underscores: ollama rejects them.")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--num-predict", type=int, default=256,
                        help="Max tokens generated per behavioural case (default 256)")
    parser.add_argument("--logits-top-k", type=int, default=20,
                        help="K for next-token top-K logit comparison (default 20)")
    parser.add_argument("--logits-device", default=None, choices=["cuda", "cpu"],
                        help="Device for the transformers forward pass (default: cuda if available, else cpu)")
    parser.add_argument("--no-behavior", action="store_true",
                        help="Skip the behavioural step (slower; requires the model to actually generate)")
    parser.add_argument("--no-logits", action="store_true",
                        help="Skip the logit-comparison step (loads the full transformers model; slow)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute all intermediate outputs, ignoring caches")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    hf_dir = Path(args.hf_model_dir).resolve()
    gguf_dir = Path(args.gguf_dir).resolve()
    modelfile = gguf_dir / "Modelfile"

    if not hf_dir.is_dir():
        sys.exit(f"ERROR: HF model dir not found: {hf_dir}")
    if not modelfile.is_file():
        sys.exit(f"ERROR: Modelfile not found at {modelfile}")

    work_dir = (Path(args.work_dir).resolve()
                if args.work_dir
                else here / "results" / f"{hf_dir.name}__vs__{gguf_dir.name}")
    work_dir.mkdir(parents=True, exist_ok=True)
    transformers_json = work_dir / "transformers.json"
    ollama_json = work_dir / "ollama.json"
    behavior_json = work_dir / "behavior.json"
    logits_json = work_dir / "logits.json"

    print(f"HF model dir   : {hf_dir}")
    print(f"GGUF dir       : {gguf_dir}")
    print(f"Modelfile      : {modelfile}")
    print(f"Work dir       : {work_dir}")
    print(f"Ollama URL     : {args.ollama_url}")
    print()

    # Step 1: transformers
    run_step(
        "Step 1/5 — transformers (render + tokenize)",
        [sys.executable, str(here / "run_transformers.py"), str(hf_dir), str(transformers_json)],
        transformers_json,
        args.force,
    )

    # Step 2: ollama (depends on Step 1's JSON for the raw-tokenizer probe)
    run_step(
        "Step 2/5 — ollama (chat + raw tokenizer probes)",
        [sys.executable, str(here / "run_ollama.py"), str(modelfile), str(ollama_json),
         "--model-name", args.ollama_model_name,
         "--transformers-output", str(transformers_json),
         "--ollama-url", args.ollama_url],
        ollama_json,
        args.force,
    )

    # Step 3: behavioural check (optional). The model actually generates here.
    if args.no_behavior:
        print("=== Step 3/5 — behavioural check: SKIPPED (--no-behavior) ===\n")
    else:
        run_step(
            "Step 3/5 — behavioural check (model generates tool_calls)",
            [sys.executable, str(here / "run_behavior.py"), str(modelfile), str(behavior_json),
             "--transformers-output", str(transformers_json),
             "--model-name", args.ollama_model_name,
             "--ollama-url", args.ollama_url,
             "--num-predict", str(args.num_predict)],
            behavior_json,
            args.force,
        )

    # Step 4: logit comparison (optional, slow — loads the full transformers model).
    if args.no_logits:
        print("=== Step 4/5 — logit comparison: SKIPPED (--no-logits) ===\n")
    else:
        logits_cmd = [sys.executable, str(here / "run_logits.py"),
                      str(hf_dir), str(modelfile), str(logits_json),
                      "--transformers-output", str(transformers_json),
                      "--model-name", args.ollama_model_name,
                      "--ollama-url", args.ollama_url,
                      "--top-k", str(args.logits_top_k)]
        if args.logits_device:
            logits_cmd += ["--device", args.logits_device]
        run_step(
            "Step 4/5 — logit comparison (transformers vs Ollama, next-token top-K)",
            logits_cmd,
            logits_json,
            args.force,
        )

    # Step 5: compare (always runs)
    print("=== Step 5/5 — compare ===")
    compare_cmd = [sys.executable, str(here / "compare.py"), str(transformers_json), str(ollama_json)]
    if not args.no_behavior and behavior_json.exists():
        compare_cmd += ["--behavior", str(behavior_json)]
    if not args.no_logits and logits_json.exists():
        compare_cmd += ["--logits", str(logits_json)]
    rc = subprocess.run(compare_cmd).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
