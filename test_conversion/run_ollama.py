"""
For each test case, query Ollama and collect the input-token count.

Two probes per case:
    chat_prompt_eval_count : tokens fed to the model when the conversation
                             is passed through Ollama's /api/chat (i.e.
                             Ollama applies its Modelfile TEMPLATE, then
                             tokenizes with the GGUF tokenizer).
    raw_prompt_eval_count  : tokens fed when the *transformers-rendered*
                             prompt is passed through /api/generate with
                             raw=true (i.e. only the GGUF tokenizer runs;
                             no chat template applied). This isolates the
                             tokenizer from the template.

The Ollama model is created from the Modelfile at startup and deleted at exit.

Usage:
    python run_ollama.py <modelfile_path> <output_json>
                         [--model-name NAME]
                         [--transformers-output JSON]
                         [--ollama-url URL]
"""

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("ERROR: this script needs the 'requests' package (pip install requests).")

sys.path.insert(0, str(Path(__file__).parent))
from test_cases import TEST_CASES  # noqa: E402


def check_ollama_alive(url):
    try:
        r = requests.get(f"{url}/api/version", timeout=3)
        r.raise_for_status()
        return r.json().get("version", "?")
    except Exception as e:
        sys.exit(
            f"ERROR: cannot reach Ollama at {url} ({e}).\n"
            "       Start it first with:  ollama serve"
        )


def validate_from_target(modelfile_path):
    """Check that the Modelfile's FROM target exists (relative to the Modelfile)."""
    modelfile_path = Path(modelfile_path).resolve()
    for line in modelfile_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("FROM "):
            target = line[len("FROM "):].strip().strip('"')
            # Ollama requires a relative path for local FROM files; absolute
            # paths trigger a misleading "no Modelfile or safetensors files
            # found" error.
            if target.startswith("/"):
                sys.exit(
                    f"ERROR: Modelfile {modelfile_path} has an absolute FROM path "
                    f"({target}). Ollama needs it relative to the Modelfile."
                )
            resolved = (modelfile_path.parent / target).resolve()
            if not resolved.exists():
                sys.exit(
                    f"ERROR: Modelfile {modelfile_path} references\n"
                    f"  FROM {target}\n"
                    f"but {resolved} does not exist.\n"
                    f"Either rename the GGUF or edit the FROM line in the Modelfile.\n"
                    f"(NB: ollama reports this as 'invalid model name' — misleading.)"
                )
            return
    sys.exit(f"ERROR: no FROM line found in {modelfile_path}")


def ollama_create(name, modelfile_path):
    """Create (or overwrite) an ollama model from the given Modelfile."""
    modelfile_path = Path(modelfile_path).resolve()
    validate_from_target(modelfile_path)
    print(f"[ollama] Creating model '{name}' from {modelfile_path}")
    # Run from the modelfile's directory so its FROM ./X.gguf resolves.
    result = subprocess.run(
        ["ollama", "create", name, "-f", modelfile_path.name],
        cwd=str(modelfile_path.parent),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(
            "ERROR: 'ollama create' failed:\n"
            f"  STDOUT: {result.stdout}\n"
            f"  STDERR: {result.stderr}"
        )


def ollama_delete(name):
    print(f"[ollama] Removing temporary model '{name}'")
    subprocess.run(["ollama", "rm", name], capture_output=True, text=True)


def normalize_for_ollama(messages):
    """Convert OpenAI-style messages to the variant Ollama's /api/chat accepts.

    Differences observed empirically:
      - tool_calls[].function.arguments must be an OBJECT, not a JSON string.
      - The OpenAI-style {"type": "function", "function": {...}} wrapper around
        each tool_call is tolerated, but we strip it to be safe.
    """
    out = []
    for msg in messages:
        m = dict(msg)
        tcs = m.get("tool_calls")
        if tcs:
            new_tcs = []
            for tc in tcs:
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass  # leave it; ollama will complain again
                new_tcs.append({"function": {"name": fn["name"], "arguments": args}})
            m["tool_calls"] = new_tcs
        out.append(m)
    return out


def _post(url, payload):
    r = requests.post(url, json=payload, timeout=300)
    if r.status_code >= 400:
        # Surface Ollama's actual complaint instead of a bare HTTPError.
        raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text}")
    return r.json()


def ollama_chat(url, model, messages, tools):
    payload = {
        "model": model,
        "messages": normalize_for_ollama(messages),
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0, "seed": 0},
    }
    if tools is not None:
        payload["tools"] = tools
    return _post(f"{url}/api/chat", payload)


def ollama_generate_raw(url, model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0, "seed": 0},
    }
    return _post(f"{url}/api/generate", payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile_path", help="Path to the Modelfile")
    parser.add_argument("output_json", help="Path to write the JSON results")
    parser.add_argument("--model-name", default="test-chat-template-tmp",
                        help="Temporary ollama model name (will be created and removed). "
                             "Must match ollama's naming rules: lowercase letters, digits, "
                             "hyphens and periods only (no underscores).")
    parser.add_argument("--transformers-output", default=None,
                        help="Path to the transformers.json (enables the raw tokenizer probe)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    version = check_ollama_alive(args.ollama_url)
    print(f"[ollama] Server reachable (version {version})")

    transformers_by_name = {}
    if args.transformers_output and Path(args.transformers_output).exists():
        ref = json.loads(Path(args.transformers_output).read_text())
        transformers_by_name = {r["name"]: r for r in ref}
        print(f"[ollama] Loaded {len(transformers_by_name)} transformers references "
              f"(will probe tokenizer with raw=true)")
    else:
        print("[ollama] No transformers reference available; skipping raw-tokenizer probe")

    ollama_create(args.model_name, args.modelfile_path)

    results = []
    try:
        for case in TEST_CASES:
            print(f"[ollama]   {case['name']}")
            entry = {"name": case["name"]}
            try:
                chat_resp = ollama_chat(
                    args.ollama_url, args.model_name,
                    case["messages"], case.get("tools"),
                )
                entry["chat_prompt_eval_count"] = chat_resp.get("prompt_eval_count")
            except Exception as e:
                traceback.print_exc()
                entry["chat_error"] = f"{type(e).__name__}: {e}"

            ref = transformers_by_name.get(case["name"])
            if ref and "rendered_prompt" in ref:
                try:
                    raw_resp = ollama_generate_raw(
                        args.ollama_url, args.model_name, ref["rendered_prompt"],
                    )
                    entry["raw_prompt_eval_count"] = raw_resp.get("prompt_eval_count")
                except Exception as e:
                    traceback.print_exc()
                    entry["raw_error"] = f"{type(e).__name__}: {e}"

            results.append(entry)
    finally:
        ollama_delete(args.model_name)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[ollama] Wrote {len(results)} results to {out}")


if __name__ == "__main__":
    main()
