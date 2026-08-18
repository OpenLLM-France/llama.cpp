"""
Behavioural test: for each test case that has an `expected_behavior` field,
actually run the model and verify the assistant response satisfies the
expectation (e.g. emits a tool_call with the right name + args).

IMPORTANT design note — why /api/generate raw=true:

    Ollama's /api/chat tool-call parser (at least in 0.24) silently drops
    model output when it detects a tool_call tag but fails to extract a
    valid call mid-stream. The model can emit a perfectly well-formed
    <tool_call>...</tool_call> block and the chat API still returns
    {"content": "", "tool_calls": null}. That hides whether the *model*
    works.

    To test the model behind Ollama, we bypass the chat layer entirely:
      1. Take the transformers-rendered prompt (already computed in
         transformers.json — same exact string the model would see at
         training time).
      2. Feed it via /api/generate with raw=true (no template, no parsing).
      3. Read back the raw text the model emitted, parse <tool_call> blocks
         ourselves with a regex, and check the expectation.

    This tests the GGUF model + tokenizer end-to-end without Ollama's chat
    quirks getting in the way.

Currently supports one kind of expectation:

    expected_behavior = {
        "tool_call": {
            "name":              "get_weather",
            "required_args":     ["location"],
            "args_must_contain": {"location": "paris"},  # case-insensitive substring
        }
    }

Output JSON: one entry per case with expected_behavior, including
{name, expected, raw_output, parsed_tool_call, pass, fail_reason}.

Usage:
    python run_behavior.py <modelfile_path> <output_json>
                           --transformers-output <transformers.json>
                           [--model-name NAME]
                           [--ollama-url URL]
                           [--num-predict N]
"""

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("ERROR: this script needs the 'requests' package (pip install requests).")

sys.path.insert(0, str(Path(__file__).parent))
from test_cases import TEST_CASES  # noqa: E402
from run_ollama import check_ollama_alive, ollama_create, ollama_delete, _post  # noqa: E402


# Matches "<tool_call> ... </tool_call>" with any whitespace inside.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def ollama_generate_raw(url, model, prompt, num_predict):
    payload = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0,
            "seed": 0,
            # Stop at the chat-message terminator so we don't waste tokens
            # generating into the next turn.
            "stop": ["<|im_end|>"],
        },
    }
    return _post(f"{url}/api/generate", payload)


def parse_tool_call_from_text(text):
    """Find the first <tool_call>...</tool_call> block and parse its JSON.

    Returns (call_dict_or_None, error_str_or_None) where call_dict is
    {"name": str, "arguments": dict} on success.
    """
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None, "no <tool_call>...</tool_call> block in output"
    body = m.group(1)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        return None, f"<tool_call> body is not valid JSON: {e}; body={body!r}"
    if "name" not in obj:
        return None, f"<tool_call> body has no 'name' field; body={obj!r}"
    args = obj.get("arguments", {})
    if isinstance(args, str):
        # Some templates emit arguments as a JSON-encoded string.
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return {"name": obj["name"], "arguments": args}, None


def evaluate_tool_call(expected, parsed):
    """Check the parsed tool call matches expectation.

    Returns (pass: bool, reason: str | None).
    """
    if parsed["name"] != expected["name"]:
        return False, f"wrong tool name: got {parsed['name']!r}, expected {expected['name']!r}"

    args = parsed["arguments"]
    if not isinstance(args, dict):
        return False, f"arguments not a dict: {args!r}"

    for key in expected.get("required_args", []):
        if key not in args:
            return False, f"missing required arg {key!r} (got args: {list(args)})"

    for key, needle in expected.get("args_must_contain", {}).items():
        val = args.get(key)
        if val is None:
            return False, f"missing arg {key!r}"
        if needle.lower() not in str(val).lower():
            return False, f"arg {key!r}={val!r} does not contain {needle!r}"

    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile_path")
    parser.add_argument("output_json")
    parser.add_argument("--transformers-output", required=True,
                        help="Path to transformers.json (provides the exact "
                             "rendered prompt for each case)")
    parser.add_argument("--model-name", default="test-chat-template-tmp",
                        help="Temporary ollama model name (created/deleted by the script).")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--num-predict", type=int, default=256,
                        help="Max tokens the model may generate per case.")
    args = parser.parse_args()

    version = check_ollama_alive(args.ollama_url)
    print(f"[behavior] Ollama reachable (version {version})")

    tf_path = Path(args.transformers_output)
    if not tf_path.exists():
        sys.exit(f"ERROR: transformers output not found at {tf_path}. "
                 "Run run_transformers.py first.")
    transformers_by_name = {r["name"]: r for r in json.loads(tf_path.read_text())}

    cases_with_behavior = [c for c in TEST_CASES if c.get("expected_behavior")]
    if not cases_with_behavior:
        print("[behavior] No test cases have an 'expected_behavior' field; nothing to do.")
        Path(args.output_json).write_text("[]")
        return

    print(f"[behavior] {len(cases_with_behavior)} behavioural case(s) to run")

    ollama_create(args.model_name, args.modelfile_path)
    try:
        results = []
        for case in cases_with_behavior:
            print(f"[behavior]   {case['name']}")
            entry = {
                "name": case["name"],
                "expected_behavior": case["expected_behavior"],
            }
            try:
                tf = transformers_by_name.get(case["name"])
                if tf is None or "rendered_prompt" not in tf:
                    raise RuntimeError(
                        f"no rendered_prompt for {case['name']} in transformers output"
                    )
                resp = ollama_generate_raw(
                    args.ollama_url, args.model_name,
                    tf["rendered_prompt"], num_predict=args.num_predict,
                )
                raw_output = resp.get("response", "")
                entry["raw_output"] = raw_output
                entry["eval_count"] = resp.get("eval_count")

                eb = case["expected_behavior"]
                if "tool_call" in eb:
                    parsed, parse_err = parse_tool_call_from_text(raw_output)
                    entry["parsed_tool_call"] = parsed
                    if parsed is None:
                        entry["pass"] = False
                        entry["fail_reason"] = parse_err
                    else:
                        ok, reason = evaluate_tool_call(eb["tool_call"], parsed)
                        entry["pass"] = ok
                        entry["fail_reason"] = reason
                else:
                    entry["pass"] = False
                    entry["fail_reason"] = f"unknown expected_behavior keys: {list(eb)}"
            except Exception as e:
                traceback.print_exc()
                entry["pass"] = False
                entry["fail_reason"] = f"{type(e).__name__}: {e}"

            marker = "OK  " if entry.get("pass") else "FAIL"
            print(f"               -> {marker}  {entry.get('fail_reason') or ''}")
            results.append(entry)
    finally:
        ollama_delete(args.model_name)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[behavior] Wrote {len(results)} results to {out}")


if __name__ == "__main__":
    main()
