"""
Render each test case with the transformers chat template + tokenize.

Outputs a JSON file with, for each test case:
    name, messages, tools, add_generation_prompt,
    rendered_prompt, token_count, token_ids

Usage:
    python run_transformers.py <hf_model_dir> <output_json>
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from test_cases import TEST_CASES  # noqa: E402


def render_case(tokenizer, case):
    messages = case["messages"]
    tools = case.get("tools")

    # If the conversation ends on an assistant turn, we are NOT prompting for
    # another generation; otherwise we are (mirrors Ollama's behaviour).
    add_generation_prompt = messages[-1]["role"] != "assistant"

    kwargs = {"add_generation_prompt": add_generation_prompt}
    if tools is not None:
        kwargs["tools"] = tools

    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)

    # apply_chat_template may return a tensor; normalize to list[int]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    return {
        "name": case["name"],
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": add_generation_prompt,
        "rendered_prompt": rendered,
        "token_count": len(token_ids),
        "token_ids": token_ids,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_model_dir", help="Path to HuggingFace transformers model directory")
    parser.add_argument("output_json", help="Path to write the JSON results")
    args = parser.parse_args()

    print(f"[transformers] Loading tokenizer from {args.hf_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)

    results = []
    for case in TEST_CASES:
        print(f"[transformers]   {case['name']}")
        try:
            results.append(render_case(tokenizer, case))
        except Exception as e:
            traceback.print_exc()
            results.append({"name": case["name"], "error": f"{type(e).__name__}: {e}"})

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[transformers] Wrote {len(results)} results to {out}")


if __name__ == "__main__":
    main()
