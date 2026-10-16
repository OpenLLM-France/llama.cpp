"""
Compare transformers.json and ollama.json (token counts) and optionally
behavior.json (functional tool-call check), then print a per-test report.

Token-count comparison (two checks per case):

    Tokenizer
        Pass the transformers-rendered prompt through Ollama with raw=true
        and compare prompt_eval_count to len(transformers token_ids).
        Tests just the GGUF tokenizer, isolated from the chat template.

    Chat template
        Pass the conversation through Ollama's /api/chat.
        Tests Ollama's template + GGUF tokenizer together.

Known acceptable divergences (reported as [WARN], not [FAIL]):

    [tokenizer +1 in tool cases]
        SentencePiece single-space-after-special-token quirk in llama.cpp's
        BPE tokenizer. The model sees one extra space-prefix token where the
        HF tokenizer didn't. Harmless (same decoded string). See llama.cpp
        issue tracker for the upstream bug.

    [chat template -N in tool cases]
        Ollama renders each tool via Go's json.Marshal (compact JSON, no
        spaces). The jinja template uses tojson (pretty JSON, with spaces).
        Same data, same field order, just whitespace. The model parses both
        identically; cosmetic.

Behavioural section (only if --behavior path is provided):

    For each case with an `expected_behavior` field, run_behavior.py asked
    Ollama to actually generate a turn and checked whether the assistant's
    response satisfied the expectation (correct tool_call name + args).

Logits section (only if --logits path is provided):

    Per-case next-token top-K log-probability comparison between
    transformers (reference) and Ollama (GGUF). Catches subtle conversion
    or quantization regressions that the binary behavioural test misses.
    Aggregate metrics: top-1 agreement rate, mean KL divergence.

Exit code is 0 iff no [FAIL] anywhere; [WARN]s are non-fatal.

    Heuristic thresholds for the logits section:
      top-1 agreement       < 50%   -> [FAIL]   (very likely conversion bug)
      top-1 agreement       < 80%   -> [WARN]
      aggregate |Δlp_top1|  > 1.0   -> [FAIL]   (model is very differently confident)
      aggregate |Δlp_top1|  > 0.3   -> [WARN]
      aggregate mean|Δlp|_top3  > 3.0  -> [WARN]   (not a FAIL — top-3 is
                                                    sensitive to tail noise; use
                                                    only as a soft signal)

    Notes:
      - Top-1 lp diff is the primary numeric signal. For fp16-vs-fp16 it
        is typically < 0.1. For Q4_K_M, < 0.5 is normal.
      - mean|Δlp|_top3 is reported for completeness but is noisy by nature:
        fp16 softmax precision degrades for low-probability tokens, so a
        single tail outlier can drag the mean up. Use top-1 metrics for
        pass/fail; treat mean_top3 as informational.
      - Top-1 mismatches often differ only in vocab variant (e.g. `▁The`
        vs `The` for the same word) — those are real model behaviour
        differences worth noting but typically caused by SP/llama.cpp
        tokenization quirks, not gross conversion errors.

Usage:
    python compare.py <transformers_json> <ollama_json>
                      [--behavior <behavior_json>]
                      [--logits   <logits_json>]
"""

import argparse
import json
import sys
from pathlib import Path


def load(path):
    return {r["name"]: r for r in json.loads(Path(path).read_text())}


def fmt(status):
    return {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}[status]


def classify(case_has_tools, tf_count, actual_count, kind):
    """Return (status, label, note) for one column."""
    if actual_count is None:
        return "ok", "skipped", None  # neutral; not a failure if intentionally skipped
    diff = actual_count - tf_count
    if diff == 0:
        return "ok", f"{tf_count} vs {actual_count}", None

    if kind == "tokenizer" and diff == 1 and case_has_tools:
        return ("warn",
                f"{tf_count} vs {actual_count}  (+1)",
                "SPM single-space-after-special-token quirk (llama.cpp tokenizer bug; harmless)")
    if kind == "chat" and diff < 0 and case_has_tools:
        return ("warn",
                f"{tf_count} vs {actual_count}  ({diff:+d})",
                "Ollama renders tools as compact JSON; jinja uses pretty JSON (whitespace only; cosmetic)")
    return "fail", f"{tf_count} vs {actual_count}  ({diff:+d})", None


def report_counts(tf_map, ol_map):
    """Returns (has_any_fail, has_any_warn, collected_notes_by_name)."""
    names = sorted(set(tf_map) | set(ol_map))
    notes_by_name = {}  # name -> list of strings
    any_fail = False
    any_warn = False

    print(f"\n{'name':<40}  {'tokenizer':<32}  {'chat template':<32}")
    print("-" * 110)

    for name in names:
        tf_r = tf_map.get(name)
        ol_r = ol_map.get(name)

        if tf_r is None or "error" in tf_r:
            err = tf_r.get("error", "missing") if tf_r else "missing"
            print(f"{name:<40}  transformers ERROR: {err}")
            any_fail = True
            continue
        if ol_r is None:
            print(f"{name:<40}  ollama ERROR: missing")
            any_fail = True
            continue

        tf_count = tf_r["token_count"]
        has_tools = tf_r.get("tools") is not None

        # Tokenizer probe
        raw_err = ol_r.get("raw_error")
        if raw_err:
            tok_status, tok_label = "fail", f"err: {raw_err[:24]}"
            tok_note = None
        else:
            tok_status, tok_label, tok_note = classify(
                has_tools, tf_count, ol_r.get("raw_prompt_eval_count"), "tokenizer")

        # Chat template probe
        chat_err = ol_r.get("chat_error")
        if chat_err:
            chat_status, chat_label = "fail", f"err: {chat_err[:24]}"
            chat_note = None
        else:
            chat_status, chat_label, chat_note = classify(
                has_tools, tf_count, ol_r.get("chat_prompt_eval_count"), "chat")

        tok_cell = f"{fmt(tok_status)} {tok_label}"
        chat_cell = f"{fmt(chat_status)} {chat_label}"
        print(f"{name:<40}  {tok_cell:<32}  {chat_cell:<32}")

        notes = []
        if tok_note: notes.append(f"tokenizer: {tok_note}")
        if chat_note: notes.append(f"chat: {chat_note}")
        if notes:
            notes_by_name[name] = notes

        any_fail = any_fail or (tok_status == "fail" or chat_status == "fail")
        any_warn = any_warn or (tok_status == "warn" or chat_status == "warn")

    # Print accumulated WARN notes (each unique note once is more readable)
    if notes_by_name:
        print()
        print("WARN notes:")
        printed = set()
        for name, notes in notes_by_name.items():
            for n in notes:
                if n not in printed:
                    print(f"  - {n}")
                    printed.add(n)
        print("  (cases with these warnings: " +
              ", ".join(sorted(notes_by_name)) + ")")

    return any_fail, any_warn


def report_logits(logits_list):
    """Logits section. Returns (has_any_fail, has_any_warn)."""
    print(f"\n{'name':<40}  {'top1':<5}  {'|Δlp_top1|':<11}  {'mean|Δlp|_top3':<15}  "
          f"{'top5':<5}  {'miss':<5}  {'tf top-1':<22}  {'ol top-1':<22}")
    print("-" * 140)

    top1_total = 0
    top1_matches = 0
    top1_diffs = []
    top3_means = []
    skipped = []
    any_error = False

    for r in logits_list:
        name = r["name"]
        if "skipped" in r:
            skipped.append((name, r["skipped"]))
            continue
        if "error" in r:
            print(f"{name:<40}  ERROR: {r['error']}")
            any_error = True
            continue
        cmp = r.get("comparison")
        if not cmp:
            print(f"{name:<40}  no comparison (empty logprobs?)")
            any_error = True
            continue

        top1_total += 1
        if cmp["top1_match"]:
            top1_matches += 1
        if cmp.get("top1_lp_diff") is not None:
            top1_diffs.append(cmp["top1_lp_diff"])
        if cmp.get("mean_lp_diff_top3") is not None:
            top3_means.append(cmp["mean_lp_diff_top3"])

        f = lambda v: (f"{v:.4f}" if v is not None else "n/a")
        miss_s = f"{cmp['tf_top5_missing_in_ollama_topk']}/5"
        tf1 = cmp["tf_top1"]; ol1 = cmp["ol_top1"]
        tf_lbl = f"{tf1['tok']!r}@{tf1['lp']}"
        ol_lbl = f"{ol1['tok']!r}@{ol1['lp']}"
        marker = "✓" if cmp["top1_match"] else "✗"
        print(f"{name:<40}  {marker:<5}  {f(cmp.get('top1_lp_diff')):<11}  "
              f"{f(cmp.get('mean_lp_diff_top3')):<15}  {cmp['top5_overlap']}/5    "
              f"{miss_s:<5}  {tf_lbl[:21]:<22}  {ol_lbl[:21]:<22}")

    if skipped:
        print()
        print("  Skipped (no add_generation_prompt — no canonical next token):")
        for n, why in skipped:
            print(f"    - {n}  ({why})")

    print()
    if top1_total == 0:
        print("  (no logit comparisons completed)")
        return any_error, False

    top1_rate = top1_matches / top1_total
    agg_top1 = (sum(top1_diffs) / len(top1_diffs)) if top1_diffs else None
    agg_top3 = (sum(top3_means) / len(top3_means)) if top3_means else None

    print(f"  aggregate over {top1_total} comparable case(s):")
    print(f"    top-1 agreement      = {top1_matches}/{top1_total} = {top1_rate*100:.1f}%")
    if agg_top1 is not None:
        print(f"    aggregate |Δlp_top1|      = {agg_top1:.4f}")
    if agg_top3 is not None:
        print(f"    aggregate mean|Δlp|_top3  = {agg_top3:.4f}")

    fail = False
    warn = False
    if top1_rate < 0.5:
        print(f"  [FAIL] top-1 agreement {top1_rate*100:.1f}% < 50% — likely a conversion bug")
        fail = True
    elif top1_rate < 0.8:
        print(f"  [WARN] top-1 agreement {top1_rate*100:.1f}% < 80% — investigate the mismatching cases")
        warn = True
    if agg_top1 is not None:
        if agg_top1 > 1.0:
            print(f"  [FAIL] |Δlp_top1| {agg_top1:.4f} > 1.0 — confidence on top token diverges sharply")
            fail = True
        elif agg_top1 > 0.3:
            print(f"  [WARN] |Δlp_top1| {agg_top1:.4f} > 0.3 — model is less confident on chosen tokens; investigate")
            warn = True
    if agg_top3 is not None and agg_top3 > 3.0:
        # WARN-only: top-3 mean is noisy by nature (fp16 softmax tail).
        print(f"  [WARN] mean|Δlp|_top3 {agg_top3:.4f} > 3.0 — distribution shifted in top-3 (soft signal)")
        warn = True
    return (fail or any_error), warn


def report_behavior(behavior_map):
    """Behavioural section. Returns has_any_fail."""
    print(f"\n{'name':<40}  {'behaviour':<60}")
    print("-" * 110)

    any_fail = False
    for name in sorted(behavior_map):
        r = behavior_map[name]
        ok = r.get("pass") is True
        reason = r.get("fail_reason") or ""
        status = "ok" if ok else "fail"
        print(f"{name:<40}  {fmt(status)} {reason[:55]}")
        if not ok:
            any_fail = True
            # Print actual model output details for debugging
            raw = r.get("raw_output")
            if raw is not None:
                snippet = raw if len(raw) <= 200 else raw[:200] + "...(truncated)"
                print(f"  raw model output: {snippet!r}")
            parsed = r.get("parsed_tool_call")
            if parsed:
                print(f"  parsed tool_call: name={parsed.get('name')!r} args={parsed.get('arguments')!r}")
    return any_fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transformers_json")
    parser.add_argument("ollama_json")
    parser.add_argument("--behavior", default=None,
                        help="Optional behaviour JSON from run_behavior.py")
    parser.add_argument("--logits", default=None,
                        help="Optional logits JSON from run_logits.py")
    args = parser.parse_args()

    tf_map = load(args.transformers_json)
    ol_map = load(args.ollama_json)

    print("=" * 120)
    print("TOKEN COUNT COMPARISON")
    print("=" * 120)
    count_fail, count_warn = report_counts(tf_map, ol_map)

    behavior_fail = False
    if args.behavior and Path(args.behavior).exists():
        beh_map = load(args.behavior)
        if beh_map:
            print()
            print("=" * 120)
            print("BEHAVIOURAL CHECK  (model actually called the right tool)")
            print("=" * 120)
            behavior_fail = report_behavior(beh_map)

    logits_fail = False
    logits_warn = False
    if args.logits and Path(args.logits).exists():
        logits_list = json.loads(Path(args.logits).read_text())
        if logits_list:
            print()
            print("=" * 120)
            print("LOGIT COMPARISON  (next-token top-K distribution; transformers vs Ollama)")
            print("=" * 120)
            logits_fail, logits_warn = report_logits(logits_list)

    print()
    any_fail = count_fail or behavior_fail or logits_fail
    any_warn = count_warn or logits_warn
    if any_fail:
        print(">>> RESULT: FAIL")
        sys.exit(1)
    elif any_warn:
        print(">>> RESULT: PASS (with known acceptable warnings)")
        sys.exit(0)
    else:
        print(">>> RESULT: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
