#!/usr/bin/env python3
"""
Run every applicable conversion test for a converted model.

Usage:
    python test.py <input_folder>
                   [--gguf <folder>] [--fp8 <folder>]
                   [--skip-tokenizer] [--skip-gguf] [--skip-fp8]
                   [--gguf-file <path>] [--vllm] [--verbose]

Steps (each is skipped either on the corresponding --skip flag, or when its
inputs aren't available):
    1. Tokenizer round-trip test    tests/test-tokenizer-random.py
    2. GGUF conversion test         test_conversion/test_main.py
    3. FP8 conversion test          test_conversion_fp8/test_fp8.py

Defaults:
    --gguf   defaults to <input_folder>-GGUF (falls back to <input_folder>-gguf).
    --fp8    defaults to <input_folder>-FP8 if that folder exists; otherwise the
             FP8 test is silently skipped (no failure).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRCDIR = Path(__file__).resolve().parent


def pick_gguf_file(gguf_dir: Path) -> Path | None:
    if not gguf_dir.is_dir():
        return None
    all_gguf = [
        p for p in gguf_dir.glob("*.gguf")
        if "imatrix" not in p.name.lower()
    ]
    # Prefer a full model file, but the tokenizer test only needs the vocab,
    # so fall back to a vocab-only GGUF when that is all that is available.
    non_vocab = sorted(p for p in all_gguf if "vocab" not in p.name.lower())
    if non_vocab:
        return non_vocab[0]
    vocab = sorted(all_gguf)
    return vocab[0] if vocab else None


def run(label: str, cmd: list[str]) -> bool:
    print()
    print("=" * 78)
    print(f"[{label}]")
    print("$ " + " ".join(str(x) for x in cmd))
    print("=" * 78)
    rc = subprocess.run(cmd, cwd=str(SRCDIR)).returncode
    ok = rc == 0
    print(f"[{label}] -> {'PASS' if ok else f'FAIL (exit {rc})'}")
    return ok


def resolve_gguf_dir(input_folder: Path, override: Path | None) -> Path | None:
    if override is not None:
        return override
    for suffix in ("-GGUF", "-gguf"):
        candidate = input_folder.with_name(input_folder.name + suffix)
        if candidate.is_dir():
            return candidate
    return None


def resolve_fp8_dir(input_folder: Path, override: Path | None) -> Path | None:
    if override is not None:
        return override
    candidate = input_folder.with_name(input_folder.name + "-FP8")
    return candidate if candidate.is_dir() else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all applicable conversion tests for a Luciole-style model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_folder", type=Path,
                   help="original HF-format model directory (before conversion)")
    p.add_argument("--gguf", type=Path, default=None,
                   help="GGUF conversion folder (default: <input_folder>-GGUF)")
    p.add_argument("--fp8", type=Path, default=None,
                   help="FP8 conversion folder (default: <input_folder>-FP8 if present)")
    p.add_argument("--gguf-file", type=Path, default=None,
                   help="specific .gguf file to feed the tokenizer test "
                        "(default: first non-imatrix/non-vocab .gguf in the GGUF dir)")
    p.add_argument("--skip-tokenizer", action="store_true")
    p.add_argument("--skip-gguf", action="store_true")
    p.add_argument("--skip-fp8", action="store_true")
    p.add_argument("--skip-vllm", action="store_true",
                   help="skip the vLLM cross-check inside the FP8 test (on by default)")
    p.add_argument("--fp8-device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="device for the FP8 test's HF inference (pass 'cpu' to work around "
                        "Triton/mamba-ssm crashes on Blackwell)")
    p.add_argument("--disable-verbose", action="store_true",
                   help="do not pass --verbose to individual tests (verbose on by default)")
    p.add_argument("--ollama-url", default="http://localhost:11434",
                   help="Ollama endpoint used by the GGUF conversion test (default: %(default)s)")
    return p.parse_args()


def ollama_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        import urllib.error
        import urllib.request
        with urllib.request.urlopen(f"{url}/api/version", timeout=timeout) as r:
            return r.status == 200
    except (OSError, urllib.error.URLError):
        return False


def ensure_ollama(url: str, wait_seconds: int = 30) -> bool:
    """Return True if Ollama is reachable at `url`, starting it in the
    background if it isn't and the `ollama` binary is available on PATH."""
    if ollama_reachable(url):
        return True
    if not shutil.which("ollama"):
        return False
    print(f"[ollama] not reachable at {url}; starting `ollama serve` in the background...")
    log_path = Path("/tmp/ollama-autostart.log")
    log_file = open(log_path, "wb")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(wait_seconds):
        time.sleep(1)
        if ollama_reachable(url):
            print(f"[ollama] up (logs: {log_path})")
            return True
    print(f"[ollama] failed to come up within {wait_seconds}s (see {log_path})")
    return False


def main() -> int:
    args = parse_args()

    # .absolute() (not .resolve()) so the symlink itself is preserved: the
    # user's naming convention (Luciole-1B-Instruct-1.1) is what -GGUF/-FP8
    # siblings sit next to, not the underlying dpo_luciole_...-step_N target.
    input_folder = args.input_folder.absolute()
    if not input_folder.is_dir():
        print(f"error: input folder not found: {input_folder}", file=sys.stderr)
        return 2

    gguf_dir = resolve_gguf_dir(input_folder, args.gguf)
    fp8_dir = resolve_fp8_dir(input_folder, args.fp8)

    print(f"input:     {input_folder}")
    print(f"gguf dir:  {gguf_dir or '(not found)'}")
    print(f"fp8 dir:   {fp8_dir or '(not found — FP8 test will be skipped)'}")

    results: dict[str, bool] = {}

    if not args.skip_tokenizer:
        if gguf_dir is None:
            print("[tokenizer] SKIP (no GGUF folder)")
        else:
            gguf_file = args.gguf_file or pick_gguf_file(gguf_dir)
            if gguf_file is None:
                print(f"[tokenizer] SKIP (no .gguf file in {gguf_dir})")
            else:
                cmd = [sys.executable, "tests/test-tokenizer-random.py",
                       str(gguf_file), str(input_folder)]
                if not args.disable_verbose:
                    cmd.append("--verbose")
                results["tokenizer"] = run("tokenizer", cmd)

    if not args.skip_gguf:
        if gguf_dir is None:
            print("[gguf-conversion] SKIP (no GGUF folder)")
        elif not (gguf_dir / "Modelfile").is_file():
            print(f"[gguf-conversion] SKIP (no Modelfile in {gguf_dir})")
        elif not ensure_ollama(args.ollama_url):
            print(f"[gguf-conversion] SKIP (Ollama unreachable at {args.ollama_url};")
            print( "                  `ollama` binary not on PATH — install it or pass --ollama-url)")
        else:
            cmd = [sys.executable, "test_conversion/test_main.py",
                   str(input_folder), str(gguf_dir),
                   "--ollama-url", args.ollama_url]
            results["gguf-conversion"] = run("gguf-conversion", cmd)

    if not args.skip_fp8:
        if fp8_dir is None:
            pass
        else:
            cmd = [sys.executable, "test_conversion_fp8/test_fp8.py",
                   str(input_folder), str(fp8_dir),
                   "--device", args.fp8_device]
            if not args.skip_vllm:
                cmd.append("--vllm")
            if not args.disable_verbose:
                cmd.append("--verbose")
            results["fp8-conversion"] = run("fp8-conversion", cmd)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if not results:
        print("(no tests ran)")
        return 2
    for name, ok in results.items():
        print(f"  {name:20s} {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
