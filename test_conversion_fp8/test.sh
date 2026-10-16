#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

usage() {
    cat <<EOF
Usage: bash test.sh <original_folder> <fp8_folder> [--vllm]
                    [--max-new-tokens N] [--num-prompts N]
                    [--min-agreement F] [--seed N]

Compare inference between an HF model and its FP8-quantized sibling
(produced by convert_hf_to_fp8.py). See test_fp8.py for details.
EOF
    exit 1
}

[ $# -lt 2 ] && usage

python3 test_fp8.py "$@"
