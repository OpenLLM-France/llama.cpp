#!/usr/bin/env bash
# Thin wrapper — real logic lives in test.py.
set -e
cd "$(dirname "$0")"
exec python3 test.py "$@"
