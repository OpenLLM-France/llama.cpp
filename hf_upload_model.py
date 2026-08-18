#!/usr/bin/env python3
"""
Upload a local folder (or single file) to a HuggingFace repository.

Usage:
    python hf_upload_model.py <path> <repo_id> [--message MSG]
                              [--private | --public]
                              [--create-repo {auto,always,never}]
                              [--repo-type {model,dataset,space}]

Examples:
    python hf_upload_model.py ./Luciole-1B-Instruct-GGUF \\
        OpenLLM-France/Luciole-1B-Instruct-1.0-GGUF \\
        --message "add Q4_K_M + BF16 + assets"

Authentication: uses your HF token from the env (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)
or the cached token from `huggingface-cli login`. Falls back to an interactive
prompt only when neither is present.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import huggingface_hub


def connect(repo_id: str, repo_type: str, create_mode: str, private: bool):
    api = huggingface_hub.HfApi()
    try:
        api.whoami()
    except Exception:
        huggingface_hub.login()
        api = huggingface_hub.HfApi()

    exists = True
    try:
        api.repo_info(repo_id, repo_type=repo_type)
    except huggingface_hub.utils.RepositoryNotFoundError:
        exists = False

    if create_mode == "never" and not exists:
        raise SystemExit(f"repo {repo_id} does not exist and --create-repo=never")
    if create_mode == "always" or (create_mode == "auto" and not exists):
        print(f"creating https://huggingface.co/{repo_id}")
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
    return api


def upload(
    input_path: str,
    repo_id: str,
    message: str | None = None,
    create_repo: str = "auto",
    private: bool = True,
    repo_type: str = "model",
) -> None:
    src = Path(input_path)
    if not src.exists():
        raise SystemExit(f"{input_path} does not exist")

    api = connect(repo_id, repo_type, create_repo, private)
    repo_url = f"https://huggingface.co/{repo_id}"

    if src.is_dir():
        print(f"uploading folder {src} -> {repo_url}")
        api.upload_folder(
            folder_path=str(src),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=message or f"upload {src.name}",
            ignore_patterns=["__pycache__", ".offload", ".git", ".DS_Store"],
        )
    else:
        print(f"uploading file {src} -> {repo_url}")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=src.name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=message or f"upload {src.name}",
        )
    print(f"done: {repo_url}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a folder or file to a HuggingFace repo")
    p.add_argument("input", type=str, help="local folder or file to upload")
    p.add_argument("repo_id", type=str, help="target repo id, e.g. OpenLLM-France/Luciole-1B-Instruct-1.0-GGUF")
    p.add_argument("--message", type=str, default=None, help="commit message")
    p.add_argument("--repo-type", choices=["model", "dataset", "space"], default="model")
    p.add_argument("--create-repo", choices=["auto", "always", "never"], default="auto",
                   help="whether to create the repo if it does not exist (default: auto)")
    visibility = p.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true", default=True,
                            help="create as private (default)")
    visibility.add_argument("--public", dest="private", action="store_false",
                            help="create as public")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    upload(
        input_path=args.input,
        repo_id=args.repo_id,
        message=args.message,
        create_repo=args.create_repo,
        private=args.private,
        repo_type=args.repo_type,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
