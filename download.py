#!/usr/bin/env python3
"""Fetch Ling-3.0-flash MXFP4 target + DSpark drafter into ~/models."""
import os, sys, pathlib
from huggingface_hub import snapshot_download

DEST = pathlib.Path.home() / "models"
REPOS = [
    ("inclusionAI/Ling-3.0-flash-fp4",    pathlib.Path(os.environ.get("MODEL_DIR", DEST / "Ling-3.0-flash-fp4"))),
    ("inclusionAI/Ling-3.0-flash-dspark", pathlib.Path(os.environ.get("DRAFT_DIR", DEST / "Ling-3.0-flash-dspark"))),
]
for repo, local in REPOS:
    print(f"==> {repo} -> {local}", flush=True)
    p = snapshot_download(
        repo_id=repo,
        local_dir=str(local),
        max_workers=8,
        token=os.environ.get("HF_TOKEN") or True,
    )
    print(f"    done: {p}", flush=True)
print("ALL DOWNLOADS COMPLETE", flush=True)
