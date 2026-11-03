"""Explicit online download. Normal runtime uses only these pinned cached files."""

from pathlib import Path
import json
from huggingface_hub import HfApi, snapshot_download

CACHE = Path(__file__).parent / ".cache"
models = [
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    (
        "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "b5c6e9da73abc3711f593f705371cdbe9e0fe422",
    ),
    ("Qwen/Qwen2.5-1.5B-Instruct", None),
]
manifest = {}
for name, rev in models:
    if rev is None:
        lock = Path(__file__).parent / "models.json"
        rev = json.loads(lock.read_text()).get(name) if lock.exists() else None
        rev = rev or HfApi().model_info(name).sha
    manifest[name] = rev
    print(name, rev, flush=True)
    snapshot_download(
        name,
        revision=rev,
        cache_dir=CACHE,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "tokenizer*",
            "vocab*",
            "merges*",
            "*.txt",
        ],
    )
(Path(__file__).parent / "models.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
