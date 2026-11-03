"""Pinned, cached-only CPU inference; first-use downloads are explicit in preload.py."""

import json
from functools import lru_cache
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    StoppingCriteria,
)

CACHE = Path(__file__).parent / ".cache"
MANIFEST = Path(__file__).parent / "models.json"
torch.set_num_threads(4)


@lru_cache(maxsize=3)
def load(kind):
    name = {
        "embed": "sentence-transformers/all-MiniLM-L6-v2",
        "rerank": "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "agent": "Qwen/Qwen2.5-1.5B-Instruct",
    }[kind]
    if not MANIFEST.exists():
        raise RuntimeError("Run the model preload command first; no silent downloads.")
    rev = json.loads(MANIFEST.read_text())[name]
    options = dict(revision=rev, cache_dir=CACHE, local_files_only=True)
    snapshot = CACHE / ("models--" + name.replace("/", "--")) / "snapshots" / rev
    if not (snapshot / "model.safetensors").exists():
        raise RuntimeError(
            f"{name}: model weights are not cached yet. Finish runtime/preload.py before running this lesson."
        )
    tokenizer = AutoTokenizer.from_pretrained(name, **options)
    factory = {
        "embed": AutoModel,
        "rerank": AutoModelForSequenceClassification,
        "agent": AutoModelForCausalLM,
    }[kind]
    model = factory.from_pretrained(name, **options, dtype=torch.float32).eval()
    return tokenizer, model


def embeddings(texts):
    tok, model = load("embed")
    counts = [len(tok(t)["input_ids"]) for t in texts]
    batch = tok(
        texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        vectors = torch.nn.functional.normalize(
            (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9), p=2, dim=1
        )
    return vectors.tolist(), counts


def rerank(query, rows):
    tok, model = load("rerank")
    pairs = [(query, r["text"]) for r in rows]
    counts = [len(tok(q, t)["input_ids"]) for q, t in pairs]
    tokens = tok(
        pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    with torch.inference_mode():
        scores = model(**tokens).logits.reshape(-1).tolist()
    return [
        dict(r, logit=float(s), pair_tokens=n) for r, s, n in zip(rows, scores, counts)
    ]


class StopInference(StoppingCriteria):
    def __init__(self, check, charge=None):
        self.check = check
        self.charge = charge

    def __call__(self, input_ids, scores, **kwargs):
        if self.charge:
            self.charge(input_ids)
        self.check()
        return False
