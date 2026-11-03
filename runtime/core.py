"""Pure, testable evidence and policy operations. No generated answers or model mocks."""

import hashlib
import math
import re
from pathlib import Path
import json
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

CORPUS = json.loads(
    (Path(__file__).parents[1] / "src/lib/data/day2-corpus.json").read_text()
)
CORPUS_VERSION = hashlib.sha256(json.dumps(CORPUS, sort_keys=True).encode()).hexdigest()


def strict_json(text):
    """Reject non-JSON constants and numbers that overflow the finite tool domain."""

    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def number(value):
        result = float(value)
        return result if math.isfinite(result) else reject(value)

    return json.loads(text, parse_constant=reject, parse_float=number)


def chunks(documents, size, overlap):
    if type(size) is not int or type(overlap) is not int or not 0 <= overlap < size:
        raise ValueError("Require integer 0 ≤ overlap < chunk size.")
    result = []
    for doc in documents:
        words = doc["text"].split()
        for start in range(0, len(words), size - overlap):
            end = min(start + size, len(words))
            result.append(
                dict(
                    id=f"{doc['id']}:{start}-{end}",
                    source=doc["id"],
                    start=start,
                    end=end,
                    text=" ".join(words[start:end]),
                )
            )
            if end == len(words):
                break
    return result


def cosine(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("Matching nonempty vector shapes required.")
    norm = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    if not norm:
        raise ValueError("Cosine is undefined for a zero vector.")
    return sum(x * y for x, y in zip(a, b)) / norm


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: StrictStr = Field(min_length=1, max_length=200)
    top_k: StrictInt = Field(ge=1, le=3)


class RatioArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    numerator: StrictFloat | StrictInt
    denominator: StrictFloat | StrictInt


def validate(name, args):
    if name == "document_search":
        value = SearchArgs.model_validate(args)
        if not value.query.strip():
            raise ValueError("query must contain text.")
    elif name == "ratio":
        value = RatioArgs.model_validate(args)
        n, d = value.numerator, value.denominator
        if not math.isfinite(n) or not math.isfinite(d) or not 0 <= n <= d or d <= 0:
            raise ValueError(
                "Require finite 0 ≤ numerator ≤ denominator and denominator > 0."
            )
    else:
        raise ValueError(f"{name} is outside the tool allowlist.")
    return value.model_dump()


def search(query, top_k):
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    rows = []
    for doc in CORPUS["studies"]:
        words = re.findall(r"[a-z0-9]+", doc["text"].lower())
        # Number of distinct query terms present; literal lexical score, not embeddings.
        score = len(terms & set(words))
        if score:
            rows.append(
                dict(
                    **doc,
                    score=score,
                    version=CORPUS_VERSION,
                    start=0,
                    end=len(doc["text"].split()),
                )
            )
    return sorted(rows, key=lambda r: (-r["score"], r["id"]))[:top_k]


def ratio(numerator, denominator):
    validate("ratio", dict(numerator=numerator, denominator=denominator))
    return dict(
        proportion=numerator / denominator, percent=100 * numerator / denominator
    )
