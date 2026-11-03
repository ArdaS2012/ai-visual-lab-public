"""Witness cached-only real inference and assert semantic/numerical invariants."""

import json
import math
import socket
from pathlib import Path
from server import retrieval
from core import CORPUS_VERSION


def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected network connection during cached inference")


socket.socket.connect = forbidden
settings = dict(
    query="When does registration for the Cedar research workshop close?",
    size=80,
    overlap=20,
    candidates=5,
    selected=2,
)
a = retrieval(settings)
b = retrieval(dict(settings, size=40))
assert a["corpus_version"] == b["corpus_version"] == CORPUS_VERSION
assert len(a["query_vector"]) == 384
assert abs(math.sqrt(sum(v * v for v in a["query_vector"])) - 1) < 1e-5
for result in [a, b]:
    assert len(result["candidates"]) == 5
    assert sum(c["selected"] for c in result["candidates"]) == 2
    for row in result["chunks"]:
        assert len(row["vector"]) == 384
        assert len(row["text"].split()) == row["end"] - row["start"]
        assert -1.000001 <= row["cosine"] <= 1.000001
    expected = sorted(result["chunks"], key=lambda r: (-r["cosine"], r["id"]))[:5]
    assert {c["id"] for c in expected} == {c["id"] for c in result["candidates"]}
    assert [c["logit"] for c in result["candidates"]] == sorted(
        [c["logit"] for c in result["candidates"]], reverse=True
    )
    assert [c["initial_rank"] for c in result["candidates"]] == [
        next(i + 1 for i, e in enumerate(expected) if e["id"] == c["id"])
        for c in result["candidates"]
    ]
assert a["settings"]["size"] == 80 and b["settings"]["size"] == 40
out = Path(__file__).parents[1] / "verification/day2-retrieval-invariants.json"
out.write_text(
    json.dumps(
        dict(
            passed=True,
            network="socket connections forbidden",
            models=a["models"],
            corpus_version=CORPUS_VERSION,
            vector_shape=384,
            settings=[a["settings"], b["settings"]],
            seconds=[a["seconds"], b["seconds"]],
            selected=[
                [c["id"] for c in r["candidates"] if c["selected"]] for r in [a, b]
            ],
        ),
        indent=2,
    )
    + "\n"
)
print(out.read_text())
