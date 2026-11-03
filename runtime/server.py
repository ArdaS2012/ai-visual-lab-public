"""Trainer-only, loopback inference service. Next.js provides the same-origin bridge."""

import asyncio
import json
import queue
import threading
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from core import CORPUS, CORPUS_VERSION, chunks, cosine
from models import embeddings, rerank, MANIFEST, CACHE
from agent import run_agent
from evaluation import run_evaluation, run_judge

app = FastAPI()
LOCK = threading.Lock()


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=300)
    size: int = Field(ge=20, le=120)
    overlap: int = Field(ge=0, le=40)
    candidates: int = Field(ge=1, le=8)
    selected: int = Field(ge=1, le=3)


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: str = Field(min_length=1, max_length=300)
    mode: str = Field(pattern="^(live|invalid|forbidden|loop)$")
    model_calls: int = Field(ge=1, le=6)
    tool_calls: int = Field(ge=1, le=8)
    tokens: int = Field(ge=100, le=12000)
    deadline: int = Field(ge=1, le=180)


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evidence: str = Field(pattern="^(full|missing)$")
    format: str = Field(pattern="^(json|text)$")


class JudgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_id: str = Field(pattern="^(seats|deadline|total|unknown)$")
    raw: str = Field(min_length=1, max_length=3000)


@app.get("/health")
def health():
    models = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    assets = {
        name: (
            CACHE
            / ("models--" + name.replace("/", "--"))
            / "snapshots"
            / rev
            / "model.safetensors"
        ).exists()
        for name, rev in models.items()
    }
    return dict(
        ready=bool(assets) and all(assets.values()),
        assets=assets,
        models=models,
        location="Local CPU; cached model files; no remote inference",
    )


def retrieval(settings):
    started = time.monotonic()
    if not settings["query"].strip():
        raise ValueError("Enter a question.")
    if settings["overlap"] >= settings["size"]:
        raise ValueError("Overlap must be smaller than chunk size.")
    rows = chunks(CORPUS["retrieval"], settings["size"], settings["overlap"])
    vectors, counts = embeddings([settings["query"]] + [r["text"] for r in rows])
    q = vectors[0]
    rows = [
        dict(r, vector=v, model_tokens=n, cosine=cosine(q, v))
        for r, v, n in zip(rows, vectors[1:], counts[1:])
    ]
    ordered = sorted(rows, key=lambda r: (-r["cosine"], r["id"]))[
        : settings["candidates"]
    ]
    ordered = [dict(r, initial_rank=i + 1) for i, r in enumerate(ordered)]
    ranked = sorted(
        rerank(settings["query"], ordered), key=lambda r: (-r["logit"], r["id"])
    )
    ranked = [
        dict(r, final_rank=i + 1, selected=i < settings["selected"])
        for i, r in enumerate(ranked)
    ]
    return dict(
        settings=settings,
        corpus_version=CORPUS_VERSION,
        models=json.loads(MANIFEST.read_text()),
        query_vector=q,
        query_tokens=counts[0],
        chunks=rows,
        candidates=ranked,
        seconds=round(time.monotonic() - started, 3),
    )


async def stream(request, job):
    if not LOCK.acquire(False):
        raise HTTPException(
            429, "Another model experiment is running. Wait or cancel it."
        )
    cancel = threading.Event()
    out = queue.Queue()

    def work():
        try:
            job(cancel, out.put)
        except Exception as exc:
            out.put(dict(kind="error", data=dict(reason=str(exc))))
        finally:
            out.put(None)
            LOCK.release()

    threading.Thread(target=work, daemon=True).start()

    async def body():
        try:
            while True:
                if await request.is_disconnected():
                    cancel.set()
                    break
                try:
                    item = out.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.04)
                    continue
                if item is None:
                    break
                yield json.dumps(item, allow_nan=False) + "\n"
        finally:
            cancel.set()

    return StreamingResponse(
        body(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"}
    )


@app.post("/retrieval")
async def retrieval_route(payload: RetrievalRequest, request: Request):
    def job(cancel, send):
        send(
            dict(
                kind="loading",
                data=dict(reason="Computing real embeddings and pair scores…"),
            )
        )
        result = retrieval(payload.model_dump())
        if not cancel.is_set():
            send(dict(kind="retrieval", data=result))

    return await stream(request, job)


@app.post("/agent")
async def agent_route(payload: AgentRequest, request: Request):
    if not payload.task.strip():
        raise HTTPException(422, "Enter a research question.")
    return await stream(
        request, lambda cancel, send: run_agent(payload.model_dump(), cancel, send)
    )


@app.post("/evaluation")
async def evaluation_route(payload: EvaluationRequest, request: Request):
    return await stream(request, lambda cancel, send: run_evaluation(payload.model_dump(), cancel, send))


@app.post("/evaluation-judge")
async def judge_route(payload: JudgeRequest, request: Request):
    return await stream(request, lambda cancel, send: run_judge(payload.model_dump(), cancel, send))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8010)
