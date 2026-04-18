"""
FastAPI REST server for the vector database.

Endpoints:
  POST   /insert         — encode text and insert
  POST   /query          — encode text and search k-NN
  DELETE /delete/{id}    — remove entry by id
  GET    /health         — liveness probe
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vecdb import VectorDB


# ---------------------------------------------------------------------------
# Config via environment variables
# ---------------------------------------------------------------------------

DATA_DIR        = os.getenv("VECDB_DATA_DIR",        "data")
DIM             = int(os.getenv("VECDB_DIM",          "384"))
M               = int(os.getenv("VECDB_M",            "16"))
EF_CONSTRUCTION = int(os.getenv("VECDB_EF_CONST",     "200"))
ENCODER_MODEL   = os.getenv("VECDB_MODEL",            "all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Lifespan — open and cleanly close the DB around the server's lifetime
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = VectorDB(
        data_dir=DATA_DIR,
        dim=DIM,
        M=M,
        ef_construction=EF_CONSTRUCTION,
        encoder_model=ENCODER_MODEL,
    )
    yield
    app.state.db.close()


app = FastAPI(
    title="VecDB",
    description="Lightweight vector database with HNSW indexing",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class InsertRequest(BaseModel):
    text: str = Field(..., description="Text to encode and insert")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class InsertResponse(BaseModel):
    id: int
    message: str = "inserted"


class QueryRequest(BaseModel):
    text: str     = Field(..., description="Query text")
    k: int        = Field(default=5, ge=1, le=1000, description="Number of results")
    ef: int       = Field(default=-1, description="Search beam width (-1 = auto)")


class QueryResult(BaseModel):
    id: int
    distance: float
    text: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    results: list[QueryResult]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "size": len(app.state.db)}


@app.post("/insert", response_model=InsertResponse, status_code=201)
def insert(req: InsertRequest):
    node_id = app.state.db.insert(req.text, req.metadata)
    return InsertResponse(id=node_id)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    hits = app.state.db.query(req.text, k=req.k, ef=req.ef)
    results = [
        QueryResult(
            id=h["id"],
            distance=h["distance"],
            text=h.get("text", ""),
            metadata=h.get("metadata", {}),
        )
        for h in hits
    ]
    return QueryResponse(results=results)


@app.delete("/delete/{node_id}", status_code=200)
def delete(node_id: int):
    ok = app.state.db.delete(node_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    return {"id": node_id, "message": "deleted"}
