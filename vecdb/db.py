"""
VectorDB — thin database layer on top of the C++ HNSW index.

Responsibilities:
  - Encode text → float32 vectors via sentence-transformers
  - Write-ahead log (WAL) for crash recovery
  - Persist / restore index + metadata to disk
  - Expose insert / query / delete
"""

from __future__ import annotations

import json
import os
import struct
import time
import threading
from pathlib import Path
from typing import Any

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import hnsw_index


_ENCODER = None
_ENCODER_LOCK = threading.Lock()

def _get_encoder(model_name: str):
    global _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is None:
            from sentence_transformers import SentenceTransformer
            _ENCODER = SentenceTransformer(model_name)
    return _ENCODER


# ---------------------------------------------------------------------------
# WAL record layout (binary, append-only):
#   [op: u8][id: i32][vec_len: i32][vec: float32 * vec_len][meta_len: i32][meta: utf-8]
#
# op = 0 → INSERT
# op = 1 → DELETE
# ---------------------------------------------------------------------------

_OP_INSERT = 0
_OP_DELETE = 1


class VectorDB:
    """
    A lightweight vector database backed by the HNSW C++ index.

    Parameters
    ----------
    data_dir : str | Path
        Directory to store the WAL and serialised index.
    dim : int
        Vector dimensionality.  384 for all-MiniLM-L6-v2 (default model).
    M : int
        HNSW M parameter.
    ef_construction : int
        HNSW ef_construction parameter.
    encoder_model : str
        sentence-transformers model name.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        dim: int = 384,
        M: int = 16,
        ef_construction: int = 200,
        encoder_model: str = "all-MiniLM-L6-v2",
    ):
        self.data_dir      = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dim           = dim
        self.encoder_model = encoder_model

        self._wal_path     = self.data_dir / "vecdb.wal"
        self._index_path   = self.data_dir / "vecdb.index"
        self._meta_path    = self.data_dir / "vecdb.meta.json"

        self._lock         = threading.Lock()
        # metadata store: id → {text, metadata, timestamp}
        self._meta: dict[int, dict] = {}
        # per-id vector cache for query responses
        self._vecs: dict[int, np.ndarray] = {}

        # Try loading from disk; otherwise create fresh
        if Path(str(self._index_path) + ".npz").exists() and self._meta_path.exists():
            self._load_from_disk(M, ef_construction)
        else:
            self._idx = hnsw_index.HNSWIndex(
                dim=dim, M=M, ef_construction=ef_construction
            )

        # Replay any WAL records not already reflected in the saved index
        self._wal_file = open(self._wal_path, "ab+")
        self._replay_wal()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        """Encode *text*, insert into index, return assigned id."""
        vec = self._encode(text)
        with self._lock:
            node_id = self._idx.insert(vec)
            self._meta[node_id] = {
                "text":      text,
                "metadata":  metadata or {},
                "timestamp": time.time(),
            }
            self._vecs[node_id] = vec
            self._wal_append_insert(node_id, vec, metadata or {})
        return node_id

    def insert_vector(self, vec: np.ndarray, metadata: dict[str, Any] | None = None) -> int:
        """Insert a pre-computed float32 numpy vector directly."""
        vec = np.asarray(vec, dtype=np.float32)
        with self._lock:
            node_id = self._idx.insert(vec)
            self._meta[node_id] = {
                "text":      "",
                "metadata":  metadata or {},
                "timestamp": time.time(),
            }
            self._vecs[node_id] = vec
            self._wal_append_insert(node_id, vec, metadata or {})
        return node_id

    def query(
        self,
        text: str,
        k: int = 5,
        ef: int = -1,
        return_vectors: bool = False,
    ) -> list[dict]:
        """
        Encode *text* and return the k nearest neighbours.

        Each result dict has keys: id, distance, text, metadata, timestamp.
        """
        vec = self._encode(text)
        return self.query_vector(vec, k=k, ef=ef, return_vectors=return_vectors)

    def query_vector(
        self,
        vec: np.ndarray,
        k: int = 5,
        ef: int = -1,
        return_vectors: bool = False,
    ) -> list[dict]:
        vec = np.asarray(vec, dtype=np.float32)
        with self._lock:
            raw = self._idx.search(vec, k=k, ef=ef)
        results = []
        for dist, node_id in raw:
            entry = {
                "id":        node_id,
                "distance":  float(dist),
                **self._meta.get(node_id, {}),
            }
            if return_vectors and node_id in self._vecs:
                entry["vector"] = self._vecs[node_id].tolist()
            results.append(entry)
        return results

    def delete(self, node_id: int) -> bool:
        """Lazily remove a node. Returns False if id not found."""
        with self._lock:
            if node_id not in self._meta:
                return False
            self._idx.remove(node_id)
            del self._meta[node_id]
            self._vecs.pop(node_id, None)
            self._wal_append_delete(node_id)
        return True

    def save(self):
        """Persist index snapshot + metadata to disk (checkpoint)."""
        with self._lock:
            # Save vectors as a numpy .npz archive keyed by node id
            if self._vecs:
                ids  = list(self._vecs.keys())
                mats = np.stack([self._vecs[i] for i in ids], axis=0)
                np.savez(self._index_path, ids=np.array(ids, dtype=np.int32), vecs=mats)
            else:
                # Write an empty sentinel
                np.savez(self._index_path, ids=np.array([], dtype=np.int32),
                         vecs=np.empty((0, self.dim), dtype=np.float32))

            with open(self._meta_path, "w") as f:
                json.dump(
                    {str(k): v for k, v in self._meta.items()},
                    f,
                    indent=2,
                )
            # Truncate WAL — everything is now in the snapshot
            self._wal_file.seek(0)
            self._wal_file.truncate()

    def __len__(self) -> int:
        return len(self._meta)

    def close(self):
        self.save()
        self._wal_file.close()

    # ------------------------------------------------------------------
    # WAL helpers
    # ------------------------------------------------------------------

    def _wal_append_insert(self, node_id: int, vec: np.ndarray, meta: dict):
        vec_bytes  = vec.tobytes()
        meta_bytes = json.dumps(meta).encode()
        record = (
            struct.pack("<B", _OP_INSERT) +
            struct.pack("<i", node_id) +
            struct.pack("<i", len(vec_bytes)) +
            vec_bytes +
            struct.pack("<i", len(meta_bytes)) +
            meta_bytes
        )
        self._wal_file.write(record)
        self._wal_file.flush()

    def _wal_append_delete(self, node_id: int):
        record = (
            struct.pack("<B", _OP_DELETE) +
            struct.pack("<i", node_id)
        )
        self._wal_file.write(record)
        self._wal_file.flush()

    def _replay_wal(self):
        """Read the WAL and apply any records not yet in the snapshot."""
        wal_path = self._wal_path
        if not wal_path.exists():
            return

        replayed = 0
        with open(wal_path, "rb") as f:
            while True:
                op_byte = f.read(1)
                if not op_byte:
                    break
                op = struct.unpack("<B", op_byte)[0]

                if op == _OP_INSERT:
                    node_id  = struct.unpack("<i", f.read(4))[0]
                    vec_len  = struct.unpack("<i", f.read(4))[0]
                    vec      = np.frombuffer(f.read(vec_len), dtype=np.float32).copy()
                    meta_len = struct.unpack("<i", f.read(4))[0]
                    meta     = json.loads(f.read(meta_len).decode())

                    if node_id not in self._meta:
                        # Re-insert — use the same node_id ordering
                        new_id = self._idx.insert(vec)
                        self._meta[new_id] = {
                            "text":      meta.get("text", ""),
                            "metadata":  meta,
                            "timestamp": time.time(),
                        }
                        self._vecs[new_id] = vec
                        replayed += 1

                elif op == _OP_DELETE:
                    node_id = struct.unpack("<i", f.read(4))[0]
                    if node_id in self._meta:
                        self._idx.remove(node_id)
                        del self._meta[node_id]
                        self._vecs.pop(node_id, None)
                        replayed += 1
                else:
                    break  # corrupt record — stop

        if replayed:
            print(f"[WAL] Replayed {replayed} record(s) from {wal_path}")

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def _load_from_disk(self, M: int, ef_construction: int):
        try:
            data = np.load(str(self._index_path) + ".npz")
            ids  = data["ids"].tolist()
            vecs = data["vecs"]

            with open(self._meta_path, "r") as f:
                raw = json.load(f)
            self._meta = {int(k): v for k, v in raw.items()}

            self._idx = hnsw_index.HNSWIndex(dim=self.dim, M=M, ef_construction=ef_construction)
            for i, node_id in enumerate(ids):
                self._idx.insert(vecs[i])
                self._vecs[node_id] = vecs[i]

            print(f"[VectorDB] Loaded snapshot: {len(ids)} entries")
        except Exception as e:
            print(f"[VectorDB] Could not load snapshot ({e}), starting fresh")
            self._idx  = hnsw_index.HNSWIndex(dim=self.dim, M=M, ef_construction=ef_construction)
            self._meta = {}

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> np.ndarray:
        enc = _get_encoder(self.encoder_model)
        vec = enc.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)
