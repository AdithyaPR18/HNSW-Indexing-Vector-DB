"""
Test the FastAPI layer using TestClient (in-process, no network).
"""

import sys
import os
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["VECDB_DATA_DIR"] = tempfile.mkdtemp(prefix="vecdb_api_test_")
os.environ["VECDB_DIM"]      = "384"

from fastapi.testclient import TestClient
from server import app


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    print(f"  /health  ✓  (size={data['size']})")


def test_insert_and_query(client):
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Vector databases enable semantic similarity search",
        "Python is a popular programming language",
        "HNSW graphs provide approximate nearest neighbour search",
    ]

    ids = []
    for t in texts:
        r = client.post("/insert", json={"text": t, "metadata": {"source": "test"}})
        assert r.status_code == 201, r.text
        ids.append(r.json()["id"])
    print(f"  Inserted {len(ids)} documents  ✓")

    # Query something topically related to vector search
    r = client.post("/query", json={"text": "approximate nearest neighbor search", "k": 3})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) == 3
    print(f"  Query returned {len(results)} results:")
    for res in results:
        print(f"    [{res['id']}] dist={res['distance']:.4f}  \"{res['text'][:60]}\"")

    top_texts = [res["text"] for res in results]
    assert any("HNSW" in t or "nearest" in t or "Vector" in t for t in top_texts), \
        f"Expected vector-search related result in top-3, got: {top_texts}"
    print("  Semantic relevance check  ✓")

    return ids


def test_delete(client, ids):
    target = ids[0]
    r = client.delete(f"/delete/{target}")
    assert r.status_code == 200, r.text
    print(f"  Deleted id={target}  ✓")

    r = client.delete(f"/delete/{target}")
    assert r.status_code == 404
    print(f"  Double-delete 404  ✓")


if __name__ == "__main__":
    print("Testing FastAPI server...")
    with TestClient(app) as client:
        test_health(client)
        ids = test_insert_and_query(client)
        test_delete(client, ids)

    data_dir = os.environ["VECDB_DATA_DIR"]
    shutil.rmtree(data_dir, ignore_errors=True)
    print("\nAll API tests passed.")
