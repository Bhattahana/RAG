"""
Embedding & Vector Store Pipeline
===================================
Reads  : chunks/chunks.json         (output of chunking.py)
Stores : vector_store/              (ChromaDB persistent database)

Features
--------
- Embeds all chunks using all-MiniLM-L6-v2 (same model as chunker, loaded once)
- Stores text + embeddings + full metadata in ChromaDB
- Separate collections for text chunks and table_nl chunks
- Batch processing with progress bar (handles large docs without OOM)
- Idempotent — re-running won't duplicate; it upserts by chunk_id
- Includes a quick sanity-query at the end to verify the store works
- Saves embed_audit.json with per-chunk embedding stats

Usage
-----
    cd C:\\Users\\Ahana\\Desktop\\ONGC_RAG
    python embed_store.py

Query the store later
---------------------
    from embed_store import query_store
    results = query_store("rated power of motor", n_results=5)
"""

import json
import sys
import time
from pathlib import Path

# ── Dependency checks ─────────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    sys.exit("ChromaDB not found.\nRun: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("sentence-transformers not found.\nRun: pip install sentence-transformers")

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False          # graceful fallback — plain print

# =====================================
# PATHS & CONFIG
# =====================================
CHUNKS_FILE      = r"chunks\chunks.json"
VECTOR_STORE_DIR = r"vector_store"
EMBED_AUDIT_FILE = r"chunks\embed_audit.json"

EMBED_MODEL      = "all-MiniLM-L6-v2"
BATCH_SIZE       = 64           # chunks per embedding batch (safe for 8 GB RAM)

COLLECTION_TEXT  = "tender_text"
COLLECTION_TABLE = "tender_tables"

# =====================================
# LOAD CHUNKS
# =====================================
chunks_path = Path(CHUNKS_FILE)
if not chunks_path.exists():
    sys.exit(f"ERROR: {CHUNKS_FILE} not found. Run chunking.py first.")

with open(chunks_path, encoding="utf-8") as f:
    all_chunks = json.load(f)

text_chunks  = [c for c in all_chunks if c["chunk_type"] == "text"]
table_chunks = [c for c in all_chunks if c["chunk_type"] == "table_nl"]

print(f"Chunks loaded      : {len(all_chunks)}")
print(f"  ├─ text          : {len(text_chunks)}")
print(f"  └─ table_nl      : {len(table_chunks)}")

# =====================================
# LOAD EMBEDDING MODEL (once)
# =====================================
print(f"\nLoading embedding model: {EMBED_MODEL} …")
model = SentenceTransformer(EMBED_MODEL)
print("Model ready.\n")

# =====================================
# INIT CHROMADB
# =====================================
Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(
    path=VECTOR_STORE_DIR,
    settings=Settings(anonymized_telemetry=False),
)

# Get-or-create both collections
col_text  = client.get_or_create_collection(
    name=COLLECTION_TEXT,
    metadata={"hnsw:space": "cosine"},   # cosine similarity
)
col_table = client.get_or_create_collection(
    name=COLLECTION_TABLE,
    metadata={"hnsw:space": "cosine"},
)

print(f"ChromaDB store     : {Path(VECTOR_STORE_DIR).resolve()}")
print(f"Collections        : '{COLLECTION_TEXT}', '{COLLECTION_TABLE}'\n")


# =====================================
# HELPERS
# =====================================

def build_metadata(chunk: dict) -> dict:
    """
    ChromaDB metadata values must be str | int | float | bool.
    Convert lists (page_range) to string.
    """
    return {
        "page":            chunk.get("page", 0),
        "page_range":      str(chunk.get("page_range", [chunk.get("page", 0)])),
        "page_type":       chunk.get("page_type", ""),
        "section":         chunk.get("section", ""),
        "subsection":      chunk.get("subsection", ""),
        "chunk_type":      chunk.get("chunk_type", ""),
        "char_count":      chunk.get("char_count", 0),
        "token_count":     chunk.get("token_count", 0),
        "tess_confidence": float(chunk.get("tess_confidence", 0)),
        "easy_confidence": float(chunk.get("easy_confidence", 0)),
        "source_file":     chunk.get("source_file", ""),
        "vendor":          chunk.get("vendor", ""),   
    }


def embed_and_upsert(chunks: list[dict], collection, label: str) -> list[dict]:
    """
    Embed chunks in batches and upsert into ChromaDB collection.
    Returns audit records.
    """
    if not chunks:
        print(f"  No {label} chunks to embed.")
        return []

    audit = []
    total = len(chunks)
    batches = range(0, total, BATCH_SIZE)

    iterator = tqdm(batches, desc=f"Embedding {label}", unit="batch") if _TQDM \
               else batches

    for batch_start in iterator:
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        t0 = time.perf_counter()
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine sim = dot product after norm
        )
        elapsed = time.perf_counter() - t0

        ids        = [str(c["chunk_id"]) for c in batch]
        metadatas  = [build_metadata(c) for c in batch]
        embed_list = [e.tolist() for e in embeddings]

        collection.upsert(
            ids=ids,
            embeddings=embed_list,
            documents=texts,
            metadatas=metadatas,
        )

        for chunk, emb in zip(batch, embeddings):
            audit.append({
                "chunk_id":    chunk["chunk_id"],
                "page":        chunk["page"],
                "section":     chunk.get("section", ""),
                "chunk_type":  chunk["chunk_type"],
                "token_count": chunk.get("token_count", 0),
                "embed_dim":   len(emb),
                "embed_norm":  float((emb ** 2).sum() ** 0.5),
            })

        if not _TQDM:
            done = min(batch_start + BATCH_SIZE, total)
            print(f"  {label}: {done}/{total} chunks  ({elapsed:.2f}s for batch)")

    return audit


# =====================================
# EMBED & STORE
# =====================================
print("── Embedding text chunks ────────────────────────────────────────────")
audit_text  = embed_and_upsert(text_chunks,  col_text,  "text")

print("\n── Embedding table→NL chunks ────────────────────────────────────────")
audit_table = embed_and_upsert(table_chunks, col_table, "table_nl")

# =====================================
# SAVE AUDIT
# =====================================
all_audit = audit_text + audit_table
Path(EMBED_AUDIT_FILE).parent.mkdir(parents=True, exist_ok=True)
with open(EMBED_AUDIT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_audit, f, indent=4, ensure_ascii=False)

# =====================================
# STATS
# =====================================
print(f"\n{'='*55}")
print(f"Vectors stored (text)    : {col_text.count()}")
print(f"Vectors stored (tables)  : {col_table.count()}")
print(f"Embedding dimensions     : {all_audit[0]['embed_dim'] if all_audit else 'N/A'}")
print(f"Vector store location    : {Path(VECTOR_STORE_DIR).resolve()}")
print(f"Embed audit saved        : {Path(EMBED_AUDIT_FILE).resolve()}")
print(f"{'='*55}\n")

# =====================================
# SANITY QUERY
# =====================================
print("── Sanity query: 'rated power of motor' ─────────────────────────────")
q_emb = model.encode(["rated power of motor"], normalize_embeddings=True)[0].tolist()

for col, name in [(col_text, "text"), (col_table, "table_nl")]:
    if col.count() == 0:
        continue
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(2, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    print(f"\nTop results from '{name}' collection:")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1 - dist, 4)          # cosine similarity (1 = identical)
        print(f"  [page {meta['page']} | {meta['section']} | score {score}]")
        print(f"  {doc[:120]}…" if len(doc) > 120 else f"  {doc}")

print("\nEmbedding pipeline complete. Vector store is ready for RAG queries.")


# =====================================
# REUSABLE QUERY FUNCTION
# =====================================
def query_store(
    query: str,
    n_results: int = 5,
    collection: str = "both",          # "text" | "table_nl" | "both"
    filters: dict = None,              # e.g. {"page_type": "TABLE"}
) -> list[dict]:
    """
    Query the vector store from any other script:

        from embed_store import query_store
        hits = query_store("VFD input voltage", n_results=5)
        for h in hits:
            print(h["score"], h["page"], h["text"][:200])
    """
    q_vec = model.encode([query], normalize_embeddings=True)[0].tolist()
    cols  = []
    if collection in ("text",  "both"): cols.append(col_text)
    if collection in ("table_nl", "both"): cols.append(col_table)

    hits = []
    for col in cols:
        if col.count() == 0:
            continue
        kwargs = dict(
            query_embeddings=[q_vec],
            n_results=min(n_results, col.count()),
            include=["documents", "metadatas", "distances"],
        )
        if filters:
            kwargs["where"] = filters
        res = col.query(**kwargs)
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            hits.append({
                "score":      round(1 - dist, 4),
                "text":       doc,
                "page":       meta["page"],
                "section":    meta["section"],
                "chunk_type": meta["chunk_type"],
                "token_count":meta["token_count"],
            })

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:n_results]
