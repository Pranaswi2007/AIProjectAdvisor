"""
RAG Pipeline — Chunking, Embedding, and Vector Indexing
========================================================
Step 1 : chunk_document  — split a Document into overlapping Chunks
Step 2 : embed_texts      — convert text to 384-dim vectors (sentence-transformers)
Step 3 : index_document   — store chunks in ChromaDB (one collection per project)
Step 4 : search           — retrieve the k most relevant chunks for a query
Helpers: list_sources, delete_document, get_project_stats
"""

import re
from functools import lru_cache
from typing import Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import Chunk, Document, SearchResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_PATH  = "./chroma_db"          # folder where the vector DB is persisted
EMBED_MODEL  = "all-MiniLM-L6-v2"    # ~90 MB, runs on CPU, 384-dim embeddings
CHUNK_SIZE   = 800                    # max characters per chunk — fewer chunks = faster indexing
CHUNK_OVERLAP = 50                    # characters shared between neighbouring chunks

_model  = None   # sentence-transformer model (lazy-loaded)
_client = None   # ChromaDB persistent client (lazy-loaded)


# ---------------------------------------------------------------------------
# Step 1: Chunking
# ---------------------------------------------------------------------------

def chunk_document(doc: Document) -> list[Chunk]:
    """Split a Document into overlapping Chunks.

    The splitter tries paragraph breaks → line breaks → sentence ends → words,
    so each chunk stays as semantically coherent as possible.
    Overlap ensures a sentence split at a boundary still appears whole in one
    of the two neighbouring chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(doc.text)
    return [
        Chunk(
            id=f"{doc.id}-{i}",
            document_id=doc.id,
            project_id=doc.project_id,
            source=doc.filename,
            index=i,
            text=piece,
        )
        for i, piece in enumerate(pieces)
    ]


# ---------------------------------------------------------------------------
# Step 2: Embeddings
# ---------------------------------------------------------------------------

def get_model():
    """Load the embedding model once and cache it globally."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of strings into 384-dimensional embedding vectors.

    Encodes in one batched call so the model processes multiple chunks in
    parallel via optimised matrix operations — much faster than one-by-one.
    """
    vectors = get_model().encode(
        texts,
        normalize_embeddings=True,   # unit-length vectors → cosine sim = dot product
        batch_size=128,              # process 128 chunks at a time on CPU
        show_progress_bar=False,
    )
    return vectors.tolist()


@lru_cache(maxsize=256)
def _embed_query_cached(query: str) -> list:
    """Embed a single query string with an LRU cache.

    Repeated or identical queries (e.g. re-submitting the same search) skip
    the model entirely and return the cached vector instantly.
    """
    vectors = get_model().encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors[0].tolist()


# ---------------------------------------------------------------------------
# Step 3: Vector Store
# ---------------------------------------------------------------------------

def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection(project_id: str):
    """Return (or create) the ChromaDB collection for a project.

    Collection names must only contain letters, digits, underscores, hyphens,
    be 3-63 chars, and start/end with an alphanumeric character.
    """
    sanitised = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)
    sanitised = sanitised.strip("_-")          # ChromaDB forbids leading/trailing _ or -
    if not sanitised:
        sanitised = "default"
    name = "project-" + sanitised             # prefix guarantees alphanumeric start
    # Truncate to 63 chars (ChromaDB limit)
    name = name[:63]
    return _get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def index_document(doc: Document) -> int:
    """Chunk, embed and store one Document.

    If the same file was uploaded before, its old chunks are removed first so a
    re-upload does not leave stale data behind.

    Returns the number of chunks stored.
    """
    chunks = chunk_document(doc)
    if not chunks:
        return 0

    collection = get_collection(doc.project_id)

    # Remove any previous version of this file
    collection.delete(where={"source": doc.filename})

    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embed_texts([c.text for c in chunks]),
        metadatas=[
            {
                "source":       c.source,
                "chunk_index":  c.index,
                "document_id":  c.document_id,
            }
            for c in chunks
        ],
    )
    return len(chunks)


def delete_document(project_id: str, filename: str) -> int:
    """Remove all chunks belonging to a specific file from the vector store.

    Returns the number of chunks deleted.
    """
    collection = get_collection(project_id)
    existing = collection.get(where={"source": filename}, include=["metadatas"])
    count = len(existing["metadatas"])
    if count:
        collection.delete(where={"source": filename})
    return count


# ---------------------------------------------------------------------------
# Step 4: Retrieval
# ---------------------------------------------------------------------------

def search(
    project_id: str,
    query: str,
    k: int = 10,
    min_score: float = 0.0,
    max_per_source: int = 0,
) -> list[SearchResult]:
    """Return the k chunks most semantically similar to the query.

    Results are spread across files so no single file monopolises all slots,
    but the cap is computed dynamically so you always get k results when
    enough chunks exist.

    Args:
        project_id:     which project's knowledge base to search.
        query:          natural-language question or keyword phrase.
        k:              number of results to return to the caller.
        min_score:      discard results below this cosine similarity (0–1).
        max_per_source: max chunks from any single file. 0 = auto (computed
                        from number of sources so every file gets a fair share
                        but total always reaches k).
    """
    collection = get_collection(project_id)
    total = collection.count()
    if total == 0:
        return []

    # --- work out how many sources exist so we can set a fair per-source cap
    all_metas = collection.get(include=["metadatas"])["metadatas"]
    num_sources = len({m["source"] for m in all_metas}) or 1

    if max_per_source == 0:
        # Give each source an equal share, but never less than ceil(k/num_sources).
        # This guarantees we can fill k slots even if some files have fewer chunks.
        import math
        max_per_source = math.ceil(k / num_sources)

    # Fetch more candidates than k so the per-source cap doesn't leave us short
    fetch_n = min(max(k * 4, 20), total)

    result = collection.query(
        query_embeddings=[_embed_query_cached(query)],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[SearchResult] = []
    source_counts: dict[str, int] = {}

    for text, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        score = round(1.0 - dist, 4)
        if score < min_score:
            continue

        source = meta["source"]
        if source_counts.get(source, 0) >= max_per_source:
            continue

        source_counts[source] = source_counts.get(source, 0) + 1
        hits.append(
            SearchResult(
                text=text,
                source=source,
                chunk_index=meta["chunk_index"],
                score=score,
                document_id=meta.get("document_id", ""),
            )
        )
        if len(hits) >= k:
            break

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_sources(project_id: str) -> dict[str, int]:
    """Return {filename: chunk_count} for every indexed file in this project."""
    collection = get_collection(project_id)
    metas = collection.get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for m in metas:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return counts


def get_project_stats(project_id: str) -> dict:
    """Summary statistics for the project's knowledge base."""
    sources = list_sources(project_id)
    collection = get_collection(project_id)
    return {
        "project_id":     project_id,
        "document_count": len(sources),
        "chunk_count":    collection.count(),
        "sources":        sources,
    }


def get_all_text(project_id: str) -> str:
    """Return the FULL text of all indexed documents for a project.

    Chunks are reassembled in document order (by source, then chunk index)
    and delimited by filename headers. This bypasses vector similarity
    search entirely so NO section or table row is missed.

    Used for the full-context routing path in combined_agent when the
    total project text is small enough to fit in the model's context window.
    """
    collection = get_collection(project_id)
    if collection.count() == 0:
        return ""

    results = collection.get(include=["documents", "metadatas"])
    if not results["documents"]:
        return ""

    # Group chunks by source and sort by chunk index within each source
    by_source: dict[str, list[tuple[int, str]]] = {}
    for doc_text, meta in zip(results["documents"], results["metadatas"]):
        source = meta.get("source", "unknown")
        idx = meta.get("chunk_index", 0)
        by_source.setdefault(source, []).append((idx, doc_text))

    # Reassemble: sort chunks within each source, join, delimit by filename
    parts: list[str] = []
    for source in sorted(by_source.keys()):
        chunks = sorted(by_source[source], key=lambda x: x[0])
        full_text = "\n".join(text for _, text in chunks)
        parts.append(f"--- FILE: {source} ---\n{full_text}")

    return "\n\n".join(parts)
