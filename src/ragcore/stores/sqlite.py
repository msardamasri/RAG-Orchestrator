"""The default store: one SQLite file holding the chunks, a word index (FTS5) and a vector index (sqlite-vec).

Both searches run against the same rows, which is what makes hybrid retrieval cheap here: no second service,
nothing to keep in sync.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
import threading
from collections.abc import Sequence
from pathlib import Path

import sqlite_vec

from ragcore.types import Chunk, Document, Filters, Hit

SAFE_KEY = re.compile(r"^[A-Za-z0-9_]+$")
TERM = re.compile(r"\w+(?:-\w+)*", re.UNICODE)


def pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def fts_query(text: str) -> str:
    """Any of the query's words, each quoted so punctuation and FTS operators in user text are harmless.

    A hyphenated code such as ``EXP-204`` stays one quoted term, which FTS5 reads as the phrase ``EXP 204``.
    """
    terms = [t for t in TERM.findall(text) if len(t) > 1]
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in dict.fromkeys(terms))


def filter_sql(filters: Filters | None, alias: str = "c") -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    for key, value in (filters or {}).items():
        if not SAFE_KEY.match(key):
            raise ValueError(f"invalid filter key: {key!r}")
        column = f"{alias}.doc_id" if key == "doc_id" else f"json_extract({alias}.metadata, '$.{key}')"
        if isinstance(value, (list, tuple, set)):
            values = list(value)
            if not values:
                clauses.append("0")
                continue
            clauses.append(f"{column} IN ({','.join('?' * len(values))})")
            params.extend(values)
        else:
            clauses.append(f"{column} = ?")
            params.append(value)
    return (" AND ".join(clauses), params) if clauses else ("", [])


class SQLiteStore:
    supports_lexical = True

    def __init__(self, path: str | Path = "ragcore.db", dim: int = 384, embedder_name: str = ""):
        self.path = str(path)
        self.dim = dim
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._init(embedder_name)

    def _init(self, embedder_name: str) -> None:
        with self.lock, self.db:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS documents(
                    doc_id TEXT PRIMARY KEY, hash TEXT NOT NULL, source TEXT, metadata TEXT);
                CREATE TABLE IF NOT EXISTS chunks(
                    rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, doc_id TEXT NOT NULL,
                    position INTEGER NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL, embedding BLOB NOT NULL);
                CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text, content='chunks', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2');
                """
            )
            stored = {r["key"]: r["value"] for r in self.db.execute("SELECT key, value FROM meta")}
            if "dim" in stored and int(stored["dim"]) != self.dim:
                raise RuntimeError(
                    f"{self.path} holds {stored['dim']}-dimension vectors from {stored.get('embedder') or 'another '
                    'embedder'}, but the configured embedder produces {self.dim}. Use a different store path or "
                    "delete the file and ingest again.")
            if "dim" not in stored:
                self.db.execute("INSERT INTO meta(key, value) VALUES('dim', ?)", (str(self.dim),))
                self.db.execute("INSERT INTO meta(key, value) VALUES('embedder', ?)", (embedder_name,))
            self.db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
                f"embedding float[{self.dim}] distance_metric=cosine)")

    # -- writes -----------------------------------------------------------------------------------
    def document_hash(self, doc_id: str) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT hash FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return row["hash"] if row else None

    def _delete_chunks(self, doc_id: str) -> None:
        rows = self.db.execute("SELECT rowid, text FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
        for row in rows:
            self.db.execute("INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', ?, ?)",
                            (row["rowid"], row["text"]))
            self.db.execute("DELETE FROM chunks_vec WHERE rowid = ?", (row["rowid"],))
        self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def replace_document(self, document: Document, content_hash: str, chunks: Sequence[Chunk],
                         vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        with self.lock, self.db:
            self._delete_chunks(document.id)
            self.db.execute(
                "INSERT INTO documents(doc_id, hash, source, metadata) VALUES(?,?,?,?) "
                "ON CONFLICT(doc_id) DO UPDATE SET hash = excluded.hash, source = excluded.source, "
                "metadata = excluded.metadata",
                (document.id, content_hash, document.source, json.dumps(document.metadata)))
            for chunk, vector in zip(chunks, vectors, strict=True):
                if len(vector) != self.dim:
                    raise ValueError(f"vector has {len(vector)} dimensions, the store expects {self.dim}")
                blob = pack(vector)
                cursor = self.db.execute(
                    "INSERT INTO chunks(id, doc_id, position, text, metadata, embedding) VALUES(?,?,?,?,?,?)",
                    (chunk.id, chunk.doc_id, chunk.position, chunk.text, json.dumps(chunk.metadata), blob))
                self.db.execute("INSERT INTO chunks_fts(rowid, text) VALUES(?, ?)", (cursor.lastrowid, chunk.text))
                self.db.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES(?, ?)", (cursor.lastrowid, blob))

    def delete_document(self, doc_id: str) -> None:
        with self.lock, self.db:
            self._delete_chunks(doc_id)
            self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    # -- reads ------------------------------------------------------------------------------------
    @staticmethod
    def _chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(id=row["id"], doc_id=row["doc_id"], text=row["text"], position=row["position"],
                     metadata=json.loads(row["metadata"]))

    def search_vector(self, vector: Sequence[float], k: int, filters: Filters | None = None) -> list[Hit]:
        blob = pack(vector)
        where, params = filter_sql(filters)
        with self.lock:
            if where:
                # A filtered search scans only the matching rows and computes the exact distance.
                rows = self.db.execute(
                    f"SELECT c.*, vec_distance_cosine(c.embedding, ?) AS distance FROM chunks c "
                    f"WHERE {where} ORDER BY distance LIMIT ?", [blob, *params, k]).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT c.*, v.distance AS distance FROM (SELECT rowid, distance FROM chunks_vec "
                    "WHERE embedding MATCH ? AND k = ?) v JOIN chunks c ON c.rowid = v.rowid ORDER BY v.distance",
                    (blob, k)).fetchall()
        hits = []
        for row in rows:
            similarity = 1.0 - float(row["distance"])
            hits.append(Hit(chunk=self._chunk(row), score=similarity, scores={"vector": similarity}))
        return hits

    def search_lexical(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        match = fts_query(query)
        if not match:
            return []
        where, params = filter_sql(filters)
        sql = ("SELECT c.*, bm25(chunks_fts) AS rank FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
               "WHERE chunks_fts MATCH ?" + (f" AND {where}" if where else "") + " ORDER BY rank LIMIT ?")
        with self.lock:
            rows = self.db.execute(sql, [match, *params, k]).fetchall()
        # bm25() is "more negative is better"; flip it so higher is better like every other score.
        return [Hit(chunk=self._chunk(r), score=-float(r["rank"]), scores={"lexical": -float(r["rank"])})
                for r in rows]

    def count(self) -> int:
        with self.lock:
            return int(self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def document_ids(self) -> list[str]:
        with self.lock:
            return [r["doc_id"] for r in self.db.execute("SELECT doc_id FROM documents ORDER BY doc_id")]

    def close(self) -> None:
        with self.lock:
            self.db.close()
