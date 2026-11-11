"""Qdrant (vectors + chunk metadata) and SQLite (document registry + session profile).

Every chunk is ONE Qdrant point with THREE named vectors, all from the same bekko model:
  content : the chunk text                         -> matches details in the body
  section : section heading + keywords + summary   -> matches "what does it say about X"
  doc     : file name + title + summary + keywords -> matches "in the Q3 report ..."
A query is embedded once, searched against all three, and the rankings are fused (weighted RRF).
"""
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from app.kb.embed import DIM, get_embedder
from app.kb.models import Chunk, DocProfile, IngestResult

COLLECTION = "kb_chunks"
WEIGHTS = {"content": 1.0, "section": 0.6, "doc": 0.3}  # how much each vector's rank counts
RRF_K = 60


@dataclass
class Hit:
    text: str
    doc_name: str
    doc_title: str
    section: str
    page_start: int | None
    page_end: int | None
    rrf: float
    relevance: float  # best cosine of the content/section vectors (the doc vector alone never counts)

    @property
    def cite(self) -> str:
        pages = ""
        if self.page_start:
            pages = f", p.{self.page_start}" if self.page_start == self.page_end else f", p.{self.page_start}-{self.page_end}"
        return f"{self.doc_name} › {self.section}{pages}"


def _ids(doc_id: str, n: int) -> list[str]:
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}")) for i in range(n)]


class KnowledgeBase:
    def __init__(self, embedder=None, qdrant_url: str | None = None, db_path: str | Path | None = None):
        self.embedder = embedder or get_embedder()
        url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        # ":memory:" = throwaway, "path:/dir" = embedded Qdrant persisted on disk (no server), otherwise a Qdrant server URL
        self.qdrant = (QdrantClient(":memory:") if url == ":memory:"
                       else QdrantClient(path=url[5:]) if url.startswith("path:") else QdrantClient(url=url))
        if not self.qdrant.collection_exists(COLLECTION):
            self.qdrant.create_collection(
                COLLECTION,
                vectors_config={n: models.VectorParams(size=DIM, distance=models.Distance.COSINE) for n in WEIGHTS},
            )
            for field in ("session_id", "doc_id", "doc_name"):
                self.qdrant.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)
        self.db_path = str(db_path or os.getenv("KB_DB", "kb.sqlite"))
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents(doc_id TEXT PRIMARY KEY, session_id TEXT, filename TEXT, sha TEXT,
                  title TEXT, doc_type TEXT, summary TEXT, keywords TEXT, n_pages INT, ocr_pages INT,
                  n_sections INT, n_chunks INT, created REAL);
                CREATE TABLE IF NOT EXISTS profiles(session_id TEXT PRIMARY KEY, overview TEXT, keywords TEXT, updated REAL);
            """)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    # ---- write ----
    def find_by_sha(self, session_id: str, sha: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM documents WHERE session_id=? AND sha=?", (session_id, sha)).fetchone()
        return dict(row) if row else None

    def add(self, session_id: str, sha: str, result: IngestResult, chunks: list[Chunk]) -> None:
        p = result.profile
        content = self.embedder.embed([c.text for c in chunks])
        section = self.embedder.embed([f"{c.section.heading}. {', '.join(c.section.keywords)}. {c.section.summary}" for c in chunks])
        doc = np.repeat(self.embedder.embed([f"{result.filename} {p.title}. {p.summary} {', '.join(p.keywords)}"]), len(chunks), axis=0)
        points = [
            models.PointStruct(
                id=pid,
                vector={"content": content[i].tolist(), "section": section[i].tolist(), "doc": doc[i].tolist()},
                payload={
                    "session_id": session_id, "doc_id": result.doc_id, "doc_name": result.filename, "doc_title": p.title,
                    "doc_type": p.doc_type, "section": c.section.heading, "section_index": c.section.index,
                    "section_keywords": c.section.keywords, "section_level": c.section.level, "section_summary": c.section.summary, "part": c.part, "page_start": c.section.page_start,
                    "page_end": c.section.page_end, "text": c.text,
                },
            )
            for i, (pid, c) in enumerate(zip(_ids(result.doc_id, len(chunks)), chunks))
        ]
        self.qdrant.upsert(COLLECTION, points=points)
        with self._db() as db:
            db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                result.doc_id, session_id, result.filename, sha, p.title, p.doc_type, p.summary, json.dumps(p.keywords),
                result.n_pages, result.ocr_pages, len(result.sections), len(chunks), time.time()))

    # ---- read ----
    def search(self, session_id: str, query: str, k: int = 5, doc_name: str | None = None) -> list[Hit]:
        must = [models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]
        if doc_name:
            must.append(models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name)))
        flt = models.Filter(must=must)
        qv = self.embedder.embed([query])[0].tolist()
        fused: dict[str, dict] = {}
        for name, weight in WEIGHTS.items():
            pts = self.qdrant.query_points(COLLECTION, query=qv, using=name, limit=20, query_filter=flt, with_payload=True).points
            for rank, pt in enumerate(pts, start=1):
                e = fused.setdefault(str(pt.id), {"payload": pt.payload, "rrf": 0.0, "relevance": 0.0})
                e["rrf"] += weight / (RRF_K + rank)
                if name != "doc":
                    e["relevance"] = max(e["relevance"], pt.score)
        top = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:k]
        return [Hit(e["payload"]["text"], e["payload"]["doc_name"], e["payload"]["doc_title"], e["payload"]["section"],
                    e["payload"]["page_start"], e["payload"]["page_end"], round(e["rrf"], 4), round(e["relevance"], 3)) for e in top]

    def documents(self, session_id: str) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM documents WHERE session_id=? ORDER BY created", (session_id,)).fetchall()
        return [{**dict(r), "keywords": json.loads(r["keywords"])} for r in rows]

    def get_profile(self, session_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM profiles WHERE session_id=?", (session_id,)).fetchone()
        return {"overview": row["overview"], "keywords": json.loads(row["keywords"])} if row else None

    def save_profile(self, session_id: str, overview: str, keywords: list[str]) -> None:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO profiles VALUES (?,?,?,?)", (session_id, overview, json.dumps(keywords), time.time()))

    def prompt_block(self, session_id: str) -> str:
        """What the agent sees on every turn: a compact description of this session's knowledge base."""
        docs, prof = self.documents(session_id), self.get_profile(session_id)
        if not docs:
            return "KNOWLEDGE BASE: empty (no documents yet)."
        lines = [f"KNOWLEDGE BASE ({len(docs)} document{'s' if len(docs) != 1 else ''}, searchable with search_knowledge_base):"]
        if prof:
            lines += [f"Overview: {prof['overview']}", f"Keywords: {', '.join(prof['keywords'])}"]
        lines.append('Documents (to search only one, pass its exact file name as doc_name):')
        lines += [f"- \"{d['filename']}\" [{d['doc_type']}]: {d['title']}. {d['summary'][:170].rstrip()}... (keywords: {', '.join(d['keywords'][:5])})" for d in docs]
        return "\n".join(lines)

    def resolve_doc_name(self, session_id: str, name: str) -> str | None:
        """What the user/model typed -> the exact stored file name. Case-insensitive; then by stem, title or substring.
        Returns None when nothing matches (the tool then lists the valid names)."""
        docs = self.documents(session_id)
        q = name.strip().strip('"\'').lower()
        if not q:
            return None
        for match in (
            lambda d: d["filename"].lower() == q,
            lambda d: d["filename"].lower().rsplit(".", 1)[0] == q.rsplit(".", 1)[0],
            lambda d: q in d["filename"].lower() or d["filename"].lower().rsplit(".", 1)[0] in q,
            lambda d: q in d["title"].lower(),
        ):
            found = [d["filename"] for d in docs if match(d)]
            if len(found) == 1:
                return found[0]
        return None

    def delete_document(self, kb_id: str, doc_id: str) -> bool:
        """PROVIDED: remove ONE document from the knowledge base `kb_id` (vectors, row, and a profile rebuilt from what is left).
        Blocking I/O: call it with asyncio.to_thread. Returns True if the document existed."""
        with self._db() as db:
            if not db.execute("SELECT 1 FROM documents WHERE session_id=? AND doc_id=?", (kb_id, doc_id)).fetchone():
                return False
            db.execute("DELETE FROM documents WHERE session_id=? AND doc_id=?", (kb_id, doc_id))
        self.qdrant.delete(COLLECTION, models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="session_id", match=models.MatchValue(value=kb_id)),
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])))
        left = self.documents(kb_id)
        with self._db() as db:
            if not left:
                db.execute("DELETE FROM profiles WHERE session_id=?", (kb_id,))
                return True
        counts: dict[str, int] = {}
        for d in left:
            for k in d["keywords"]:
                counts[k.lower()] = counts.get(k.lower(), 0) + 1
        prof = self.get_profile(kb_id) or {"overview": "", "keywords": []}
        self.save_profile(kb_id, prof["overview"], sorted(counts, key=counts.get, reverse=True)[:15])
        return True

    def outline(self, kb_id: str, doc_id: str) -> list[dict]:
        """PROVIDED: the sections of one document in reading order:
        [{"index", "heading", "level", "page_start", "page_end", "keywords", "summary"}]. Blocking I/O: use asyncio.to_thread."""
        flt = models.Filter(must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=kb_id)),
                                  models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        found: dict[int, dict] = {}
        offset = None
        while True:
            points, offset = self.qdrant.scroll(COLLECTION, scroll_filter=flt, limit=256, offset=offset, with_payload=True)
            for pt in points:
                pl = pt.payload
                found.setdefault(pl["section_index"], {
                    "index": pl["section_index"], "heading": pl["section"], "level": pl.get("section_level", 1),
                    "page_start": pl["page_start"], "page_end": pl["page_end"], "keywords": pl["section_keywords"],
                    "summary": pl.get("section_summary", "")})
            if offset is None:
                break
        return [found[i] for i in sorted(found)]

    def stats(self, kb_id: str) -> dict:
        """PROVIDED: {"documents": n, "sections": n, "chunks": n, "keywords": [most common first, lowercased]} for one knowledge base."""
        docs = self.documents(kb_id)
        counts: dict[str, int] = {}
        for d in docs:
            for k in d["keywords"]:
                counts[k.lower()] = counts.get(k.lower(), 0) + 1
        return {"documents": len(docs), "sections": sum(d["n_sections"] for d in docs), "chunks": sum(d["n_chunks"] for d in docs),
                "keywords": sorted(counts, key=counts.get, reverse=True)}

    def reset(self, session_id: str) -> None:
        self.qdrant.delete(COLLECTION, models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))])))
        with self._db() as db:
            db.execute("DELETE FROM documents WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM profiles WHERE session_id=?", (session_id,))
