"""
RAG Engine v2 — FAISS + BM25 + Reranker + Query Expansion + Semantic Cache
"""
import pickle
import hashlib
import numpy as np
from datetime import datetime

_engine = None

# ── Semantic Cache ─────────────────────────────────────────────────────────────
class SemanticCache:
    """
    Stores (question_embedding, answer) pairs.
    If a new question is cosine-similar > threshold to a cached one → return cached answer.
    """
    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self.embeddings = []   # list of np arrays
        self.entries    = []   # list of {"question", "answer", "sources", "ts"}

    def _sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def get(self, q_emb: np.ndarray):
        best_score, best_entry = 0, None
        for emb, entry in zip(self.embeddings, self.entries):
            s = self._sim(q_emb, emb)
            if s > best_score:
                best_score, best_entry = s, entry
        if best_score >= self.threshold:
            return best_entry, best_score
        return None, best_score

    def set(self, q_emb: np.ndarray, question: str, answer: str, sources: list):
        self.embeddings.append(q_emb)
        self.entries.append({
            "question": question,
            "answer":   answer,
            "sources":  sources,
            "ts":       datetime.utcnow().isoformat(),
        })

    def all_entries(self):
        return self.entries


# ── RAG Engine ─────────────────────────────────────────────────────────────────
class RAGEngine:
    def __init__(self, index_path: str, chunks_path: str):
        import faiss
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from rank_bm25 import BM25Okapi

        print("⏳ Loading RAG engine...")

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.bm25 = BM25Okapi([c["text"].split() for c in self.chunks])
        self.cache = SemanticCache(threshold=0.92)

        try:
            self.reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
            print("✅ Reranker loaded")
        except Exception as e:
            self.reranker = None
            print(f"⚠️ Reranker unavailable: {e}")

        print(f"✅ Ready — {self.index.ntotal} vectors | {len(self.chunks)} chunks")

    # ── Embed ──────────────────────────────────────────────────────────────────
    def embed(self, text: str) -> np.ndarray:
        return self.embed_model.encode([text], normalize_embeddings=True).astype("float32")[0]

    # ── Query Expansion ────────────────────────────────────────────────────────
    def expand_query(self, query: str, groq_api_key: str) -> list[str]:
        """Ask the LLM to generate 3 alternative phrasings of the query."""
        from groq import Groq
        try:
            client = Groq(api_key=groq_api_key)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Generate 3 different ways to ask the following question. "
                        f"Return ONLY the 3 questions, one per line, no numbering, no extra text.\n\n"
                        f"Question: {query}"
                    )
                }],
                max_tokens=150,
                temperature=0.7,
            )
            lines = resp.choices[0].message.content.strip().split("\n")
            expansions = [l.strip() for l in lines if l.strip()][:3]
            return [query] + expansions   # original + 3 expansions
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}")
            return [query]

    # ── Vector Search ──────────────────────────────────────────────────────────
    def vector_search(self, query: str, top_k: int = 20) -> list:
        q = self.embed(query).reshape(1, -1)
        dists, idxs = self.index.search(q, top_k)
        results = []
        for d, i in zip(dists[0], idxs[0]):
            if i == -1:
                continue
            c = self.chunks[i].copy()
            c["vector_score"] = float(d)
            results.append(c)
        return results

    # ── BM25 Search ────────────────────────────────────────────────────────────
    def bm25_search(self, query: str, top_k: int = 20) -> list:
        scores = self.bm25.get_scores(query.split())
        results = []
        for i in np.argsort(scores)[::-1][:top_k]:
            if scores[i] < 0.01:
                continue
            c = self.chunks[i].copy()
            c["bm25_score"] = float(scores[i])
            results.append(c)
        return results

    # ── Hybrid Search (multi-query) ────────────────────────────────────────────
    def hybrid_search(self, queries: list[str], top_k: int = 20, alpha: float = 0.7) -> list:
        """Run hybrid search over multiple query expansions and merge results."""
        combined = {}
        for query in queries:
            vr = self.vector_search(query, top_k)
            br = self.bm25_search(query, top_k)
            mv = max((r["vector_score"] for r in vr), default=1)
            mb = max((r.get("bm25_score", 0) for r in br), default=1)
            for r in vr:
                g = r["global_id"]
                score = alpha * (r["vector_score"] / (mv + 1e-9))
                if g in combined:
                    combined[g]["hybrid_score"] = max(combined[g]["hybrid_score"], score)
                else:
                    r["hybrid_score"] = score
                    combined[g] = r
            for r in br:
                g = r["global_id"]
                bn = (1 - alpha) * (r.get("bm25_score", 0) / (mb + 1e-9))
                if g in combined:
                    combined[g]["hybrid_score"] += bn
                else:
                    r["hybrid_score"] = bn
                    combined[g] = r
        return sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]

    # ── Rerank ─────────────────────────────────────────────────────────────────
    def rerank(self, query: str, candidates: list, top_n: int = 5) -> list:
        if not candidates:
            return []
        if self.reranker:
            try:
                scores = self.reranker.predict([[query, c["text"]] for c in candidates])
                for c, s in zip(candidates, scores):
                    c["rerank_score"] = float(s)
                return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
            except Exception as e:
                print(f"⚠️ Rerank failed: {e}")
        for c in candidates[:top_n]:
            c["rerank_score"] = c.get("hybrid_score", 0)
        return candidates[:top_n]

    # ── Ask ────────────────────────────────────────────────────────────────────
    def ask(self, query: str, groq_api_key: str) -> tuple[str, list, dict]:
        """
        Returns: (answer, sources, meta)
        meta includes: cache_hit, cache_score, num_expansions
        """
        from groq import Groq

        # 1. Check semantic cache
        q_emb = self.embed(query)
        cached, cache_score = self.cache.get(q_emb)
        if cached:
            return cached["answer"], cached["sources"], {
                "cache_hit": True,
                "cache_score": round(cache_score, 4),
                "expansions": [],
            }

        # 2. Query expansion
        queries = self.expand_query(query, groq_api_key)

        # 3. Hybrid search over all expansions
        candidates = self.hybrid_search(queries, top_k=20)

        # 4. Rerank using original query
        chunks = self.rerank(query, candidates, top_n=5)

        # 5. Build prompt & call LLM
        ctx = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
        prompt = f"<context>\n{ctx}\n</context>\n\nQuestion: {query}\nAnswer:"

        system = """You are an expert assistant specialized in mountains and geography.
1. Answer ONLY from the context between <context> and </context>.
2. If the answer is not found say: I could not find this information in the documents.
3. Always mention the source and page number after each fact.
4. Answer in English only. Be concise and clear."""

        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        answer = response.choices[0].message.content

        sources = [
            {
                "source":   c.get("source", ""),
                "page_num": c.get("page_num", ""),
                "score":    round(c.get("rerank_score", c.get("hybrid_score", 0)), 4),
            }
            for c in chunks
        ]

        # 6. Store in cache
        self.cache.set(q_emb, query, answer, sources)

        return answer, sources, {
            "cache_hit":   False,
            "cache_score": round(cache_score, 4),
            "expansions":  queries[1:],   # the 3 expansions (not original)
        }
