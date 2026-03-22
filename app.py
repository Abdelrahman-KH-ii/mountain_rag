"""
Mountain RAG Assistant v2
Features: Query Expansion | Semantic Cache | Multi-session History | Ragas Eval
"""
import os
import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

st.set_page_config(
    page_title="Mountain RAG Assistant",
    page_icon="Mountain",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner="Loading AI models... first time takes ~30s")
def get_engine():
    from rag_engine import RAGEngine
    return RAGEngine(
        index_path=os.path.join(os.path.dirname(__file__), "rag_index", "faiss.index"),
        chunks_path=os.path.join(os.path.dirname(__file__), "rag_index", "chunks.pkl"),
    )

engine = get_engine()

if "conversations" not in st.session_state:
    first_id = str(uuid.uuid4())[:8]
    st.session_state.conversations = {
        first_id: {"title": "New Chat", "messages": []}
    }
    st.session_state.active_conv = first_id

if "active_conv" not in st.session_state:
    st.session_state.active_conv = list(st.session_state.conversations.keys())[0]

def active_messages():
    return st.session_state.conversations[st.session_state.active_conv]["messages"]

def new_conversation():
    cid = str(uuid.uuid4())[:8]
    st.session_state.conversations[cid] = {"title": "New Chat", "messages": []}
    st.session_state.active_conv = cid

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Mountain RAG")
    st.divider()

    if st.button("New Chat", use_container_width=True):
        new_conversation()
        st.rerun()

    st.markdown("### Chat History")

    for cid, conv in st.session_state.conversations.items():
        is_active = cid == st.session_state.active_conv
        label = f"{'> ' if is_active else ''}{conv['title']}"
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f"conv_{cid}", use_container_width=True):
                st.session_state.active_conv = cid
                st.rerun()
        with col2:
            if st.button("X", key=f"del_{cid}"):
                del st.session_state.conversations[cid]
                if not st.session_state.conversations:
                    new_conversation()
                else:
                    st.session_state.active_conv = list(st.session_state.conversations.keys())[0]
                st.rerun()

    st.divider()

    cache_entries = engine.cache.all_entries()
    st.markdown(f"**Semantic Cache:** {len(cache_entries)} entries")
    if cache_entries:
        with st.expander("Cached Questions"):
            for e in cache_entries:
                st.caption(f"- {e['question'][:50]}")

    st.divider()
    st.caption(f"{engine.index.ntotal} vectors | {len(engine.chunks)} chunks")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Mountain RAG Assistant")

with st.container(border=True):
    st.markdown(
        "**AI assistant for the world's most famous mountains** — "
        "Hybrid Search (FAISS + BM25) · Cross-Encoder Reranking · "
        "Query Expansion · Semantic Caching · Groq LLaMA-3.1-8B"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("Geography and geology facts")
    c2.markdown("Source and page per answer")
    c3.markdown("Query expansion (3x)")
    c4.markdown("Semantic cache")

st.divider()

# ── Chat messages ─────────────────────────────────────────────────────────────
messages = active_messages()

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        meta = msg.get("meta", {})
        if meta and msg["role"] == "assistant":
            badge_cols = st.columns(4)
            if meta.get("cache_hit"):
                badge_cols[0].success(f"Cache hit ({meta['cache_score']:.2f})")
            else:
                badge_cols[0].info("Live search")
            exps = meta.get("expansions", [])
            if exps:
                badge_cols[1].info(f"{len(exps)} expansions used")

        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** | score: `{s['score']:.4f}`")

        if msg.get("ragas"):
            with st.expander("Ragas Evaluation"):
                r = msg["ragas"]
                cols = st.columns(3)
                cols[0].metric("Faithfulness",      f"{r.get('faithfulness', 'N/A')}")
                cols[1].metric("Answer Relevancy",   f"{r.get('answer_relevancy', 'N/A')}")
                cols[2].metric("Context Precision",  f"{r.get('context_precision', 'N/A')}")

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about mountains... e.g. What is the height of Everest?")

if user_input:
    msgs = active_messages()

    conv = st.session_state.conversations[st.session_state.active_conv]
    if conv["title"] == "New Chat":
        conv["title"] = user_input[:35] + ("..." if len(user_input) > 35 else "")

    msgs.append({"role": "user", "content": user_input, "sources": [], "meta": {}})

    with st.spinner("Expanding query and searching..."):
        try:
            answer, sources, meta = engine.ask(user_input, GROQ_API_KEY)
        except Exception as e:
            answer = f"Error: {str(e)}"
            sources, meta = [], {}

    msgs.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
        "meta":    meta,
        "ragas":   None,
    })

    st.rerun()

# ── Ragas Eval button ─────────────────────────────────────────────────────────
msgs = active_messages()
if len(msgs) >= 2 and msgs[-1]["role"] == "assistant" and msgs[-1]["ragas"] is None:
    st.divider()
    if st.button("Evaluate last answer with Ragas", use_container_width=True):
        last_q = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), None)
        last_a = msgs[-1]["content"]
        last_sources = msgs[-1].get("sources", [])
        contexts = [s["source"] for s in last_sources]

        with st.spinner("Running Ragas evaluation..."):
            try:
                from datasets import Dataset
                from ragas import evaluate
                from ragas.metrics import faithfulness, answer_relevancy, context_precision

                dataset = Dataset.from_dict({
                    "question":     [last_q],
                    "answer":       [last_a],
                    "contexts":     [contexts],
                    "ground_truth": [last_a],
                })
                result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
                msgs[-1]["ragas"] = {
                    "faithfulness":      round(result["faithfulness"], 3),
                    "answer_relevancy":  round(result["answer_relevancy"], 3),
                    "context_precision": round(result["context_precision"], 3),
                }
            except Exception as e:
                msgs[-1]["ragas"] = {"error": str(e)}

        st.rerun()