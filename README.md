# Mountain RAG Assistant

A production-ready AI chatbot that answers questions about the world's most famous mountains using Retrieval-Augmented Generation (RAG).

Live Demo: [mountainrag.streamlit.app](https://mountainrag.streamlit.app)

---

## Features

| Feature | Description |
|---|---|
| Query Expansion | Generates 3 alternative phrasings of each question to improve search recall |
| Semantic Cache | Caches answers and returns them instantly for similar questions |
| Multi-session History | Supports multiple conversations in the sidebar |
| Ragas Evaluation | Evaluates each answer on Faithfulness, Relevancy, and Context Precision |
| Hybrid Search | Combines FAISS vector search with BM25 keyword search |
| Cross-Encoder Reranker | Uses BGE-Reranker to select the top 5 most relevant chunks |

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Vector Search | FAISS (HNSW) |
| Keyword Search | BM25 (rank-bm25) |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Reranker | BAAI/bge-reranker-base |
| LLM | Groq LLaMA-3.1-8B-Instant |
| Evaluation | Ragas |

---

## Run Locally

```bash
git clone https://github.com/Abdelrahman-KH-ii/mountain_rag.git
cd mountain_rag
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Deploy on Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and click New app
3. Select the repository and set Main file path to `app.py`
4. Under Advanced settings, add the following secret:
```
GROQ_API_KEY = "your_groq_api_key_here"
```
5. Click Deploy

---

## Project Structure

```
mountain_rag/
├── app.py            - Streamlit UI
├── rag_engine.py     - RAG pipeline (search, rerank, cache, query expansion)
├── requirements.txt
├── rag_index/
│   ├── faiss.index   - FAISS vector index
│   └── chunks.pkl    - Document chunks with metadata
└── README.md
```

---

## How It Works

1. The user asks a question
2. The system generates 3 alternative versions of the question (Query Expansion)
3. All 4 queries run through Hybrid Search (FAISS + BM25)
4. The top 20 results are reranked by a Cross-Encoder model
5. The top 5 chunks are passed as context to Groq LLaMA-3.1-8B
6. The answer is returned and cached for future similar questions
