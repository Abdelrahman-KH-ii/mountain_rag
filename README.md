# 🏔️ Mountain RAG Assistant v2

AI chatbot للإجابة على أسئلة عن أشهر جبال العالم — Production-Ready RAG.

## ✨ المميزات

| Feature | الوصف |
|---|---|
| 🔀 Query Expansion | بيولد 3 صيغ مختلفة للسؤال لتحسين البحث |
| ⚡ Semantic Cache | بيحفظ الإجابات ويرجعها فورًا لو السؤال متشابه |
| 💬 Multi-session History | محادثات متعددة في الـ sidebar |
| 📊 Ragas Evaluation | بيقيّم كل إجابة (Faithfulness / Relevancy / Precision) |
| 🔍 Hybrid Search | FAISS vector + BM25 keyword |
| 🎯 Cross-Encoder Reranker | BGE-Reranker لأدق 5 نتائج |

## 🚀 تشغيل محلي

```bash
git clone https://github.com/YOUR_USERNAME/mountain-rag.git
cd mountain-rag
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy على Streamlit Cloud

1. ارفع على GitHub
2. روح [share.streamlit.io](https://share.streamlit.io) → New app
3. اختار الريبو → Main file: `app.py`
4. في **Secrets**:
```
GROQ_API_KEY = "your_key_here"
```
5. Deploy ✅

## 🗂️ هيكل المشروع

```
mountain-rag/
├── app.py           ← Streamlit UI
├── rag_engine.py    ← RAG + Cache + Query Expansion
├── requirements.txt
├── .env
└── rag_index/
    ├── faiss.index
    └── chunks.pkl
```
