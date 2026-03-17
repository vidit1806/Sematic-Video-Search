# 🎥 Keyword Indexing of subtitled videos

A high-performance hybrid search system that helps users find specific moments within video lectures by understanding the **meaning** behind their questions — not just matching keywords.

---

## 📊 Key Results

| Metric | Score |
|--------|-------|
| Precision@10 | **95%** |
| Mean Reciprocal Rank (MRR) | **1.0** |
| Content discovery improvement | **50% faster** |
| Dataset | 28+ hours of transcripts |

---

## 💡 The Problem

Standard keyword search fails when users don't know the exact terminology used in a lecture. This system solves that by combining **semantic understanding** with **keyword precision** — retrieving the right moment even when the question is phrased differently from the transcript.

---

## ⚙️ How It Works

1. User types a **natural language question**
2. System searches transcripts using **hybrid BM25 + FAISS** retrieval
3. **Flan-T5** generates a clear answer from retrieved segments
4. User sees the exact transcript snippet with **highlighted evidence**

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=flat)
![BM25](https://img.shields.io/badge/BM25-Keyword_Search-green?style=flat)

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Vector Search | FAISS |
| Semantic Embeddings | multi-qa-mpnet-base-dot-v1, all-mpnet-base-v2 |
| Keyword Search | BM25 |
| Generative AI | google/flan-t5-base |
| Data Processing | Regex, NLTK |

---

## ✨ Key Features

- **Hybrid Retrieval** — combines BM25 keyword search with FAISS semantic search using a late-fusion algorithm
- **Generative QA** — Flan-T5 synthesises natural language answers strictly from retrieved transcripts
- **Smart Highlighting** — automatically highlights transcript segments used to generate the answer
- **Interactive UI** — Streamlit dashboard with clickable video timestamps

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/vidit18s/Keyword-Indexing-of-Subtitled-Videos.git
cd Keyword-Indexing-of-Subtitled-Videos
```

**2. Install dependencies**
```bash
pip install -r Requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
├── app.py                        # Streamlit application
├── preprocessing.py              # Data cleaning and indexing
├── create_embeddings.py          # Embedding generation
├── finetune_retriever.py         # Retriever fine-tuning
├── evaluate_all_mpnet.py         # Evaluation scripts
├── generate_test_set_all_mpnet.py
├── split_data.py
├── Requirements.txt
└── README.md
```

---

*MSc Dissertation — University of Liverpool, 2025*
