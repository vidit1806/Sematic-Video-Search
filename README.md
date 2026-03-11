# Semantic Video Transcript Retrieval System

## Overview
A high-performance hybrid search system that helps users find specific moments 
within video lectures by understanding the meaning behind their questions — 
not just matching keywords.

Built as an MSc dissertation project at the University of Liverpool.

## Results
- **95% Precision@10** on retrieval accuracy
- **1.0 Mean Reciprocal Rank (MRR)** — top result is always relevant
- **50% reduction** in content discovery time compared to manual search

## How It Works
1. User types a natural language question
2. System searches transcripts using both keyword and semantic matching
3. Flan-T5 generates a clear answer based on retrieved segments
4. User is shown the exact transcript snippet and can jump to that moment in the video

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Vector Search | FAISS |
| Semantic Embeddings | multi-qa-mpnet-base-dot-v1, all-mpnet-base-v2 |
| Keyword Search | BM25 |
| Generative AI | google/flan-t5-base |
| Data Processing | Regex, NLTK |

## Key Features
- **Hybrid Retrieval** — combines BM25 keyword search with FAISS semantic search using a late-fusion algorithm
- **Generative QA** — Flan-T5 synthesises natural language answers strictly from retrieved transcripts
- **Smart Highlighting** — automatically highlights transcript segments used to generate the answer
- **Interactive UI** — dark-themed Streamlit dashboard with clickable video timestamps

## Project Structure
```
transcript-retrieval/
│
├── app.py                  # Streamlit application
├── retrieval.py            # Hybrid search pipeline
├── preprocessing.py        # Data cleaning and indexing
├── requirements.txt        # Dependencies
└── README.md
```

## Dataset
28+ hours of video lecture transcripts, cleaned and indexed using Regex and NLTK.

## Background
Standard keyword search fails when users don't know the exact terminology 
used in a lecture. This system solves that by combining semantic understanding 
with keyword precision — retrieving the right moment even when the question 
is phrased differently from the transcript.
