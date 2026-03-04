Semantic Video Transcript Retrieval System
Overview:
This project is a high-performance Hybrid Search and Generative QA system designed to help users find specific moments within video lectures. Unlike standard keyword searches, this system understands the meaning behind your questions to retrieve the most relevant video segments.

Core Features:
Hybrid Retrieval Engine: Combines BM25 (Keyword Search) with Sentence-Transformers (Semantic Search) for maximum accuracy.
Generative AI Answers: Uses the Flan-T5 model to synthesize a clear, natural language answer based strictly on the retrieved video transcripts.
Smart Highlighting: Automatically highlights the specific parts of the transcript that were used to generate the AI's answer.
Interactive UI: A clean, dark-themed Streamlit dashboard that allows users to watch the exact video segment corresponding to their search.

Technical Stack:
Frontend: Streamlit
Vector Search: FAISS (Facebook AI Similarity Search)
Embedding Models: multi-qa-mpnet-base-dot-v1 and all-mpnet-base-v2
LLM (Generative): google/flan-t5-base
Keyword Algorithm: Rank-BM25
