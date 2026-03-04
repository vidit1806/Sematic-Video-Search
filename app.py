# Import all the heavy-lifting libraries we need for our smart search system
import streamlit as st
import pandas as pd
import json
import re
import os
import torch
import numpy as np
import nltk # For sentence tokenization
import faiss # For efficient similarity search
from datetime import datetime 
from rank_bm25 import BM25Okapi # For keyword-based search
from sentence_transformers import SentenceTransformer, util # For semantic search and embeddings
from transformers import T5ForConditionalGeneration, T5Tokenizer # For generative AI

# CONFIGURATION
DATA_PATH = "processed_transcripts.json"
RETRIEVAL_MODELS = {
    "QA-Tuned (Multi-QA MPNet)": "multi-qa-mpnet-base-dot-v1",
    "General Purpose (MPNet)": "all-mpnet-base-v2"
}
GENERATIVE_MODEL_NAME = 'google/flan-t5-base'

# Page Setup & Custom CSS 
st.set_page_config(layout="wide", page_title="Lecture Search", page_icon="🎓")
st.markdown("""
<style>
    /* Main App Style */
    .stApp { background-color: #121212; color: #E0E0E0; }
    h1 { color: #D2B48C; text-align: center; font-family: 'Garamond', serif; }
    h2, h3 { color: #B0C4DE; }
    .st-emotion-cache-16txtl3 { background-color: #1E1E1E; }
    .result-card { border: 1px solid #444; border-radius: 10px; padding: 18px; margin: 15px 0; background-color: #1E1E1E; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); transition: 0.3s; }
    .result-card:hover { box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2); border-color: #D2B48C; }
    .result-card details > summary { font-size: 1.1em; font-weight: bold; cursor: pointer; color: #B0C4DE; }
    .result-card details > p { background-color: #282828; padding: 10px; border-radius: 5px; margin-top: 10px; }
    .watch-link { text-decoration: none; background-color: #D2B48C; color: black !important; padding: 6px 12px; border-radius: 5px; font-weight: bold; display: inline-block; margin-left: 10px; }
    .watch-link:hover { background-color: #C1A37C; color: black !important; }
    .ai-answer-box { background-color: #1E1E1E; border-left: 6px solid #D2B48C; padding: 20px; border-radius: 8px; margin: 20px 0; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# NLTK data check 
@st.cache_resource
def download_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
download_nltk_punkt()

# Helper Functions (WITH CORRECT IMPLEMENTATIONS) 

def log_feedback(query, answer, context, rating, comment=""):
    """Save user feedback to improve our system - like a suggestion box"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "context": "\n---\n".join(context.tolist()),
        "rating": rating,
        "comment": comment
    }
    with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load our video transcript database from a JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.set_index("chunk_id")
    return df

@st.cache_resource
def initialize_models_and_data(_df: pd.DataFrame, retrieval_model_name: str):
    """Load all AI models and prepare search indices"""
    with st.spinner(f"Loading {retrieval_model_name} and building indices..."):
        retrieval_model = SentenceTransformer(retrieval_model_name)
        
        # --- Create Embeddings Directly from the DataFrame ---
        st.info("Generating embeddings for the current dataset... (Cached after first run)")
        corpus_embeddings = retrieval_model.encode(_df['text_original'].tolist(), convert_to_tensor=True, show_progress_bar=True)
        
        corpus_embeddings_np = corpus_embeddings.cpu().numpy().astype('float32')
        faiss.normalize_L2(corpus_embeddings_np)
        
        index = faiss.IndexFlatIP(corpus_embeddings_np.shape[1])
        index.add(corpus_embeddings_np)
        
        tokenizer = T5Tokenizer.from_pretrained(GENERATIVE_MODEL_NAME)
        generative_model = T5ForConditionalGeneration.from_pretrained(GENERATIVE_MODEL_NAME)
        
        tokenized_corpus = [doc.split(" ") for doc in _df["text_cleaned"]]
        bm25 = BM25Okapi(tokenized_corpus)
        
    st.success(f"Models (Retriever: {retrieval_model_name}) initialized!")
    return retrieval_model, index, tokenizer, generative_model, bm25

@st.cache_data(show_spinner=False)
def generate_answer(query: str, _results_df: pd.DataFrame, _tokenizer, _model):
    """Generate a comprehensive answer using AI"""
    if _results_df.empty:
        return "I couldn't find relevant information to answer your question."
    context = "\n---\n".join(_results_df['text_original'].head(5).tolist())
    prompt = f"Synthesize a clear and comprehensive answer based ONLY on the provided context. Do not use outside knowledge. If the context is insufficient, state that the information isn't available.\n\nCONTEXT:\n---\n{context}\n---\nQUESTION: \"{query}\"\n\nANSWER:"
    inputs = _tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    outputs = _model.generate(inputs.input_ids, max_length=512, min_length=64, num_beams=5, early_stopping=True, repetition_penalty=2.0)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)

def highlight_answer_sources(answer: str, text: str, model: SentenceTransformer, threshold=0.6):
    """Highlight parts of text that contributed to the AI's answer"""
    try:
        if not answer: return text
        sentences = nltk.sent_tokenize(text)
        if not sentences: return text
        answer_embedding = model.encode(answer, convert_to_tensor=True)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        similarities = util.cos_sim(answer_embedding, sentence_embeddings)[0]
        highlighted_text = ""
        for i, sentence in enumerate(sentences):
            if similarities[i] > threshold:
                highlighted_text += f' <mark style="background-color: #D2B48C; color: black;">{sentence}</mark>'
            else:
                highlighted_text += f" {sentence}"
        return highlighted_text.strip()
    except Exception:
        return text

def highlight_query_sources(query: str, text: str, model: SentenceTransformer) -> str:
    """Highlight the most relevant sentence for the user's query"""
    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences: return text
        query_embedding = model.encode(query, convert_to_tensor=True)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, sentence_embeddings)
        most_similar_idx = torch.argmax(similarities)
        most_similar_sentence = sentences[most_similar_idx]
        return text.replace(most_similar_sentence, f'<mark style="background-color: #6a6a6a;">{most_similar_sentence}</mark>')
    except Exception:
        return text

def display_results(results_df: pd.DataFrame, query: str, retrieval_model: SentenceTransformer, answer: str | None = None):
    """Show search results in a beautiful, interactive format"""
    if results_df.empty:
        st.warning("No relevant video segments found.")
        return
    st.subheader(f"🎥 Top {len(results_df)} Relevant Video Segments:")
    query_keywords = set(re.findall(r'\b\w{3,}\b', query.lower()))
    for index, row in results_df.iterrows():
        video_link, result_text = row['video_url'], row['text_original']
        if answer:
            highlighted_text = highlight_answer_sources(answer, result_text, retrieval_model)
        else:
            highlighted_text = highlight_query_sources(query, result_text, retrieval_model)
        for word in query_keywords:
            highlighted_text = re.sub(f"(\\b{re.escape(word)}\\b)", r"<mark>\1</mark>", highlighted_text, flags=re.IGNORECASE)
        confidence_score = row.get('confidence', 0.0)
        st.markdown(f'<div class="result-card"><p><strong>Confidence: {confidence_score:.2%}</strong><a href="{video_link}" target="_blank" class="watch-link">🎬 Watch Segment ({row["start"]} - {row["end"]})</a></p><details open><summary>Show transcript segment</summary><p>{highlighted_text}</p></details></div>', unsafe_allow_html=True)

# Main Streamlit App
st.title("Semantic Video Transcript Retrieval System")
st.subheader("A hybrid retrieval and generative system for semantic search over video transcripts")

try:
    df = load_data(DATA_PATH)
    
    with st.sidebar:
        st.header("⚙️ Controls")
        model_choice_name = st.radio("Select Retrieval Model:", RETRIEVAL_MODELS.keys(), index=0, help="Switching models requires a one-time load.")
        selected_model_id = RETRIEVAL_MODELS[model_choice_name]
        search_type = st.radio("Search Method:", ["Hybrid Search (Recommended)", "Semantic Search", "Keyword Search"], index=0)
        top_n = st.slider("Max # of results:", min_value=1, max_value=10, value=5)
        st.markdown("---")
        st.info("Ask a question to find relevant clips and get a summary.")
        st.markdown("---")
        st.subheader("Example Questions:")
        if st.button("What is a bayesian filter?"):
            st.session_state.run_search = True
            st.session_state.query_text = "what is a bayesian filter for robot localization?"
        if st.button("Explain transformer architecture"):
            st.session_state.run_search = True
            st.session_state.query_text = "can you explain the transformer architecture?"

    retrieval_model, faiss_index, tokenizer, generative_model, bm25 = initialize_models_and_data(df, selected_model_id)
    chunk_id_list = df.index.tolist()

except FileNotFoundError:
    st.error(f"Error: Critical data file '{DATA_PATH}' not found. Please run preprocessing.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during initialization: {e}")
    st.stop()

if 'run_search' not in st.session_state: st.session_state.run_search = False
if 'query_text' not in st.session_state: st.session_state.query_text = ""
if 'results_df' not in st.session_state: st.session_state.results_df = pd.DataFrame()
if 'answer' not in st.session_state: st.session_state.answer = None

with st.form(key='search_form'):
    query_text = st.text_input("🔍 Ask your question here:", value=st.session_state.query_text)
    submit_button = st.form_submit_button(label='Search')
    if submit_button and query_text:
        st.session_state.run_search = True
        st.session_state.query_text = query_text

if st.session_state.run_search and st.session_state.query_text:
    query = st.session_state.query_text
    st.session_state.results_df = pd.DataFrame()
    st.session_state.answer = None

    with st.spinner("🧠 Finding relevant video segments..."):
        query_embedding_1d = retrieval_model.encode(query, convert_to_tensor=False).astype('float32')
        query_embedding_2d = query_embedding_1d.reshape(1, -1)
        faiss.normalize_L2(query_embedding_2d)
        
        distances, indices = faiss_index.search(query_embedding_2d, faiss_index.ntotal)
        
        semantic_scores = {
            chunk_id_list[idx]: score 
            for idx, score in zip(indices[0], distances[0]) 
            if idx != -1
        }
        
        if search_type == "Hybrid Search (Recommended)":
            if selected_model_id == 'multi-qa-mpnet-base-dot-v1':
                semantic_weight, keyword_weight = 0.85, 0.15 # Heavily favor semantic for QA-tuned models
            else:
                semantic_weight, keyword_weight = 0.50, 0.50 # Balanced for general models
            
            tokenized_query = query.lower().split(" ")
            bm25_scores_raw = bm25.get_scores(tokenized_query)
            min_bm25, max_bm25 = np.min(bm25_scores_raw), np.max(bm25_scores_raw)
            norm_bm25_scores = (bm25_scores_raw - min_bm25) / (max_bm25 - min_bm25) if max_bm25 > min_bm25 else bm25_scores_raw
            keyword_scores = {cid: score for cid, score in zip(chunk_id_list, norm_bm25_scores)}
            
            hybrid_scores = {cid: (semantic_weight * semantic_scores.get(cid, 0)) + (keyword_weight * keyword_scores.get(cid, 0)) for cid in df.index}
            top_chunk_ids = sorted(hybrid_scores.keys(), key=lambda id: hybrid_scores[id], reverse=True)[:top_n]
            if top_chunk_ids:
                st.session_state.results_df = df.loc[top_chunk_ids].copy()
                st.session_state.results_df['confidence'] = [hybrid_scores[cid] for cid in top_chunk_ids]
        
        elif search_type == "Semantic Search": # FAISS only
            valid_indices = [idx for idx in indices[0][:top_n] if idx != -1]
            top_chunk_ids = [chunk_id_list[i] for i in valid_indices]
            st.session_state.results_df = df.loc[top_chunk_ids].copy()
            st.session_state.results_df['confidence'] = [distances[0][i] for i, idx in enumerate(indices[0][:top_n]) if idx != -1]

        elif search_type == "Keyword Search": # BM25 only
            tokenized_query = query.lower().split(" ")
            bm25_scores = bm25.get_scores(tokenized_query)
            top_indices_numeric = np.argsort(bm25_scores)[::-1][:top_n]
            top_chunk_ids = [chunk_id_list[i] for i in top_indices_numeric]
            st.session_state.results_df = df.loc[top_chunk_ids].copy()
            
            top_scores_raw = bm25_scores[top_indices_numeric]
            min_score, max_score = np.min(bm25_scores), np.max(bm25_scores)
            if max_score > min_score:
                normalized_scores = (top_scores_raw - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(top_scores_raw)
            st.session_state.results_df['confidence'] = normalized_scores
            
    if not st.session_state.results_df.empty and search_type != "Keyword Search":
        with st.spinner("✍️ Formulating a response..."):
            st.session_state.answer = generate_answer(query, st.session_state.results_df, tokenizer, generative_model)

    st.session_state.run_search = False # Reset search trigger

if st.session_state.query_text: # Display the query
    if st.session_state.answer:
        st.subheader("💡In-Short")
        st.markdown(f'<div class="ai-answer-box"><p>{st.session_state.answer}</p></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.write("Was this answer helpful?")
        col1, col2 = st.columns([1,10])
        with col1:
            if st.button("👍"):
                log_feedback(st.session_state.query_text, st.session_state.answer, st.session_state.results_df['text_original'], "good")
                st.toast("Feedback submitted!")
        with col2: 
            if st.button("👎"):
                log_feedback(st.session_state.query_text, st.session_state.answer, st.session_state.results_df['text_original'], "bad")
                st.toast("Feedback submitted!") 
                
    display_results(st.session_state.results_df, st.session_state.query_text, retrieval_model, st.session_state.answer) # Display results with or without AI answer


