import json
import os
import torch
import nltk
import pandas as pd
import spacy # For NLP tasks
from collections import Counter # For counting phrase frequencies
from transformers import T5ForConditionalGeneration, T5Tokenizer # For paraphrasing
from nltk.corpus import stopwords # For stopwords
from sklearn.feature_extraction.text import TfidfVectorizer # For TF-IDF
from sentence_transformers import SentenceTransformer, util # For semantic search and embeddings

# --- Configuration ---
TRAIN_DATA_PATH = "train_data.json"
OUTPUT_QUERIES_PATH = "queries_from_training_multi.json"
FAST_SEMANTIC_MODEL_NAME = 'multi-qa-MiniLM-L6-dot-v1' # Using a fast model for semantic scoring
POWERFUL_SEMANTIC_MODEL_NAME = 'multi-qa-mpnet-base-dot-v1' # Using a more powerful model for semantic scoring
PARAPHRASE_MODEL_NAME = 'google/flan-t5-xl' # Using a powerful paraphrasing model

# Hyperparameters
NGRAM_RANGE = (1, 3) # Using 1-3 grams for better phrase extraction
TARGET_QUERY_COUNT = 100  # target query count for more focused results
TFIDF_PHRASE_COUNT = 2000   # NER phrase count to avoid overfitting
NER_PHRASE_COUNT = 400 # NER phrase count to avoid overfitting
INITIAL_FILTER_COUNT = 600  # initial filter count for faster processing
COHESION_THRESHOLD = 0.20  # Using a more lenient threshold for duplicates
DUPLICATE_QUERY_SIMILARITY_THRESHOLD = 0.90  # Keeping the same threshold for duplicates

#  Blacklist & Setup 
BLACKLIST_TOKENS = set(["let", "lets", "im", "dont", "know", "say", "going", "one", "get", "could", "would", "like", "want", "thing", "things", "lot", "way", "gon", "na", "look", "looks", "see", "mean", "means", "right", "yeah", "okay", "actually", "really", "something", "bit", "of course"])
try:
    stopwords.words("english")
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
STOP_WORDS = set(stopwords.words("english"))
try:
    nlp = spacy.load("en_core_web_trf", disable=["parser"]) 
except OSError:
    print("Downloading spaCy model 'en_core_web_trf'...")
    from spacy.cli import download
    download("en_core_web_trf")
    nlp = spacy.load("en_core_web_trf", disable=["parser"])

# Helper Functions
def pos_filter(phrases):
    filtered, ALLOWED_POS = [], {"NOUN", "PROPN", "ADJ", "VERB", "ADP", "CCONJ"}
    for phrase in phrases:
        if any(tok in BLACKLIST_TOKENS for tok in phrase.split()): continue
        doc = nlp(phrase)
        if len(doc) == 0: continue
        if not (doc[0].pos_ in {"NOUN", "ADJ", "VERB"} and doc[-1].pos_ in {"NOUN", "PROPN", "ADJ"}): continue
        if all(tok.pos_ in ALLOWED_POS for tok in doc if not tok.is_punct): filtered.append(phrase)
    return filtered

def extract_ner_phrases(df, stop_words, blacklist_tokens, top_n=100):
    all_entities = []
    for text in df['text_original']:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "LAW", "WORK_OF_ART", "EVENT", "FAC", "GPE"}:
                clean_ent = ent.text.lower().strip()
                if clean_ent and clean_ent not in stop_words and clean_ent not in blacklist_tokens and len(clean_ent.split()) > 1:
                    all_entities.append(clean_ent)
    return [phrase for phrase, count in Counter(all_entities).most_common(top_n)]

def fast_initial_filter(df, phrases, model, top_k):
    print("   → Stage 1: Fast initial filtering with multi-qa-MiniLM...")
    doc_embeddings = model.encode(df['text_original'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    phrase_embeddings = model.encode(phrases, convert_to_tensor=True, show_progress_bar=True)
    search_results = util.semantic_search(phrase_embeddings, doc_embeddings, top_k=5)
    phrase_scores = {phrases[i]: sum(hit['score'] for hit in result) / len(result) for i, result in enumerate(search_results)}
    top_phrases = sorted(phrase_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    print(f"   → Kept top {len(top_phrases)} candidates after initial filtering.")
    return [phrase for phrase, score in top_phrases]

def score_and_sort_by_cohesion(df, phrases, model):
    print("   → Stage 2: Precise cohesion scoring with multi-qa-MPNet...")
    if not phrases: return []
    doc_embeddings = model.encode(df['text_original'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    phrase_embeddings = model.encode(phrases, convert_to_tensor=True, show_progress_bar=True)
    phrase_scores = {}
    for i, phrase in enumerate(phrases):
        containing_docs_indices = [idx for idx, text in enumerate(df['text_original']) if phrase in text]
        if not containing_docs_indices: continue
        centroid_embedding = torch.mean(doc_embeddings[containing_docs_indices], dim=0)
        cohesion_score = util.cos_sim(phrase_embeddings[i], centroid_embedding).item()
        if cohesion_score >= COHESION_THRESHOLD:
            phrase_scores[phrase] = cohesion_score
    return sorted(phrase_scores.items(), key=lambda item: item[1], reverse=True)

def paraphrase_phrase_with_flan_t5(phrase: str, tokenizer, model):
    prompt = f'Please rephrase the following technical topic into a natural question that a student might ask.\n\nTOPIC: "{phrase}"\n\nQUESTION:'
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def filter_duplicate_queries(queries, model, threshold): 
    """Filters near-duplicate queries using semantic similarity."""
    print(f"\n6) Filtering near-duplicate queries with similarity threshold > {threshold}...")
    if not queries or len(queries) < 2:
        return queries
    
    query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True) # Encode all queries at once
    duplicate_pairs = util.paraphrase_mining_embeddings(
        query_embeddings, # Use precomputed embeddings
        corpus_chunk_size=len(queries), # Process all at once
        top_k=5,    # Check top 5 similar queries
        score_function=util.cos_sim # Use cosine similarity
    )
    
    to_remove = set()
    for score, i, j in duplicate_pairs:
        if score > threshold:
            if i not in to_remove:
                to_remove.add(j)

    final_queries = [query for i, query in enumerate(queries) if i not in to_remove]
    print(f"   → Removed {len(queries) - len(final_queries)} duplicates. Kept {len(final_queries)} unique queries.")
    return final_queries

def main():
    print(f"1) Load TRAINING transcripts from '{TRAIN_DATA_PATH}'...")
    df = pd.read_json(TRAIN_DATA_PATH)
    
    print("\n2) Initializing semantic models...")
    print(f"   → Fast model: '{FAST_SEMANTIC_MODEL_NAME}'")
    fast_model = SentenceTransformer(FAST_SEMANTIC_MODEL_NAME)
    print(f"   → Powerful model: '{POWERFUL_SEMANTIC_MODEL_NAME}'")
    powerful_model = SentenceTransformer(POWERFUL_SEMANTIC_MODEL_NAME)
    
    print(f"\n3) Initializing local paraphrasing model '{PARAPHRASE_MODEL_NAME}'... (This may be slow)")
    para_tokenizer = T5Tokenizer.from_pretrained(PARAPHRASE_MODEL_NAME)
    para_model = T5ForConditionalGeneration.from_pretrained(PARAPHRASE_MODEL_NAME)

    print("\n4) Extracting and refining seed phrases using two-stage semantic funnel...")

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=NGRAM_RANGE, max_features=TFIDF_PHRASE_COUNT, binary=True)
    vectorizer.fit(df["text_cleaned"])
    tfidf_phrases = vectorizer.get_feature_names_out()
    ner_phrases = extract_ner_phrases(df, STOP_WORDS, BLACKLIST_TOKENS, top_n=NER_PHRASE_COUNT)
    combined_phrases = sorted(list(set(tfidf_phrases) | set(ner_phrases)))
    pos_filtered_phrases = pos_filter(combined_phrases)
    print(f"   → Found {len(pos_filtered_phrases)} initial candidate phrases.")

    top_candidates = fast_initial_filter(df, pos_filtered_phrases, fast_model, top_k=INITIAL_FILTER_COUNT)
    sorted_cohesive_phrases = score_and_sort_by_cohesion(df, top_candidates, powerful_model)
    top_phrases_to_paraphrase = [phrase for phrase, score in sorted_cohesive_phrases[:TARGET_QUERY_COUNT]]
    
    if not top_phrases_to_paraphrase:
        print("\n❌ No high-quality phrases were found after filtering. Exiting.")
        return
        
    print(f"   → Selecting top {len(top_phrases_to_paraphrase)} phrases for paraphrasing.")

    print(f"\n5) Paraphrasing phrases into natural questions...")
    paraphrased_queries = []
    for i, phrase in enumerate(top_phrases_to_paraphrase):
        print(f"   ({i+1}/{len(top_phrases_to_paraphrase)}) Paraphrasing: '{phrase}'")
        natural_question = paraphrase_phrase_with_flan_t5(phrase, para_tokenizer, para_model)
        if natural_question and len(natural_question.split()) > 3 and "?" in natural_question:
            paraphrased_queries.append(natural_question)
        else:
            print(f"      ... Discarding low-quality paraphrase: '{natural_question}'")

    # filter out near-duplicate queries, with a threshold of 0.90.
    # powwerful_model multi-qa-mpnet-base-dot-v1/all-base-mpnet-base-v2
    unique_queries = filter_duplicate_queries(paraphrased_queries, powerful_model, DUPLICATE_QUERY_SIMILARITY_THRESHOLD)

    print(f"\n7) Saving {len(unique_queries)} generated queries to {OUTPUT_QUERIES_PATH}…")
    with open(OUTPUT_QUERIES_PATH, 'w', encoding='utf-8') as f:
        json.dump({"queries": unique_queries}, f, indent=2)
    print("✅ Done.")

if __name__ == "__main__" :
    main()