# Import Libraries - The arsenal for intelligent query generation from video transcripts
import json  # JSON handling - read training data and save generated queries
import os  # File system operations - check paths and manage files
import torch  # PyTorch - tensor operations for neural networks
import nltk  # Natural Language Toolkit - sentence splitting and stopwords
import pandas as pd  # Data manipulation - handle our transcript dataset
import spacy  # Advanced NLP - named entity recognition and POS tagging
from collections import Counter  # Count frequencies - find most common phrases
from transformers import T5ForConditionalGeneration, T5Tokenizer  # Google's T5 - paraphrase phrases into questions
from nltk.corpus import stopwords  # Filter out common words like "the", "and"
from sklearn.feature_extraction.text import TfidfVectorizer  # Classical IR - find important phrases using statistics
from sentence_transformers import SentenceTransformer, util  # Neural embeddings - understand semantic meaning

# Configuration
TRAIN_DATA_PATH = "train_data.json"  # Input: processed transcript chunks for training
OUTPUT_QUERIES_PATH = "queries_from_training_all.json"  # Output: generated natural questions
FAST_SEMANTIC_MODEL_NAME = 'multi-qa-MiniLM-L6-dot-v1'  # Fast model for initial filtering - sacrifices some accuracy for speed
POWERFUL_SEMANTIC_MODEL_NAME = 'all-mpnet-base-v2'  # Powerful model for final scoring - high accuracy but slower
PARAPHRASE_MODEL_NAME = 'google/flan-t5-xl'  # Large language model for converting phrases to natural questions

# Hyperparameters - Fine-tuned settings that control the quality vs quantity tradeoff
NGRAM_RANGE = (1, 3)  # Extract 1-3 word phrases - captures both single concepts and compound terms
TARGET_QUERY_COUNT = 100  # How many high-quality queries we want to generate
TFIDF_PHRASE_COUNT = 2000  # Initial TF-IDF phrases to consider - cast a wide net first
NER_PHRASE_COUNT = 400  # Named entities to extract - organizations, products, concepts
INITIAL_FILTER_COUNT = 600 # Candidates after fast filtering - reduce computational load
COHESION_THRESHOLD = 0.20  # Semantic similarity threshold - how related must phrase be to its contexts
DUPLICATE_QUERY_SIMILARITY_THRESHOLD = 0.90  # How similar before we consider queries duplicates

# Blacklist & Setup - Filtering out low-value words that don't make good search queries
BLACKLIST_TOKENS = set([  # Common filler words that don't add semantic value
    "let", "lets", "im", "dont", "know", "say", "going", "one", "get", "could", 
    "would", "like", "want", "thing", "things", "lot", "way", "gon", "na", "look", 
    "looks", "see", "mean", "means", "right", "yeah", "okay", "actually", "really", 
    "something", "bit", "of course"
])

# Download required NLTK data if missing - ensure we have the language tools we need
try:
    stopwords.words("english")  # Check if stopwords exist
    nltk.data.find('tokenizers/punkt')  # Check if sentence tokenizer exists
except LookupError:  # Download if missing
    nltk.download('stopwords')  # Common English words to filter out
    nltk.download('punkt')  # Sentence boundary detection

STOP_WORDS = set(stopwords.words("english"))  # Convert to set for fast lookup

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

# --- Helper Functions --- The specialized tools for intelligent phrase extraction and processing

def pos_filter(phrases):
    """Filter phrases based on grammatical structure - only keep phrases that form coherent concepts"""
    filtered = []  # Our curated list of grammatically sound phrases
    ALLOWED_POS = {"NOUN", "PROPN", "ADJ", "VERB", "ADP", "CCONJ"}  # Parts of speech that add meaning
    
    for phrase in phrases:
        # Skip phrases containing blacklisted filler words
        if any(tok in BLACKLIST_TOKENS for tok in phrase.split()): 
            continue
            
        doc = nlp(phrase)  # Analyze grammatical structure
        if len(doc) == 0: 
            continue  # Skip empty phrases
            
        # Ensure phrase starts and ends with meaningful words
        if not (doc[0].pos_ in {"NOUN", "ADJ", "VERB"} and doc[-1].pos_ in {"NOUN", "PROPN", "ADJ"}): 
            continue
            
        # Check that all words contribute semantic meaning
        if all(tok.pos_ in ALLOWED_POS for tok in doc if not tok.is_punct): 
            filtered.append(phrase)
    
    return filtered

def extract_ner_phrases(df, stop_words, blacklist_tokens, top_n=100):
    """Extract named entities from transcripts - find specific concepts, organizations, products"""
    all_entities = []  # Collect all named entities across all transcripts
    
    for text in df['text_original']:  # Process each transcript
        doc = nlp(text)  # Run NER on the text
        for ent in doc.ents:  # Examine each detected entity
            # Focus on entity types that make good search queries
            if ent.label_ in {"ORG", "PRODUCT", "LAW", "WORK_OF_ART", "EVENT", "FAC", "GPE"}:
                clean_ent = ent.text.lower().strip()  # Normalize the entity
                # Quality filters: not empty, not stopword, not blacklisted, multi-word
                if (clean_ent and clean_ent not in stop_words and 
                    clean_ent not in blacklist_tokens and len(clean_ent.split()) > 1):
                    all_entities.append(clean_ent)
    
    # Return most frequent entities - popularity indicates importance
    return [phrase for phrase, count in Counter(all_entities).most_common(top_n)]

def fast_initial_filter(df, phrases, model, top_k):
    """Stage 1 filtering: Using fast model to quickly eliminate poor candidates"""
    print("   → Stage 1: Fast initial filtering with multi-qa-MiniLM...")
    
    # Encode all document content into vector representations
    doc_embeddings = model.encode(df['text_original'].tolist(), 
                                 convert_to_tensor=True, 
                                 show_progress_bar=True)
    
    # Encode all candidate phrases into vector representations
    phrase_embeddings = model.encode(phrases, 
                                   convert_to_tensor=True, 
                                   show_progress_bar=True)
    
    # For each phrase, find the 5 most semantically similar documents
    search_results = util.semantic_search(phrase_embeddings, doc_embeddings, top_k=5)
    
    # Calculate average similarity score for each phrase
    phrase_scores = {
        phrases[i]: sum(hit['score'] for hit in result) / len(result) 
        for i, result in enumerate(search_results)
    }
    
    # Keep only the top-scoring phrases for expensive second-stage processing
    top_phrases = sorted(phrase_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    print(f"   → Kept top {len(top_phrases)} candidates after initial filtering.")
    
    return [phrase for phrase, score in top_phrases]

def score_and_sort_by_cohesion(df, phrases, model):
    """Stage 2 filtering: Use powerful model to measure semantic cohesion - academic concept"""
    print("   → Stage 2: Precise cohesion scoring with multi-qa-MPNet...")
    if not phrases: 
        return []
    
    # Encode documents with the more powerful model for precise similarity measurement
    doc_embeddings = model.encode(df['text_original'].tolist(), 
                                 convert_to_tensor=True, 
                                 show_progress_bar=True)
    
    # Encode phrases with the powerful model
    phrase_embeddings = model.encode(phrases, 
                                   convert_to_tensor=True, 
                                   show_progress_bar=True)
    
    phrase_scores = {}
    
    for i, phrase in enumerate(phrases):
        # Find documents that actually contain this phrase
        containing_docs_indices = [
            idx for idx, text in enumerate(df['text_original']) 
            if phrase in text
        ]
        
        if not containing_docs_indices: 
            continue  # Skip phrases not found in any document
        
        # Calculate semantic centroid of documents containing this phrase
        centroid_embedding = torch.mean(doc_embeddings[containing_docs_indices], dim=0)
        
        # Measure how well the phrase represents its contextual cluster
        cohesion_score = util.cos_sim(phrase_embeddings[i], centroid_embedding).item()
        
        # Only keep phrases that are semantically cohesive with their contexts
        if cohesion_score >= COHESION_THRESHOLD:
            phrase_scores[phrase] = cohesion_score
    
    # Return phrases sorted by their semantic cohesion quality
    return sorted(phrase_scores.items(), key=lambda item: item[1], reverse=True)

def paraphrase_phrase_with_flan_t5(phrase: str, tokenizer, model):
    """Convert technical phrases into natural questions using large language model"""
    # Craft a careful prompt to get natural-sounding questions
    prompt = f'''Please rephrase the following technical topic into a natural question that a student might ask.

TOPIC: "{phrase}"

QUESTION:'''
    
    # Tokenize the prompt for the model
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate natural question using beam search for quality
    outputs = model.generate(
        inputs.input_ids, 
        max_length=64,      # Reasonable question length
        num_beams=4,        # Beam search for better quality
        early_stopping=True # Stop when complete
    )
    
    # Convert tokens back to readable text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def filter_duplicate_queries(queries, model, threshold):
    """Filters near-duplicate queries using semantic similarity - avoid redundant questions"""
    print(f"\n6) Filtering near-duplicate queries with similarity threshold > {threshold}...")
    if not queries or len(queries) < 2:  # Need at least 2 queries to find duplicates
        return queries
    
    # Encode all queries for similarity comparison
    query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
    
    # Find pairs of similar queries using efficient paraphrase mining
    duplicate_pairs = util.paraphrase_mining_embeddings(
        query_embeddings,  # Pre-computed embeddings
        corpus_chunk_size=len(queries),  # Process all at once
        top_k=5,  # Find top 5 most similar pairs for each query
        score_function=util.cos_sim  # Use cosine similarity
    )
    # --- END FIX ---
    
    # Mark queries for removal - keep first occurrence, remove duplicates
    to_remove = set()
    for score, i, j in duplicate_pairs:  # score, query_index_1, query_index_2
        if score > threshold:  # Queries too similar
            if i not in to_remove:  # Keep the first one
                to_remove.add(j)  # Remove the second one

    # Filter out marked duplicates
    final_queries = [query for i, query in enumerate(queries) if i not in to_remove]
    print(f"   → Removed {len(queries) - len(final_queries)} duplicates. Kept {len(final_queries)} unique queries.")
    
    return final_queries

def main():
    """Main orchestrator - coordinates the entire query generation pipeline"""
    
    # Step 1: Load training data
    print(f"1) Load TRAINING transcripts from '{TRAIN_DATA_PATH}'...")
    df = pd.read_json(TRAIN_DATA_PATH)  # Load processed transcript chunks
    
    # Step 2: Initialize AI models - two-stage approach for efficiency
    print("\n2) Initializing semantic models...")
    print(f"   → Fast model: '{FAST_SEMANTIC_MODEL_NAME}'") # FAST_SEMANTIC_MODEL_NAME='multi-qa-MiniLM-L6-dot-v1'
    fast_model = SentenceTransformer(FAST_SEMANTIC_MODEL_NAME)  # Quick initial filtering
    
    print(f"   → Powerful model: '{POWERFUL_SEMANTIC_MODEL_NAME}'")
    powerful_model = SentenceTransformer(POWERFUL_SEMANTIC_MODEL_NAME)  # Precise final scoring
    
    # Step 3: Load text generation model for paraphrasing
    print(f"\n3) Initializing local paraphrasing model '{PARAPHRASE_MODEL_NAME}'... (This may be slow)")
    para_tokenizer = T5Tokenizer.from_pretrained(PARAPHRASE_MODEL_NAME)  # Tokenizer for T5
    para_model = T5ForConditionalGeneration.from_pretrained(PARAPHRASE_MODEL_NAME)  # T5 model itself

    # Step 4: Extract seed phrases using multiple complementary techniques
    print("\n4) Extracting and refining seed phrases using two-stage semantic funnel...")
    
    # Technique 1: TF-IDF - Statistical importance based on term frequency
    vectorizer = TfidfVectorizer(
        stop_words='english',  # Remove common words
        ngram_range=NGRAM_RANGE,  # Extract 1-3 word phrases
        max_features=TFIDF_PHRASE_COUNT,  # Limit to most important phrases
        binary=True  # Binary occurrence (not frequency) for phrase extraction
    )
    vectorizer.fit(df["text_cleaned"])  # Learn from cleaned text
    tfidf_phrases = vectorizer.get_feature_names_out()  # Get extracted phrases
    
    # Technique 2: Named Entity Recognition - Find specific concepts and entities
    ner_phrases = extract_ner_phrases(df, STOP_WORDS, BLACKLIST_TOKENS, top_n=NER_PHRASE_COUNT)
    
    # Combine both approaches for comprehensive coverage
    combined_phrases = sorted(list(set(tfidf_phrases) | set(ner_phrases)))  # Union of both sets
    
    # Filter phrases based on grammatical structure
    pos_filtered_phrases = pos_filter(combined_phrases)
    print(f"   → Found {len(pos_filtered_phrases)} initial candidate phrases.")

    # Step 5: Two-stage semantic filtering - efficiency meets precision
    
    # Stage 1: Fast filtering with lightweight model 'multi-qa-MiniLM'
    top_candidates = fast_initial_filter(df, pos_filtered_phrases, fast_model, top_k=INITIAL_FILTER_COUNT)
    
    # Stage 2: Precise scoring with powerful model
    sorted_cohesive_phrases = score_and_sort_by_cohesion(df, top_candidates, powerful_model) # Using 'all-mpnet-base-v2'
    
    # Select the best phrases for paraphrasing
    top_phrases_to_paraphrase = [phrase for phrase, score in sorted_cohesive_phrases[:TARGET_QUERY_COUNT]]
    
    if not top_phrases_to_paraphrase:  # Safety check
        print("\n❌ No high-quality phrases were found after filtering. Exiting.")
        return
        
    print(f"   → Selecting top {len(top_phrases_to_paraphrase)} phrases for paraphrasing.")

    # Step 6: Transform phrases into natural questions using AI
    print(f"\n5) Paraphrasing phrases into natural questions...")
    paraphrased_queries = []
    
    for i, phrase in enumerate(top_phrases_to_paraphrase):
        print(f"   ({i+1}/{len(top_phrases_to_paraphrase)}) Paraphrasing: '{phrase}'")
        
        # Use T5 to generate natural question from technical phrase
        natural_question = paraphrase_phrase_with_flan_t5(phrase, para_tokenizer, para_model)
        
        # Quality check: ensure it's a proper question
        if (natural_question and 
            len(natural_question.split()) > 3 and  # Not too short
            "?" in natural_question):  # Actually a question
            paraphrased_queries.append(natural_question)
        else:
            print(f"      ... Discarding low-quality paraphrase: '{natural_question}'")

    # Step 7: Remove near-duplicate questions for diversity
    unique_queries = filter_duplicate_queries(paraphrased_queries, powerful_model, DUPLICATE_QUERY_SIMILARITY_THRESHOLD)

    # Step 8: Save the generated query dataset
    print(f"\n7) Saving {len(unique_queries)} generated queries to {OUTPUT_QUERIES_PATH}…")
    with open(OUTPUT_QUERIES_PATH, 'w', encoding='utf-8') as f:
        json.dump({"queries": unique_queries}, f, indent=2)  # Pretty-printed JSON
    
    print("✅ Done.")  # Success message

if __name__ == "__main__":  # Only run if script executed directly
    main()  # Kick off the main process 