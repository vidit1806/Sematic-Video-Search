# Import all the necessary libraries for our evaluation
import json
import pandas as pd
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer, util # For semantic search and embeddings
from rank_bm25 import BM25Okapi  # The keyword search algorithm
import matplotlib.pyplot as plt  # The library for creating charts

# Configuration
# These settings control which data and models the script will use.
CORPUS_DATA_PATH = "test_data.json" # The corpus to evaluate against
QUERIES_PATH = "queries_from_training.json" # The test queries to evaluate against
MODEL_NAME = 'all-mpnet-base-v2' # The embedding model to evaluate

# Evaluation Strategy Configuration
# Here we define all the different search strategies we want to A/B test.
HYBRID_WEIGHTS = [
    {"method": "hybrid", "semantic": 0.0, "bm25": 1.0}, # Pure keyword search
    {"method": "hybrid", "semantic": 1.0, "bm25": 0.0}, # Pure semantic search
    {"method": "hybrid", "semantic": 0.50, "bm25": 0.50}, # Balanced approach
    {"method": "hybrid", "semantic": 0.85, "bm25": 0.15}, # Optimal mix based on prior experiments
    {"method": "re_rank", "top_n_bm25": 100} # Rerank top 100 BM25 results using semantic search
]

# Metric Functions
# These are the standard, industry-accepted formulas for measuring search quality.

def precision_at_k(retrieved_ids, relevant_ids, k):
    """Calculates Precision@k: Of the top k results, what fraction were relevant?"""
    if k == 0: return 0.0
    retrieved_k = set(retrieved_ids[:k]) # Top k retrieved documents
    relevant_retrieved = retrieved_k.intersection(relevant_ids) # Relevant documents in top k
    return len(relevant_retrieved) / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    """Calculates Recall@k: Of all possible relevant documents, what fraction did we find in the top k?"""
    if not relevant_ids: return 0.0
    retrieved_k = set(retrieved_ids[:k]) # Top k retrieved documents
    relevant_retrieved = retrieved_k.intersection(relevant_ids) # Relevant documents in top k
    return len(relevant_retrieved) / len(relevant_ids) 

def f1_score_at_k(retrieved_ids, relevant_ids, k):
    """Calculates F1-Score@k: The harmonic mean of precision and recall."""
    precision = precision_at_k(retrieved_ids, relevant_ids, k) 
    recall = recall_at_k(retrieved_ids, relevant_ids, k)    
    if precision + recall == 0: return 0.0  
    return 2 * (precision * recall) / (precision + recall)

def average_precision(retrieved_ids, relevant_ids):
    """Calculates Average Precision (AP): A score that heavily rewards ranking relevant documents higher."""
    if not relevant_ids: return 0.0
    precisions, relevant_count = [], 0
    for i, doc_id in enumerate(retrieved_ids): 
        if str(doc_id) in relevant_ids:     
            relevant_count += 1
            precisions.append(relevant_count / (i + 1)) 
    return sum(precisions) / len(relevant_ids) if precisions else 0.0

def reciprocal_rank(retrieved_ids, relevant_ids):
    """Calculates Reciprocal Rank: How high up is the VERY FIRST relevant result?"""
    for i, doc_id in enumerate(retrieved_ids):
        if str(doc_id) in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

# Curated Plotting Function
def plot_results(results, model_name):
    """
    Plots a curated, clean comparison of the most important retrieval strategies
    and the four most insightful metrics.
    """
    # Step 1: Select only the key strategies we want to visualize
    plot_data = {}
    
    strategies_to_plot = {
        "Keyword (BM25)": "Sem=0.0, BM25=1.0",
        "Semantic": "Sem=1.0, BM25=0.0",
        "Hybrid (Balanced 50/50)": "Sem=0.5, BM25=0.5",
        "Hybrid (Optimal 85/15)": "Sem=0.85, BM25=0.15",
        "Re-ranker": "Re-ranker (top 100)"
    }
    
    for clean_label, raw_label in strategies_to_plot.items():
        if raw_label in results:
            plot_data[clean_label] = results[raw_label]
    
    # Step 2: Extract the four specific metrics we want to plot
    labels = list(plot_data.keys())
    map_scores = [res.get('map', 0) for res in plot_data.values()]
    mrr_scores = [res.get('mrr', 0) for res in plot_data.values()]
    precision_scores = [res.get('precision_at_10', 0) for res in plot_data.values()]
    recall_scores = [res.get('recall_at_10', 0) for res in plot_data.values()]
    
    # Step 3: Create the bar chart layout
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(18, 10))
    
    rects1 = ax.bar(x - width*1.5, map_scores, width, label='MAP')
    rects2 = ax.bar(x - width*0.5, mrr_scores, width, label='MRR')
    rects3 = ax.bar(x + width*0.5, precision_scores, width, label='Precision@10')
    rects4 = ax.bar(x + width*1.5, recall_scores, width, label='Recall@10')

    ax.set_ylabel('Scores', fontsize=14)
    ax.set_title(f'Key Retrieval Strategy Performance ({model_name})', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for rects in [rects1, rects2, rects3, rects4]:
        ax.bar_label(rects, padding=3, fmt='%.3f', rotation=90, size=9)

    ax.set_ylim(0, 1.25)
    fig.tight_layout()
    
    safe_model_name = model_name.replace("/", "_")
    filename = f"evaluation_summary_{safe_model_name}.png"
    plt.savefig(filename)
    print(f"\n✅ Curated summary visualization saved to '{filename}'")

# Main Evaluation Function
def main():
    """The main function that orchestrates the entire evaluation process."""
    print(f"--- RUNNING HOLD-OUT SET EVALUATION FOR '{MODEL_NAME}' MODEL ---")
    
    df = pd.read_json(CORPUS_DATA_PATH)
    df['chunk_id'] = df['chunk_id'].astype(str)
    df = df.set_index('chunk_id')
    
    with open(QUERIES_PATH, 'r', encoding='utf-8') as f:
        queries = json.load(f)["queries"]

    print(f"\nInitializing retrieval model '{MODEL_NAME}'...")
    semantic_model = SentenceTransformer(MODEL_NAME)

    print(f"Generating embeddings for the TEST corpus from '{CORPUS_DATA_PATH}'...")
    corpus_embeddings = semantic_model.encode(df['text_original'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    tokenized_corpus = [doc.split(" ") for doc in df['text_cleaned']]
    bm25 = BM25Okapi(tokenized_corpus)
    chunk_id_list = df.index.tolist()
    
    print("\nGenerating 'pseudo ground truth' for test queries using a HYBRID approach...")
    test_set_with_ground_truth = []
    query_embeddings_for_gt = semantic_model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
    
    for i, query in enumerate(queries):
        semantic_hits = util.semantic_search(query_embeddings_for_gt[i].unsqueeze(0), corpus_embeddings, top_k=10)[0]
        semantic_relevant_ids = {chunk_id_list[hit['corpus_id']] for hit in semantic_hits}
        tokenized_query = query.lower().split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        top_n_bm25_indices = np.argsort(bm25_scores)[::-1][:10]
        bm25_relevant_ids = {chunk_id_list[j] for j in top_n_bm25_indices}
        combined_relevant_chunks = semantic_relevant_ids.union(bm25_relevant_ids)
        test_set_with_ground_truth.append({"query_id": i, "query": query, "relevant_chunks": combined_relevant_chunks})

    overall_results = {}
    for strategy in HYBRID_WEIGHTS:
        method = strategy.get("method")
        if method == "re_rank":
            weight_label = f"Re-ranker (top {strategy.get('top_n_bm25')})"
        else: 
            weight_label = f"Sem={strategy['semantic']}, BM25={strategy['bm25']}"

        print(f"\n{'='*50}\nRUNNING EVALUATION FOR: {weight_label}\n{'='*50}")
        
        all_aps, all_rrs, all_precisions, all_recalls, all_ndcgs, all_f1s = [], [], [], [], [], []

        for test in test_set_with_ground_truth:
            query, relevant_ids = test["query"], test["relevant_chunks"]
            query_embedding = semantic_model.encode(query, convert_to_tensor=True)
            tokenized_query = query.lower().split(" ")
            
            if method == "re_rank":
                top_n = strategy.get('top_n_bm25', 100)
                bm25_scores = bm25.get_scores(tokenized_query)

                top_candidate_indices = np.argsort(bm25_scores)[::-1][:top_n].copy() # fixes numpy warning
                rerank_chunk_ids = [chunk_id_list[i] for i in top_candidate_indices] # Get the chunk IDs for these top candidates
                rerank_embeddings = corpus_embeddings[top_candidate_indices] # Get their embeddings

                rerank_hits = util.semantic_search(query_embedding.unsqueeze(0), rerank_embeddings, top_k=top_n)[0] # Rerank these candidates
                sorted_chunk_ids = [rerank_chunk_ids[hit['corpus_id']] for hit in rerank_hits]  # Map back to original chunk IDs
            else:
                semantic_hits = util.semantic_search(query_embedding.unsqueeze(0), corpus_embeddings, top_k=len(df))[0] # Get all semantic scores
                semantic_score_map = {chunk_id_list[hit['corpus_id']]: hit['score'] for hit in semantic_hits} # Map chunk_id to semantic score
                bm25_scores = bm25.get_scores(tokenized_query) # Get BM25 scores for all documents
                min_bm25, max_bm25 = np.min(bm25_scores), np.max(bm25_scores) # For normalization
                hybrid_scores = {}
                for i, chunk_id in enumerate(chunk_id_list):
                    norm_bm25 = (bm25_scores[i] - min_bm25) / (max_bm25 - min_bm25) if max_bm25 > min_bm25 else 0 # Normalize BM25
                    norm_semantic = semantic_score_map.get(chunk_id, 0) # Already between 0 and 1
                    hybrid_scores[chunk_id] = (strategy["bm25"] * norm_bm25) + (strategy["semantic"] * norm_semantic) # Weighted sum
                sorted_chunk_ids = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True) # Sort by hybrid score
            
            # Calculate all metrics for this query
            all_aps.append(average_precision(sorted_chunk_ids, relevant_ids))
            all_rrs.append(reciprocal_rank(sorted_chunk_ids, relevant_ids))
            all_precisions.append(precision_at_k(sorted_chunk_ids, relevant_ids, k=10))
            all_recalls.append(recall_at_k(sorted_chunk_ids, relevant_ids, k=10))

        overall_results[weight_label] = {
            "map": np.mean(all_aps),
            "mrr": np.mean(all_rrs),
            "precision_at_10": np.mean(all_precisions),
            "recall_at_10": np.mean(all_recalls),
        }
        
        print(f"--- Results for {weight_label} ---")
        for metric, score in overall_results[weight_label].items():
            print(f"  {metric.replace('_', ' ').capitalize():<18}: {score:.4f}")

    if overall_results:
        plot_results(overall_results, MODEL_NAME)

if __name__ == "__main__":
    main()