import json
import os

# Configuration
FEEDBACK_LOG_PATH = "feedback_log.jsonl"
OUTPUT_TRAINING_FILE = "retriever_training_data.jsonl"
# This is a key parameter: how many negative examples to create for each positive one.
# More negatives can make training more robust.
NEGATIVES_PER_POSITIVE = 3

def main():
    """
    Reads the feedback log and generates a training dataset of
    (anchor, positive, negative) triplets for fine-tuning the retriever.
    """
    print("--- Preparing Retriever Fine-Tuning Data ---")

    if not os.path.exists(FEEDBACK_LOG_PATH):
        print(f" Error: Feedback log file not found at '{FEEDBACK_LOG_PATH}'.")
        print("   Please run the Streamlit app and collect some user feedback first.")
        return

    good_query_contexts = {} #  {query: [list_of_good_chunks]}
    bad_query_contexts = {}  #  {query: [list_of_bad_chunks]}

    print(f"1) Reading feedback from '{FEEDBACK_LOG_PATH}'...")
    with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            query = entry['query']
            
            # For simplicity, consider the first retrieved chunk as the key context.
            context_chunk = entry['context'].split('\n---\n')[0]

            if entry['rating'] == 'good':
                if query not in good_query_contexts:
                    good_query_contexts[query] = []
                good_query_contexts[query].append(context_chunk)
            elif entry['rating'] == 'bad':
                if query not in bad_query_contexts:
                    bad_query_contexts[query] = []
                bad_query_contexts[query].append(context_chunk)
    
    print(f"   → Found {len(good_query_contexts)} queries with 'good' feedback.")
    print(f"   → Found {len(bad_query_contexts)} queries with 'bad' feedback.")
    
    training_triplets = []
    print("\n2) Generating (anchor, positive, negative) triplets...")

    # Iterate through every query that has at least one "good" rated context
    for query, positive_contexts in good_query_contexts.items():
        # Check if this same query ALSO has "bad" rated contexts
        if query in bad_query_contexts:
            negative_contexts = bad_query_contexts[query]
            
            # Create triplets
            for positive_chunk in positive_contexts:
                # Get a sample of negative chunks for this positive chunk
                negs_sampled = np.random.choice(negative_contexts, size=NEGATIVES_PER_POSITIVE, replace=True)
                for negative_chunk in negs_sampled:
                    # Avoid the rare case where a chunk is somehow both good and bad
                    if positive_chunk != negative_chunk:
                        triplet = {
                            "anchor": query,
                            "positive": positive_chunk,
                            "negative": negative_chunk
                        }
                        training_triplets.append(triplet)

    if not training_triplets:
        print("\n Error: Could not generate any training triplets.")
        print("   This happens when no query has received both a 'good' AND a 'bad' feedback rating.")
        print("   Try to provide more varied feedback in the app.")
        return

    print(f" Generated {len(training_triplets)} training triplets.")

    # 3. Save the dataset to a new file
    print(f"\n3) Saving training data to '{OUTPUT_TRAINING_FILE}'...")
    with open(OUTPUT_TRAINING_FILE, "w", encoding="utf-8") as f:
        for triplet in training_triplets:
            f.write(json.dumps(triplet) + "\n")

    print(f" Success! Fine-tuning data is ready.")

if __name__ == "__main__":
    # Add numpy for sampling
    import numpy as np
    main()