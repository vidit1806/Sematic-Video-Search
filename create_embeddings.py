import pandas as pd
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer

# --- Configuration ---
DATA_PATH = "processed_transcripts.json"
MODELS_TO_PROCESS = [
    'multi-qa-mpnet-base-dot-v1',
    'all-mpnet-base-v2'
]

def create_and_save_embeddings(): # Create and save embeddings for multiple models
    if not os.path.exists(DATA_PATH): # Check if data file exists
        print(f"Error: Data file not found at '{DATA_PATH}'.") # Error message
        return

    df = pd.read_json(DATA_PATH) # Load data
    corpus = df['text_original'].tolist() # Extract text data

    for model_name in MODELS_TO_PROCESS: # Process each model
        safe_model_name = model_name.replace("/", "_") 
        embeddings_path = f"corpus_embeddings_{safe_model_name}.npy" # Embeddings file path
        
        print(f"Processing model: {model_name}") # Log current model all-mpnet-base-v2/multi-qa-mpnet-base-dot-v1
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
        model = SentenceTransformer(model_name, device=device) # Load model

        corpus_embeddings = model.encode( # Generate embeddings
            corpus, convert_to_tensor=True, show_progress_bar=True # Encode text data
        )

        np.save(embeddings_path, corpus_embeddings.cpu().numpy()) # Save embeddings to file
        print(f"Success! Embeddings file saved to {embeddings_path}") # Confirmation message

if __name__ == "__main__":
    create_and_save_embeddings()