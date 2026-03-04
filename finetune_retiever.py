import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

# Configuration
# This is the model we are starting with and improving
BASE_MODEL_NAME = 'multi-qa-mpnet-base-dot-v1'
# This is the training data we just created
TRAINING_DATA_PATH = "retriever_training_data.jsonl"
# This is where we will save our new, improved model
FINE_TUNED_MODEL_OUTPUT_PATH = './fine_tuned_retriever_v1'

# Training Hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 16
# The contrastive loss margin. A good default is 0.5.
# It means the model will try to make the (anchor, positive) pair 0.5 closer
# in similarity than the (anchor, negative) pair.
CONTRASTIVE_LOSS_MARGIN = 0.5

def main():
    """
    Loads a pre-trained sentence transformer, fine-tunes it on our user feedback data,
    and saves the improved model to a new directory.
    """
    print("--- Starting Retriever Model Fine-Tuning ---")
    
    # 1. Load the pre-trained model we want to fine-tune
    print(f"1) Loading base model: '{BASE_MODEL_NAME}'...")
    model = SentenceTransformer(BASE_MODEL_NAME)

    # 2. Load our custom training dataset
    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"❌ Error: Training data file not found at '{TRAINING_DATA_PATH}'.")
        print("   Please run 'prepare_finetune_data.py' first.")
        return
        
    print(f"2) Loading training data from '{TRAINING_DATA_PATH}'...")
    train_examples = []
    with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Create an InputExample for each triplet
            example = InputExample(texts=[data['anchor'], data['positive'], data['negative']])
            train_examples.append(example)
    
    print(f"   → Loaded {len(train_examples)} training examples.")

    if not train_examples:
        print("Error: No training examples to process. Exiting.")
        return

    # 3. Set up the data loader and loss function for training
    print("3) Configuring training objectives...")
    # The DataLoader will batch our examples and shuffle them for better training
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    # Use OnlineContrastiveLoss, which is perfect for (anchor, positive, negative) triplets
    train_loss = losses.OnlineContrastiveLoss(model=model, margin=CONTRASTIVE_LOSS_MARGIN)
    print(f"   → Using OnlineContrastiveLoss with a margin of {CONTRASTIVE_LOSS_MARGIN}.")

    # 4. Perform the fine-tuning
    print(f"\n4) Starting fine-tuning for {NUM_EPOCHS} epoch(s)...") 
    # 10% of training steps will be used for a warm-up phase
    warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1) 

    # Fine-tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=NUM_EPOCHS, # Number of epochs to train
              warmup_steps=warmup_steps, # Number of warmup steps for learning rate scheduler
              output_path=FINE_TUNED_MODEL_OUTPUT_PATH, 
              show_progress_bar=True) # Save the model after training
              
    print(f"\n Success! Fine-tuned model saved to '{FINE_TUNED_MODEL_OUTPUT_PATH}'.") # Confirmation message

if __name__ == "__main__":
    main()

