import pandas as pd
import json
import math
import numpy as np

# --- Configuration ---
INPUT_DATA_PATH = "processed_transcripts.json"
TRAIN_OUTPUT_PATH = "train_data.json"
TEST_OUTPUT_PATH = "test_data.json"

# --- MODIFICATION: Changed the ratio for the test set to 20% ---
TEST_SET_RATIO = 0.20

def main():
    print(f"Loading data from '{INPUT_DATA_PATH}'...")
    try:
        df = pd.read_json(INPUT_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{INPUT_DATA_PATH}'. Please run the preprocessing script first.")
        return

    # Find all unique video URLs (each one represents a lecture)
    all_lectures = df['video_url'].unique()
    num_all_lectures = len(all_lectures)

    if num_all_lectures < 2:
        print("Error: Fewer than 2 lectures found. Cannot create a train/test split.")
        return

    # Calculate the number of lectures for the test set based on the new ratio
    num_test_lectures = math.ceil(num_all_lectures * TEST_SET_RATIO)
    
    # Ensure there's at least one lecture in the training set
    if num_test_lectures >= num_all_lectures:
        num_test_lectures = num_all_lectures - 1 # Keep at least one for training

    # Using a fixed seed for reproducibility to always get the same split
    np.random.seed(42)
    np.random.shuffle(all_lectures)

    # Create train and test dataframes based on lecture URLs
    test_lectures = all_lectures[:num_test_lectures]
    train_lectures = all_lectures[num_test_lectures:]
    
    # Updated print statements to reflect the 70/30 split
    print(f"\nFound {num_all_lectures} total lectures.")
    print(f"Splitting data using an approximately {int((1-TEST_SET_RATIO)*100)}% / {int(TEST_SET_RATIO*100)}% ratio.")
    print(f"Training set: {len(train_lectures)} lectures.")
    print(f"Held-out test set: {len(test_lectures)} lectures.")
    
    print("\nLectures reserved for the TEST set:")
    for url in test_lectures:
        print(f"  - {url}")

    # Create the train and test dataframes
    train_df = df[df['video_url'].isin(train_lectures)]
    test_df = df[df['video_url'].isin(test_lectures)]

    # Save to new JSON files
    train_df.to_json(TRAIN_OUTPUT_PATH, orient='records', indent=2)
    print(f"\n✅ Saved {len(train_df)} training chunks from {len(train_lectures)} lectures to '{TRAIN_OUTPUT_PATH}'")
    
    test_df.to_json(TEST_OUTPUT_PATH, orient='records', indent=2)
    print(f"✅ Saved {len(test_df)} testing chunks from {len(test_lectures)} lectures to '{TEST_OUTPUT_PATH}'")

if __name__ == "__main__":
    main()