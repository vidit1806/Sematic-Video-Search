# Import Libraries - The toolbox for text processing and natural language understanding
import os  # File system operations - checking if files exist, creating paths
import json  # JSON handling - our output format for structured data
import re  # Regular expressions - powerful pattern matching for cleaning text
from datetime import timedelta  # Time formatting - convert seconds to HH:MM:SS format
import pandas as pd  # Data manipulation - Excel-like operations for our metadata
import nltk  # Natural Language Toolkit - the Swiss Army knife of text processing
from nltk.corpus import stopwords  # Common words like "the", "and" that don't add meaning
from nltk.stem import WordNetLemmatizer  # Converts words to their base form (running → run)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Smart text chunking
import string  # Built-in string operations - punctuation removal
import spacy  # Industrial-strength NLP - named entity recognition and more
from nltk import pos_tag  # Part-of-speech tagging - identify nouns, verbs, etc.
from nltk.corpus import wordnet  # WordNet database - for better lemmatization
import logging  # Progress tracking and error reporting

# NLTK Data Download -Ensuring we have all the language models we need

def download_nltk_data():
    """Downloads all required NLTK packages, skipping if they already exist - like updating your apps"""
    packages = [
        'punkt',  # Sentence tokenizer - splits text into sentences intelligently
        'averaged_perceptron_tagger',  # POS tagger - identifies grammatical roles
        'averaged_perceptron_tagger_eng',  # English-specific POS tagger
        'stopwords',  # List of common words to filter out
        'wordnet'  # Lexical database for word relationships and meanings
    ]
    
    for package in packages:
        try:
            # Try to find the package - different packages live in different NLTK folders
            if package == 'punkt':
                nltk.data.find(f'tokenizers/{package}')  # Tokenizers folder
            elif 'tagger' in package:
                nltk.data.find(f'taggers/{package}')  # POS taggers folder
            else:
                nltk.data.find(f'corpora/{package}')  # Language corpora folder
        except LookupError:  # Package not found
            print(f'Downloading NLTK package: {package}...')
            nltk.download(package, quiet=True)  # Download silently

print("Checking for NLTK data...")  # Let user know we're preparing
download_nltk_data()  # Actually run the check/download
print("NLTK check complete.")  # Confirmation message


# Data Preprocessing and Document Segmentation
lemmatizer = WordNetLemmatizer()  # Tool to reduce words to base forms
stop_words = set(stopwords.words("english"))  # Set of English stopwords (faster lookup than list)
TRANSLATOR = str.maketrans('', '', string.punctuation)  # Translation table to remove all punctuation
SOURCE_PATTERN = re.compile(r"\[source: \d+\]")  # Regex to find citation markers like [source: 1]
NOTICE_PATTERN = re.compile(r"\[Auto-generated transcript\. Edits may have been applied for clarity\.\]")  # Remove auto-generation notices
BRACKETS_PATTERN = re.compile(r"\[.*?\]|\(.*?\)")  # Remove content in brackets/parentheses - often metadata
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")  # Setup progress logging
nlp = spacy.load("en_core_web_trf")  # Load spaCy's transformer model - state-of-the-art NLP

# File paths - where to find input data and save output
METADATA_PATH = "video_metadata.csv.xlsx"  # Excel file with video information
TRANSCRIPT_FOLDER = "Transcripts"  # Folder containing raw transcript files
OUTPUT_FILE = "processed_transcripts.json"  # Where to save our processed chunks

# Helper Functions -The specialized tools for the text processing pipeline

def format_time(seconds: float) -> str:
    """Formats seconds into a HH:MM:SS string - makes timestamps human-readable"""
    return str(timedelta(seconds=int(seconds)))  # Convert to timedelta, then string

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts - helps lemmatizer work better"""
    tag = pos_tag([word])[0][1][0].upper()  # Get POS tag for word, take first letter
    tag_dict = {  # Map NLTK POS tags to WordNet format
        "J": wordnet.ADJ,    # Adjective
        "N": wordnet.NOUN,   # Noun
        "V": wordnet.VERB,   # Verb  
        "R": wordnet.ADV     # Adverb
    }
    return tag_dict.get(tag, wordnet.NOUN)  # Default to noun if unknown

def clean_subtitle_text(text: str) -> str:
    """Cleans and processes raw transcript text while preserving named entities - the heart of our preprocessing"""
    
    # Step 1: Remove unwanted patterns from transcripts
    text = SOURCE_PATTERN.sub("", text)  # Remove [source: X] citations
    text = NOTICE_PATTERN.sub("", text)  # Remove auto-generation notices
    text = text.lower()  # Convert to lowercase for consistency
    text = BRACKETS_PATTERN.sub("", text)  # Remove bracketed content
    text = text.translate(TRANSLATOR)  # Remove all punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace - multiple spaces become one

    # Step 2: Use spaCy to identify important named entities (people, places, organizations)
    doc = nlp(text)  # Process text with advanced NLP model en_core_web_trf
    entities = {ent.text.lower() for ent in doc.ents}  # Extract named entities and lowercase them

    # Step 3: Process each word intelligently
    tokens = text.split()  # Split into individual words
    cleaned_tokens = []  # Our cleaned word list
    
    for t in tokens:
        if t in entities:  # If it's a named entity (person, place, etc.)
            cleaned_tokens.append(t)  # Keep as these are important
        elif t not in stop_words:  # If it's not a common stopword
            # Lemmatize with proper POS tag - reduces to base form (better → good)
            cleaned_tokens.append(lemmatizer.lemmatize(t, get_wordnet_pos(t)))
        # Skip stopwords entirely - they don't add semantic meaning

    return " ".join(cleaned_tokens)  # Rejoin into clean text

# Main Execution

def main():
    """Loads metadata, processes and chunks transcripts, and saves to JSON - the main workflow"""
    
    # Step 1: Load video metadata from Excel file
    logging.info("Loading video metadata...")
    try:
        metadata_df = pd.read_excel(METADATA_PATH)  # Read Excel file into pandas DataFrame
    except FileNotFoundError:  # Handle missing file gracefully
        logging.error(f"Metadata file not found at '{METADATA_PATH}'")
        return  # Can't continue without metadata
    except Exception as e:  # Handle other Excel reading errors
        logging.error(f"Error reading metadata file: {e}")
        return

    # Step 2: Initialize text splitter - breaks long transcripts into manageable chunks
    
    text_splitter = RecursiveCharacterTextSplitter( # by langchain library. To achieve better semantic coherence in chunks
        chunk_size= 500,  # Target chunk size in characters - good for search
        chunk_overlap= 50  # Overlap between chunks - prevents splitting important concepts
    )
    
    # Initialize tracking variables
    all_chunks = []  # Will hold all processed text chunks
    chunk_counter = 0  # Unique ID counter for each chunk
    logging.info(f"Processing transcripts from '{TRANSCRIPT_FOLDER}'...")

    # Step 3: Process each video in our metadata
    for _, row in metadata_df.iterrows():  # Iterate through each row in Excel
        video_url = row["video_url"]  # Where to watch this video
        duration = row["duration_in_seconds"]  # How long the video is

        # Validate duration data - must be a positive number
        if not isinstance(duration, (int, float)) or duration <= 0:
            logging.warning(f"Invalid duration '{duration}' for video URL '{video_url}'. Skipping this row.")
            continue  # Skip this video

        # Get and validate filename
        filename_raw = row["filename"]
        if pd.isna(filename_raw):  # Check for missing filename
            logging.warning(f"Missing filename in metadata for video URL '{video_url}'. Skipping this row.")
            continue

        filename = str(filename_raw).strip()  # Clean filename
        if not filename:  # Check for empty filename
            logging.warning(f"Empty filename in metadata for video URL '{video_url}'. Skipping this row.")
            continue

        # Check if transcript file actually exists
        filepath = os.path.join(TRANSCRIPT_FOLDER, filename)  # Build full file path
        if not os.path.exists(filepath):  # File missing
            logging.warning(f"Missing transcript file, skipping: {filepath}")
            continue

        # Step 4: Read and validate transcript content
        with open(filepath, "r", encoding="utf-8") as f:  # Open with proper encoding
            raw_text = f.read().strip()  # Read entire file and remove whitespace

        if not raw_text:  # Empty file check
            logging.warning(f"Empty transcript file, skipping: {filename}")
            continue

        # Quality check - detect overly noisy transcripts (too much punctuation = OCR errors)
        if len(raw_text) > 0:
            punct_ratio = sum(c in string.punctuation for c in raw_text) / len(raw_text)  # Calculate punctuation ratio
            if punct_ratio > 0.8:  # More than 80% punctuation = likely corrupted
                logging.warning(f"Noisy transcript (>{punct_ratio*100:.1f}% punctuation): {filename}")
                continue  # Skip corrupted transcripts

        # Step 5: Split transcript into chunks and process each one
        original_text_chunks = text_splitter.split_text(raw_text)  # Break into manageable pieces
        
        # Calculate timing information - estimate where each chunk occurs in video
        total_chars = len(raw_text)  # Total characters in transcript
        chars_per_second = total_chars / duration if duration > 0 else 25  # Estimate reading speed
        char_cursor = 0  # Track position in original text

        for original_chunk in original_text_chunks:  # Process each chunk
            # Clean the chunk text for search purposes
            cleaned_chunk = clean_subtitle_text(original_chunk)
            if not cleaned_chunk:  # Skip if cleaning removed everything
                continue

            # Calculate timing for this chunk - where does it appear in the video?
            chunk_start_char = raw_text.find(original_chunk, char_cursor)  # Find chunk position
            if chunk_start_char == -1:  # If not found 
                chunk_start_char = char_cursor  # Use current position

            # Convert character positions to time positions
            start_time_secs = chunk_start_char / chars_per_second  # When chunk starts
            chunk_duration_secs = len(original_chunk) / chars_per_second  # How long chunk lasts
            end_time_secs = start_time_secs + chunk_duration_secs  # When chunk ends

            # Ensure times are valid and within video bounds
            end_time_secs = min(end_time_secs, duration)  # Don't exceed video duration
            start_time_secs = max(min(start_time_secs, end_time_secs), 0)  # Keep positive and logical

            # Step 6: Avoid duplicate chunks (same content from different videos)
            if cleaned_chunk not in {c['text_cleaned'] for c in all_chunks}:  # Check for duplicates
                # Create chunk record with all metadata
                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_counter}",  # Unique identifier
                    "video_url": video_url,  # Link to watch this segment
                    "start": format_time(start_time_secs),  # When segment starts (HH:MM:SS)
                    "end": format_time(end_time_secs),  # When segment ends (HH:MM:SS)
                    "text_cleaned": cleaned_chunk,  # Processed text for search
                    "text_original": original_chunk  # Original text for display
                })
                chunk_counter += 1  # Increment for next chunk

            # Move cursor forward for next chunk timing calculation
            char_cursor = chunk_start_char + len(original_chunk)

    # Step 7: Save all processed chunks to JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:  # Create output file
        json.dump(all_chunks, f, indent=2)  # Write JSON with nice formatting

    # Success message with statistics
    logging.info(f"✅ Success! Saved {len(all_chunks)} chunks to '{OUTPUT_FILE}'")

if __name__ == "__main__":  # Only run if script is executed directly (not imported)
    main()  # Start the main processing workflow