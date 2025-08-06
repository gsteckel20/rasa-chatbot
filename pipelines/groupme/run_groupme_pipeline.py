import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_and_clean_groupme import load_and_clean_groupme
from extract_metadata_groupme import extract_groupme_context_blocks
from store_cleaned import store_cleaned
from sentence_transformers import SentenceTransformer
import json

GROUPME_DIR = 'groupme_data'
OUTPUT_DIR = './cleaned_data'
GROUPME_BASENAME = 'groupme_cleaned'

def main():
    print("Loading embedding model (BAAI/bge-large-en)...")
    model = SentenceTransformer('BAAI/bge-large-en')
    groupme_cleaned = load_and_clean_groupme(GROUPME_DIR)
    # Flatten all messages from all chunks
    all_messages = []
    for chunk in groupme_cleaned:
        raw = chunk.get('raw', [])
        if isinstance(raw, list):
            all_messages.extend(raw)
        elif isinstance(raw, dict):
            all_messages.append(raw)
    groupme_meta = extract_groupme_context_blocks(all_messages, model)
    store_cleaned(groupme_meta, OUTPUT_DIR, GROUPME_BASENAME)

if __name__ == "__main__":
    main() 