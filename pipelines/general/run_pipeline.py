from load_and_clean import load_and_clean
from extract_metadata import extract_metadata
from store_cleaned import store_cleaned

INPUT_DIR = './scrapy/output'
OUTPUT_DIR = './cleaned_data'
BASE_FILENAME = 'uga_cleaned'

def main():
    # Step 1: Load and clean web data
    cleaned = load_and_clean(INPUT_DIR)
    # Step 2: Extract metadata
    meta = extract_metadata(cleaned)
    # Step 3: Store cleaned results
    store_cleaned(meta, OUTPUT_DIR, BASE_FILENAME)

if __name__ == "__main__":
    main()
