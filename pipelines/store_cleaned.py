import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def store_cleaned(items: List[Dict[str, Any]], output_dir: str, base_filename: str):
    Path(output_dir).mkdir(exist_ok=True)
    json_path = Path(output_dir) / f"{base_filename}.json"
    # Save JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(items)} items to {json_path}")

if __name__ == "__main__":
    import sys
    import json
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "cleaned_data"
    base_filename = sys.argv[3] if len(sys.argv) > 3 else "cleaned_output"
    with open(input_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    store_cleaned(items, output_dir, base_filename) 