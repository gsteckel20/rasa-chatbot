import sys
import json
import uuid
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.78
WINDOW_SIZE = 10


def is_question(text):
    # Simple heuristic: contains a question mark
    return isinstance(text, str) and '?' in text


def extract_groupme_context_blocks(messages, model, window_size=WINDOW_SIZE, threshold=THRESHOLD):
    print("Extracting context blocks from GroupMe messages...")
    blocks = []
    texts = [msg.get('text', '') for msg in messages]
    senders = [msg.get('name', 'Unknown') for msg in messages]
    timestamps = [msg.get('created_at', 0) for msg in messages]
    print(f"Embedding {len(texts)} messages (this may take a while)...")
    all_embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True
    )
    num_questions = sum(1 for msg in messages if is_question(msg.get('text', '')))
    processed = 0
    for i, msg in enumerate(messages):
        if is_question(msg.get('text', '')):
            processed += 1
            print(f"[Semantic Bridging] Processing question {processed}/{num_questions} at index {i}...")
            question_embedding = all_embeddings[i]
            question_time = timestamps[i]
            # Find candidate indices: next up to window_size messages within 24 hours
            candidate_idxs = []
            for j in range(i+1, min(i+1+window_size, len(messages))):
                if timestamps[j] - question_time <= 86400:  # 24 hours in seconds
                    candidate_idxs.append(j)
                else:
                    break  # Stop at first message outside 24h window
            if not candidate_idxs:
                print(f"  No candidate answers for question at index {i}, skipping.")
                continue
            candidate_embeddings = all_embeddings[candidate_idxs]
            sims = cosine_similarity([question_embedding], candidate_embeddings)[0]
            answer_idxs = [idx for idx, sim in zip(candidate_idxs, sims) if sim > threshold]
            print(f"  Found {len(answer_idxs)} semantically similar answers (threshold={threshold})")
            # Build the context block
            block_lines = [f"{senders[i]}: {texts[i]}"]
            for idx in answer_idxs:
                block_lines.append(f"{senders[idx]}: {texts[idx]}")
            block_text = "\n".join(block_lines)
            block_metadata = {
                "id": str(uuid.uuid4()),
                "question_idx": i,
                "answer_indices": answer_idxs,
                "question_text": texts[i],
                "group_id": msg.get('group_id'),
                "created_at": msg.get('created_at'),
                "platform": msg.get('platform', 'gm'),
            }
            blocks.append({"text": block_text, "metadata": block_metadata})
    return blocks


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_metadata_groupme.py <input_json> <output_json>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    with open(input_path, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    print("Loading embedding model (BAAI/bge-large-en)...")
    model = SentenceTransformer('BAAI/bge-large-en')
    blocks = extract_groupme_context_blocks(messages, model)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)
    print(f"Extracted {len(blocks)} context blocks and saved to {output_path}")


if __name__ == "__main__":
    main() 