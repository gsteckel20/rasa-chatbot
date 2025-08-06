import sys
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchRequest
from groq import Groq

# Usage: python query_qdrant.py <collection_name> <question> [k]
if len(sys.argv) < 3:
    print("Usage: python query_qdrant.py <collection_name> <question> [k]")
    sys.exit(1)

collection_name = sys.argv[1]
question = sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

# Initialize Groq client
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    print("Error: GROQ_API_KEY environment variable not set")
    print("Please set it with: export GROQ_API_KEY='your-api-key-here'")
    sys.exit(1)

groq_client = Groq(api_key=groq_api_key)

def query_groq(prompt):
    """Send prompt to Groq Llama 3.1 8B and return the response"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=512
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error querying Groq: {e}")
        return None

# Load embedding model
print("Loading embedding model (BAAI/bge-large-en)...")
model = SentenceTransformer('BAAI/bge-large-en')

# Embed the query
query_vec = model.encode(question).tolist()

# Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

# Search Qdrant
search_result = qdrant.search(
    collection_name=collection_name,
    query_vector=query_vec,
    limit=k
)

# Build the prompt
print("\nAnswer the following question using the context below.\n")
print("Context:")
context_parts = []
for i, hit in enumerate(search_result):
    text = hit.payload.get('text', '[No text found]')
    score = hit.score if hasattr(hit, 'score') else None
    if score is not None:
        print(f"<top-{i+1} chunk> (cosine similarity: {score:.4f})\n{text}\n")
        context_parts.append(f"<top-{i+1} chunk> (cosine similarity: {score:.4f})\n{text}")
    else:
        print(f"<top-{i+1} chunk>\n{text}\n")
        context_parts.append(f"<top-{i+1} chunk>\n{text}")

context_text = "\n\n".join(context_parts)
prompt = f"Answer the following question using the context below.\n\nContext:\n{context_text}\n\nQuestion: {question}"

print(f"Question: {question}")
print("\n" + "="*50)
print("AI ANSWER:")
print("="*50)

# Send to Groq and get answer
answer = query_groq(prompt)
if answer:
    print(answer)
else:
    print("Failed to get answer from Groq API") 