import sys
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

BATCH_SIZE = 64

if len(sys.argv) < 3:
    print("Usage: python embed_and_store.py <input_json> <collection_name>")
    sys.exit(1)

input_json = sys.argv[1]
collection_name = sys.argv[2]

# 1. Load data
with open(input_json, 'r') as f:
    data = json.load(f)

# 2. Load model
print("Loading embedding model (BAAI/bge-large-en)...")
model = SentenceTransformer('BAAI/bge-large-en')

# Ensure embedding dimension is valid
embedding_dim = model.get_sentence_embedding_dimension()
if embedding_dim is None:
    raise ValueError("Could not determine embedding dimension from the model. Please check the model and try again.")

# 3. Connect to Qdrant (local)
qdrant = QdrantClient("localhost", port=6333)

# 4. Create collection if not exists
print(f"Creating/recreating Qdrant collection '{collection_name}'...")
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=int(embedding_dim), distance=Distance.COSINE)
)

# 5. Prepare and upload points in batches
points = []
for i, chunk in enumerate(data):
    text = chunk['text']
    metadata = chunk['metadata']
    embedding = model.encode(text)
    points.append(
        PointStruct(
            id=metadata['id'],
            vector=embedding.tolist(),
            payload={**metadata, 'text': text}
        )
    )
    # Batch upload
    if len(points) >= BATCH_SIZE or i == len(data) - 1:
        qdrant.upsert(collection_name=collection_name, points=points)
        print(f"Uploaded {i+1} / {len(data)} points...")
        points = []

print(f"All points uploaded to Qdrant collection '{collection_name}'!") 