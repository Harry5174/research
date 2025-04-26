import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "../../data/processed/json/processed_data_50r.json"
OUTPUT_DIR = "../../vector_stores/base_conference"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_index.index")
METADATA_PATH = os.path.join(OUTPUT_DIR, "faiss_index_metadata.pkl")

# -------------------------------
# Load BGE Embedding Model
# -------------------------------
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed_texts(texts):
    prompts = [f"Represent this sentence for retrieval: {t}" for t in texts]
    return embedding_model.encode(prompts, normalize_embeddings=True)

# -------------------------------
# Build and Save FAISS Index
# -------------------------------
def build_and_save_index(docs):
    texts = [doc["chunk_text"] for doc in docs if doc.get("chunk_text", "").strip()]
    embeddings = embed_texts(texts)

    if len(embeddings) == 0:
        raise ValueError("No valid embeddings. Check your data or chunk_text field.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"FAISS index saved to {INDEX_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} pre-chunked documents from JSON.")
    build_and_save_index(data)


