# import os
# import pickle
# import json

# def examine_vector_database():
#     """Examine the contents of the vector database."""
#     base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#     metadata_path = os.path.join(base_path, "vector_db", "faiss_index_metadata.pkl")
    
#     if not os.path.exists(metadata_path):
#         print(f"Metadata file not found at {metadata_path}")
#         return
    
#     with open(metadata_path, 'rb') as f:
#         metadata = pickle.load(f)
    
#     print(f"Loaded {len(metadata)} documents from {metadata_path}")
    
#     # Print the first 5 documents
#     print("\nFirst 5 documents:")
#     for i, doc in enumerate(metadata[:5]):
#         content = doc.get("content", "")
#         print(f"\nDoc {i}:")
#         print(f"Content: {content[:300]}...")
#         print(f"Source: {doc.get('source', 'Unknown')}")
#         print(f"Page: {doc.get('page', 'Unknown')}")
#         print(f"Chunk: {doc.get('chunk', 'Unknown')}")
    
#     # Check for specific terms
#     terms = ["meld", "score", "potassium", "k+", "k level", "fluid", "ascites", "paracentesis"]
#     print("\nChecking for specific terms:")
#     for term in terms:
#         count = sum(1 for doc in metadata if term.lower() in doc.get("content", "").lower())
#         print(f"Term '{term}' found in {count} documents")
    
#     # Save a sample of documents to a JSON file for easier examination
#     sample_path = os.path.join(base_path, "vector_db", "sample_documents.json")
#     with open(sample_path, 'w') as f:
#         json.dump(metadata[:10], f, indent=2)
#     print(f"\nSaved sample documents to {sample_path}")

# if __name__ == "__main__":
#     examine_vector_database() 








# import numpy as np
# from sentence_transformers import SentenceTransformer
# from numpy.linalg import norm
# from tqdm import tqdm

# from app.rag_pipeline.interaction_handler.interaction_tools import document_retrieval

# # Load your embedding model (same used for indexing)
# embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# # Cosine similarity function
# def cosine_sim(a, b):
#     return np.dot(a, b) / (norm(a) * norm(b))

# # Embed a string using BGE with prompt
# def embed_text(text):
#     prompt = f"Represent this sentence for retrieval: {text}"
#     return embedding_model.encode([prompt], normalize_embeddings=True)[0]

# # Main semantic accuracy evaluation
# def compute_semantic_retrieval_accuracy(data, vector_database, similarity_threshold=0.8, k_values=[1, 3, 5]):
#     topk_hits = {k: 0 for k in k_values}
#     total = len(data)

#     for item in tqdm(data, desc="Evaluating semantic retrieval"):
#         query = item['query']
#         true_answer = item['original_answer']

#         answer_embedding = embed_text(true_answer)

#         for k in k_values:
#             retrieved_docs = document_retrieval(query, vector_database, k=k)

#             found = False
#             for chunk in retrieved_docs:
#                 chunk_embedding = embed_text(chunk.page_content)
#                 similarity = cosine_sim(answer_embedding, chunk_embedding)
#                 if similarity >= similarity_threshold:
#                     topk_hits[k] += 1
#                     found = True
#                     break  # Count once per k
#             if found:
#                 break  # No need to continue for higher k if already matched

#     accuracies = {f"Top-{k} Semantic Accuracy": round(topk_hits[k] / total * 100, 2) for k in k_values}
#     return accuracies

