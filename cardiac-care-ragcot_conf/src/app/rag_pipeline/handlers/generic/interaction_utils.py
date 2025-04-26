import faiss
import pickle
import numpy as np
import re

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain_openai import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import BaseRetriever
from pydantic import Field

# Import OpenAI API Key
from app.llms.connectivity import OPENAI_API_KEY

# Initialize the embedding model (consider experimenting with a stronger model if available)
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# ---------------------------
# Module: Vector Database Initialization (using FAISS)
# ---------------------------
def initialize_vector_database():
    """Loads the FAISS index and document metadata from disk."""
    # Use absolute paths based on the workspace root
    # workspace_root = "/home/harry/Desktop/research/app"
    # faiss_index_path = os.path.join(workspace_root, "src/app/vector_db/faiss_index.index")
    # metadata_path = os.path.join(workspace_root, "src/app/vector_db/faiss_index_metadata.pkl")
    
    faiss_index_path = "../../../vector_stores/base_conference/faiss_index.index"
    metadata_path = "../../../vector_stores/base_conference/faiss_index_metadata.pkl"
    
    # Load FAISS index
    loaded_index = faiss.read_index(faiss_index_path)
    print(f"Loaded FAISS index from {faiss_index_path}")

    # Load document metadata
    with open(metadata_path, "rb") as f:
        loaded_documents = pickle.load(f)
    print(f"Loaded {len(loaded_documents)} documents from {metadata_path}")

    # Convert metadata into LangChain Document objects
    documents = [Document(page_content=doc["chunk_text"], metadata=doc) for doc in loaded_documents]
    return {"index": loaded_index, "documents": documents}

# ---------------------------
# Module: Document Retrieval (with optional similarity threshold)
# ---------------------------
def document_retrieval(query, vector_database, k=5, similarity_threshold=None):
    """Retrieves relevant documents from the FAISS vector store with optional filtering based on similarity."""
    index = vector_database["index"]
    documents = vector_database["documents"]

    # Generate the query embedding and normalize
    query_embedding = embedding_model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # Perform similarity search in the FAISS index
    distances, indices = index.search(query_embedding, k)
    print("\nQuery:", query)
    print("Top similar document indices:", indices)
    print("Distances:", distances)

    # Optionally filter out documents with distance above a threshold (if provided)
    retrieved_docs = []
    for i, idx in enumerate(indices[0]):
        if similarity_threshold is None or distances[0][i] <= similarity_threshold:
            retrieved_docs.append(documents[idx])
    
    return retrieved_docs

# ---------------------------
# Custom FAISS Retriever for LangChain Compatibility
# ---------------------------
class FAISSRetriever(BaseRetriever):
    """A custom FAISS retriever compatible with LangChain's ContextualCompressionRetriever."""
    vector_database: dict = Field(...)
    k: int = 5  # default number of documents to retrieve
    similarity_threshold: float = None  # e.g., 0.35; set to None to disable

    def _get_relevant_documents(self, query: str) -> list:
        """Uses FAISS to retrieve relevant documents."""
        return document_retrieval(query, self.vector_database, k=self.k, similarity_threshold=self.similarity_threshold)

# ---------------------------
# Module: Compression Retriever Initialization
# ---------------------------


def initialize_compression_retriever(vector_database):
    """Initializes ContextualCompressionRetriever with a custom prompt using a manually constructed LLMChain."""

    # Comprehensive medical information extraction prompt with focus on cardiac care
    template = (
        "Extract ALL relevant medical information from the context that could help answer the question. "
        "Focus on clinical facts, diagnoses, symptoms, medications, procedures, lab results, vital signs, "
        "and other medical information. For cardiac care specifically, pay attention to heart function, "
        "cardiac medications, EKG findings, cardiac procedures, and cardiovascular risk factors. "
        "Include specific details like dosages, dates, measurements, and reasons for changes. "
        "Extract complete sentences or phrases that contain relevant information. "
        "If the context contains information that might be indirectly related to the question, include it as well. "
        "If no relevant information is found, reply: 'Not available'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    # Build a LangChain prompt template
    extraction_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # Create the LLM and LLMChain manually
    llm = OpenAI(api_key=OPENAI_API_KEY)
    llm_chain = LLMChain(llm=llm, prompt=extraction_prompt)

    # Pass it to the extractor
    compressor = LLMChainExtractor(llm_chain=llm_chain)

    # Use your FAISS retriever with filtering
    base_retriever = FAISSRetriever(vector_database=vector_database, k=5, similarity_threshold=0.35)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

# ---------------------------
# Module: Retrieval & Compression Debugging
# ---------------------------
def retrieve_and_compress_documents(query, vector_database, compression_retriever):
    """Retrieves and compresses documents manually to avoid LangChain filtering bugs."""

    # Manually retrieve using your own FAISS wrapper
    retrieved_documents = document_retrieval(query, vector_database, k=5, similarity_threshold=0.5)

    print("\n--- Retrieved Documents ---")
    if not retrieved_documents:
        print("No documents were retrieved for this query.")
        return [], []

    for i, doc in enumerate(retrieved_documents):
        snippet = re.sub(r'\s+', ' ', doc.page_content.strip())[:200]
        print(f"Document {i+1} metadata: {doc.metadata}")
        print(f"Document {i+1} snippet: {snippet}...\n")

    # Manually run the base compressor
    try:
        compressed_docs = compression_retriever.base_compressor.compress_documents(
            retrieved_documents, query=query
        )
    except Exception as e:
        print("Compression failed:", e)
        return [], retrieved_documents

    if not compressed_docs:
        print("\nNo compressed documents returned — model may have found nothing relevant.")
    else:
        pretty_print_docs(compressed_docs)

    return compressed_docs, retrieved_documents

# ---------------------------
# Helper: Pretty Print Compressed Documents
# ---------------------------
def pretty_print_docs(docs):
    """Prints compressed document texts in a readable format."""
    print("\n" + "-" * 100)
    for i, doc in enumerate(docs):
        snippet = re.sub(r'\s+', ' ', doc.page_content)[:200]
        print(f"Compressed Document {i+1}:\n{snippet}...\n")
        print("-" * 100)
        
# ---------------------------
# Main Execution
# ---------------------------
# if __name__ == "__main__":
#     # Load the FAISS index and metadata
#     vector_database = initialize_vector_database()

#     # Define your query (you may update this as needed)
#     query = "What is the patient's diagnosis on discharge?"

#     # Retrieve documents using FAISS (with k=5 and threshold filtering)
#     # retrieved_documents = document_retrieval(query, vector_database, k=5)
    
#     # print("\n--- Retrieved Documents Debug Print ---")
#     # for i, doc in enumerate(retrieved_documents):
#     #     snippet = re.sub(r'\s+', ' ', doc.page_content)[:200]
#     #     print(f"Document {i+1}:\n{snippet}...\n")

#     # Initialize the compression retriever with the custom extraction prompt
#     compression_retriever = initialize_compression_retriever(vector_database)

#     # Retrieve and compress documents (with debug prints)
#     compressed_docs, _ = retrieve_and_compress_documents(query, vector_database, compression_retriever)

#     print("\n--- Final Compressed Documents Debug Print ---")
#     for i, comp in enumerate(compressed_docs):
#         snippet = re.sub(r'\s+', ' ', comp.page_content)[:200]
#         print(f"Compressed Document {i+1}:\n{snippet}...\n")
