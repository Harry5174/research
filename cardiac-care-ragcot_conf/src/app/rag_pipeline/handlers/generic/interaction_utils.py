import faiss
import pickle
import numpy as np
import re
import asyncio 
from typing import Optional 

from pathlib import Path
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field 

from app.llms.connectivity import OPENAI_API_KEY

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.runnables import Runnable
from typing import Sequence, Optional 
from langchain_core.callbacks import Callbacks 


class LCELExtractorCompressor(BaseDocumentCompressor):
    """A document compressor that uses a Runnable (LCEL chain) to process content."""
    chain: Runnable # Expects a Runnable, e.g., prompt | llm | StrOutputParser

    class Config:
        arbitrary_types_allowed = True # Needed for Pydantic V2 with Runnable

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        compressed_docs = []
        inputs = [{"context": doc.page_content, "question": query} for doc in documents]
        extracted_contents = self.chain.batch(inputs, config={"callbacks": callbacks})
        
        for i, extracted_content in enumerate(extracted_contents):
            # Check if content is valid and not the "not found" phrase
            if isinstance(extracted_content, str) and \
               extracted_content.strip() != "" and \
               "no direct answer found in context" not in extracted_content.lower():
                # Preserve original metadata
                metadata = documents[i].metadata.copy()
                compressed_docs.append(Document(page_content=extracted_content, metadata=metadata))
        return compressed_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        compressed_docs = []
        inputs = [{"context": doc.page_content, "question": query} for doc in documents]
        
        extracted_contents = await self.chain.abatch(inputs, config={"callbacks": callbacks})
        
        for i, extracted_content in enumerate(extracted_contents):
            # Check if content is valid and not the "not found" phrase
            if isinstance(extracted_content, str) and \
               extracted_content.strip() != "" and \
               "no direct answer found in context" not in extracted_content.lower():
                # Preserve original metadata
                metadata = documents[i].metadata.copy()
                compressed_docs.append(Document(page_content=extracted_content, metadata=metadata))
        return compressed_docs

# Initialize the embedding model
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# ---------------------------
# Module: Vector Database Initialization (using FAISS)
# ---------------------------
def initialize_vector_database():
    here = Path(__file__).resolve()
    project_root = here
    for parent_dir in here.parents:
        if (parent_dir / ".git").exists() or \
           (parent_dir / "pyproject.toml").exists() or \
           (parent_dir / "src").is_dir() and (parent_dir / "src" / "app").is_dir():
            project_root = parent_dir
            break
    
    faiss_index_path = project_root / "src" / "app" / "vector_stores" / "base_conference" / "faiss_index.index"
    metadata_path    = project_root / "src" / "app" / "vector_stores" / "base_conference" / "faiss_index_metadata.pkl"

    if not faiss_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {faiss_index_path!r} (Project root: {project_root})")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path!r} (Project root: {project_root})")

    index = faiss.read_index(str(faiss_index_path))
    with open(metadata_path, "rb") as f:
        docs_meta = pickle.load(f)

    documents = []
    for m_data in docs_meta:
        content = m_data.pop("chunk_text", "") 
        documents.append(Document(page_content=content, metadata=m_data))
    
    return {"index": index, "documents": documents, "_docs_meta_list": docs_meta}

# ---------------------------
# Module: Document Retrieval
# ---------------------------
def document_retrieval(query: str, vector_database: dict, k: int = 15, similarity_threshold: Optional[float] = None) -> list[Document]: # Added Optional here too for consistency
    """
    Retrieves relevant documents from FAISS. Normalizes query embedding.
    Assumes FAISS distances are L2 or (1 - cosine_similarity) where lower is better.
    """
    index = vector_database["index"]
    db_documents = vector_database["documents"]

    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)
    
    retrieved_docs = []
    if indices.size == 0:
        return retrieved_docs

    for i, doc_idx in enumerate(indices[0]):
        if doc_idx == -1:
            continue
        
        if doc_idx < 0 or doc_idx >= len(db_documents):
            print(f"[Warning] Invalid document index {doc_idx} from FAISS search. Max index: {len(db_documents)-1}")
            continue

        current_distance = distances[0][i]
        if similarity_threshold is None or current_distance <= similarity_threshold:
            doc = db_documents[doc_idx]
            doc.metadata['retrieval_score'] = float(current_distance)
            doc.metadata['retrieval_score_type'] = 'distance (lower is better, e.g., L2 or 1-cosine_sim)'
            retrieved_docs.append(doc)
            
    return retrieved_docs

# ---------------------------
# Custom FAISS Retriever for LangChain Compatibility
# ---------------------------
class FAISSRetriever(BaseRetriever):
    vector_database: dict = Field(...)
    k: int = 15
    similarity_threshold: Optional[float] = None # Allow None explicitly

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return document_retrieval(query, self.vector_database, k=self.k, similarity_threshold=self.similarity_threshold)

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return await asyncio.to_thread(self._get_relevant_documents, query)


# ---------------------------
# Module: Compression Retriever Initialization
# ---------------------------
def initialize_compression_retriever(vector_database: dict,
                                     base_retriever_k: int = 20,
                                     base_retriever_threshold: Optional[float] = None):
    template = (
        "Given the context and the question, extract **only the sentences or phrases from the context that directly answer the question or provide essential supporting evidence**. "
        "Be concise and precise. Focus on specific clinical facts, diagnoses, test results, treatments, and patient outcomes directly relevant to the question. "
        "If numerical values, dates, or specific medical terminology are crucial for answering, include them. "
        "Avoid background information or details not directly addressing the question. "
        "If no directly relevant information to answer the question is found in the provided context, respond ONLY with the exact phrase: 'No direct answer found in context'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Extracted Information:"
    )
    extraction_prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    compressor_llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo-0125",
        temperature=0,
        max_tokens=250,
    )

    extraction_runnable = extraction_prompt | compressor_llm | StrOutputParser()
    
    # Use the new custom LCELExtractorCompressor
    compressor = LCELExtractorCompressor(chain=extraction_runnable)
    # --- END OF LCEL CHANGE ---

    base_retriever = FAISSRetriever(
        vector_database=vector_database,
        k=base_retriever_k,
        similarity_threshold=base_retriever_threshold
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    return compression_retriever

# ---------------------------
# Helper: Pretty Print Compressed Documents
# ---------------------------
def pretty_print_docs(docs):
    print("\n" + "-" * 100)
    for i, doc in enumerate(docs):
        snippet = re.sub(r'\s+', ' ', doc.page_content)[:300]
        print(f"Compressed Document {i+1} (Metadata: {doc.metadata}):\n{snippet}...\n")
    print("-" * 100)