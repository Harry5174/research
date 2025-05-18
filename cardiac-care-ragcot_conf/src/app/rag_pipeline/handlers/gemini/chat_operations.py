import asyncio
import re
from typing import Optional 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
try:
    from app.rag_pipeline.handlers.generic.interaction_utils import (
        initialize_vector_database,
        initialize_compression_retriever    
        )
    from app.llms.connectivity import GEMINI_API_KEY, OPENAI_API_KEY 
except ImportError:
    print("Warning: Using mock implementations for app-specific imports.")
    # GEMINI_API_KEY = "GEMINI_API_KEY"
    # OPENAI_API_KEY = "OPENAI_API_KEY" 

    def initialize_vector_database():
        print("Mock: Initializing vector database...")
        return {"documents": [Document(page_content="mock doc", metadata={})]} # Simplified mock

    # Mock for initialize_compression_retriever
    from langchain.schema import BaseRetriever as LangchainBaseRetriever 
    class MockCompressionRetriever(LangchainBaseRetriever):
        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
            print(f"Mock Compression Retriever: Getting docs for '{query}'")
            return [
                Document(page_content=f"Mock compressed golden chunk 1 for '{query}'.", metadata={"source": "mock_source_1.pdf"}),
                Document(page_content=f"Mock compressed golden chunk 2 for '{query}'.", metadata={"source": "mock_source_2.pdf"})
            ]
        def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
            print(f"Mock Compression Retriever: Getting docs for '{query}'")
            return self._aget_relevant_documents(query) 

    def initialize_compression_retriever(vector_database, base_retriever_k, base_retriever_threshold):
        print(f"Mock: Initializing compression retriever (k={base_retriever_k}, threshold={base_retriever_threshold})")
        return MockCompressionRetriever()

# ---------------------------
# Prompt Template (remains the same)
# ---------------------------
QA_TEMPLATE_STR = (
    "You are a reliable medical assistant answering clinical questions using context below.\n"
    "Use ONLY information from the context. Be accurate and specific about clinical details.\n"
    "If no relevant info is found, say 'Not available'.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer (max 3 sentences):"
)
QA_CHAIN_PROMPT = PromptTemplate.from_template(QA_TEMPLATE_STR)

# ---------------------------
# Main QA Function (Modified to use Compression Retriever)
# ---------------------------
vector_database_global = None

async def load_and_initialize_vector_database():
    global vector_database_global
    if vector_database_global is None:
        vector_database_global = initialize_vector_database()

async def Youtube_streaming(query: str, chat_history=None): # Renamed function in your example
    await load_and_initialize_vector_database()
    global vector_database_global

    # LLM for final answer generation (Gemini)
    llm_final_answer = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0,
        google_api_key=GEMINI_API_KEY,
        streaming=True
    )

    print("Initializing compression retriever...")
    compression_retriever = initialize_compression_retriever(
        vector_database_global,
        base_retriever_k=20,       
        base_retriever_threshold=None 
    )

    # Get "golden chunks" using the compression retriever.
    # This now involves FAISS retrieval + LLM-based compression/extraction using OpenAI.
    print(f"\nDEBUG: Retrieving and compressing documents for query: '{query}'...")
    
    # Ensure the OpenAI API key is available for the compressor if it's using an OpenAI model
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY" or OPENAI_API_KEY == "":
        # This check is basic; a more robust check might be needed depending on how your keys are managed.
        print("Warning: OPENAI_API_KEY may not be configured for the document compressor. Compression might fail or use a different model.")

    try:
        # ContextualCompressionRetriever has an `aget_relevant_documents` method
        retrieved_and_compressed_docs = await compression_retriever.aget_relevant_documents(query)
    except Exception as e:
        print(f"Error during compression retrieval: {e}")
        import traceback
        traceback.print_exc()
        yield {"type": "sources", "data": []}
        yield {"type": "result_chunk", "data": f"An error occurred while retrieving documents: {str(e)}"}
        yield {"type": "end_of_stream", "data": None}
        return

    print(f"DEBUG: Retrieved {len(retrieved_and_compressed_docs)} compressed documents ('golden chunks').")

    # Filter out any "No direct answer found in context" responses from the compressor itself.
    def is_valid_compressed_content(text: str) -> bool:
        text_lower = text.strip().lower()
        if "no direct answer found in context" in text_lower:
            return False
        return True

    final_context_docs = [doc for doc in retrieved_and_compressed_docs if is_valid_compressed_content(doc.page_content)]
    
    yield {"type": "sources", "data": final_context_docs}

    if not final_context_docs:
        yield {"type": "result_chunk", "data": "No relevant information was found after advanced filtering and compression."}
        yield {"type": "end_of_stream", "data": None}
        return

    # Constructing the final context from the "golden chunks"
    # The page_content of these docs IS the extracted relevant information.
    context_parts = []
    for i, doc in enumerate(final_context_docs):
        source_info = doc.metadata.get('source', f'Source {i+1}')
        context_parts.append(f"Relevant Information from {source_info}:\n{doc.page_content}")
    final_context_str = "\n\n---\n\n".join(context_parts)
    
    # Format the final prompt for the Gemini LLM
    final_prompt_for_llm = QA_CHAIN_PROMPT.format(context=final_context_str, question=query)
    
    print("\nDEBUG: Final prompt for Gemini (first 300 chars):\n" + final_prompt_for_llm[:300] + "...")
    print("\n--- Streaming Final Answer from Gemini ---")

    # Stream the final answer from Gemini
    async for chunk in llm_final_answer.astream(final_prompt_for_llm):
        if hasattr(chunk, 'content'):
            yield {"type": "result_chunk", "data": chunk.content}
        else: 
            yield {"type": "result_chunk", "data": str(chunk)}
            
    yield {"type": "end_of_stream", "data": None}

