import asyncio
import re

from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document  # Import LangChain Document

# FAISS-based data and interaction operations
from app.rag_pipeline.handlers.generic.interaction_utils import (
    initialize_vector_database,
    document_retrieval,
)
from app.llms.connectivity import OPENAI_API_KEY

# ---------------------------
# Custom Sanitizing Extractor
# ---------------------------
from langchain.retrievers.document_compressors import LLMChainExtractor

class SanitizingLLMChainExtractor(LLMChainExtractor):
    async def acompress_documents(self, docs, query, **kwargs):
        """Compress each document individually, sanitize the output, and return a list of Documents."""
        sanitized = []
        for i, doc in enumerate(docs):
            # Call the LLM chain for each document individually.
            output = await self.llm_chain.apredict(context=doc.page_content, question=query, **kwargs)
            # Debug: Print raw output for this document.
            print(f"DEBUG: Raw output for doc {i}: {output}")
            # Sanitize the output: if it's a dict, try to extract common keys; otherwise, convert to string.
            if isinstance(output, dict):
                text = output.get("text") or output.get("context") or output.get("answer") or "Not available"
            elif not isinstance(output, str):
                text = str(output)
            else:
                text = output
            # If the text is just "Not available" and the document contains the query terms,
            # use a snippet of the original document instead
            if text == "Not available" and any(term.lower() in doc.page_content.lower() for term in query.split()):
                # Extract a snippet around the query term
                snippet = doc.page_content
                for term in query.split():
                    if term.lower() in snippet.lower():
                        start_idx = max(0, snippet.lower().find(term.lower()) - 100)
                        end_idx = min(len(snippet), snippet.lower().find(term.lower()) + len(term) + 100)
                        snippet = snippet[start_idx:end_idx]
                        break
                text = f"Relevant snippet: {snippet}"
            print(f"DEBUG: Sanitized output for doc {i}: {text}")
            sanitized.append(Document(page_content=text, metadata=doc.metadata))
        return sanitized


# ---------------------------
# Initialize Compression Retriever with the Custom Extractor
# ---------------------------
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import BaseRetriever
from pydantic import Field

class FAISSRetriever(BaseRetriever):
    """A custom FAISS retriever compatible with LangChain's ContextualCompressionRetriever."""
    vector_database: dict = Field(...)
    k: int = 5  # default number of documents to retrieve
    similarity_threshold: float = None  # set to None to disable filtering

    def _get_relevant_documents(self, query: str) -> list:
        return document_retrieval(query, self.vector_database, k=self.k)#, similarity_threshold=self.similarity_threshold

def initialize_compression_retriever(vector_database):
    """Initializes ContextualCompressionRetriever using the custom SanitizingLLMChainExtractor."""
    # Custom prompt for compression
    template = (
        "Extract any information about medications from the context that is relevant to answering the question. "
        "Include the medication name AND any relevant context about why it was started, stopped, or modified. "
        "Use exact phrases from the context. If no relevant medication information is found, reply: 'Not available'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    extraction_prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
    # Create the LLMChain manually
    llm_chain = LLMChain(llm=llm, prompt=extraction_prompt)
    # Use our custom extractor
    compressor = SanitizingLLMChainExtractor(llm_chain=llm_chain)
    base_retriever = FAISSRetriever(vector_database=vector_database, k=5)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

# ---------------------------
# Main QA Function
# ---------------------------
vector_database = None
compression_retriever = None

async def load_and_initialize_vector_database():
    global vector_database
    vector_database = initialize_vector_database()

async def question_answer(query, chat_history=None, chain_type="stuff"):
    """Answers a question based on the content of documents and chat history."""
    await load_and_initialize_vector_database()
    global vector_database

    # Retrieve documents from FAISS
    retrieved_docs = document_retrieval(query, vector_database)
    
    global compression_retriever
    if not compression_retriever:
        compression_retriever = initialize_compression_retriever(vector_database)

    template = (
        "You are a highly reliable medical assistant designed to answer clinical questions using retrieval-augmented generation. "
        "Use the provided clinical context to generate an accurate, factual, and concise answer to the question. "
        "Your answer should rely only on the information available in the context. "
        "For cardiac care specifically, pay attention to heart function, cardiac medications, EKG findings, "
        "cardiac procedures, and cardiovascular risk factors. "
        "Be extremely precise about medication dosages, lab values, and measurements. "
        "If multiple values or dosages are mentioned, clearly state which one is the most recent or final value. "
        "If the context doesn't contain enough information to answer the question completely, acknowledge this limitation. "
        "If the information seems contradictory, explain the discrepancy.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (3-5 sentences max):"
    )
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if chat_history:
        for message in chat_history:
            if message["role"] == "user":
                memory.chat_memory.add_user_message(message["content"])
            elif message["role"] == "assistant":
                memory.chat_memory.add_ai_message(message["content"])

    if chat_history:
        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY),
            chain_type=chain_type,
            retriever=compression_retriever,
            memory=memory,
            chain_type_kwargs={'prompt': QA_CHAIN_PROMPT},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY),
            chain_type=chain_type,
            retriever=compression_retriever,
            chain_type_kwargs={'prompt': QA_CHAIN_PROMPT},
            return_source_documents=True,
            verbose=True
        )

    # Manually compress the retrieved documents using the base compressor
    base_compressor = compression_retriever.base_compressor
    compressed_docs = await base_compressor.acompress_documents(retrieved_docs, query=query)
    
    # Debug: Print sanitized outputs
    for idx, doc in enumerate(compressed_docs):
        print(f"DEBUG: Compressed Document {idx+1}: {doc.page_content}")

    # Extract the page_content from each Document object and combine them into a single context
    context_texts = [doc.page_content for doc in compressed_docs if doc.page_content != "Not available"]
    combined_context = "\n".join(context_texts)
    
    # Pass sanitized compressed docs to the QA chain
    response = await qa.ainvoke({"query": query, "context": combined_context})

    result = response.get("result")
    source_documents = response.get("source_documents", [])
    return {"result": result, "source_documents": source_documents}

# async def main():
#     query = "What is the dosage of furosemide prescribed on discharge?"
#     result_dict = await question_answer(query)
#     print("\n--- Final QA Response ---")
#     print(result_dict["result"])
#     print("\n--- Source Documents ---")
#     for doc in result_dict["source_documents"]:
#         snippet = re.sub(r'\s+', ' ', doc.page_content.strip())[:300]
#         print(snippet, "\n")

# if __name__ == "__main__":
#     asyncio.run(main())