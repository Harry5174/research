import asyncio
import re

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

from app.rag_pipeline.handlers.generic.interaction_utils import (
    initialize_vector_database,
    document_retrieval,
)
from app.llms.connectivity import GEMINI_API_KEY

# from langchain.retrievers.document_compressors import LLMChainExtractor
# from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import BaseRetriever
from pydantic import Field


class FAISSRetriever(BaseRetriever):
    vector_database: dict = Field(...)
    k: int = 5

    def _get_relevant_documents(self, query: str) -> list:
        return document_retrieval(query, self.vector_database, k=self.k)

# ---------------------------
# Enhanced Document Processor (map-style)
# ---------------------------
async def process_docs_individually(llm, prompt_template, query, docs):
    answers = []
    for i, doc in enumerate(docs):
        single_prompt = prompt_template.format(context=doc.page_content, question=query)
        output = await llm.apredict(single_prompt)
        print(f"DEBUG: Answer for doc {i}: {output}")
        answers.append(Document(page_content=output, metadata=doc.metadata))
    return answers

# ---------------------------
# Prompt Template
# ---------------------------
QA_TEMPLATE = (
    "You are a reliable medical assistant answering clinical questions using context below.\n"
    "Use ONLY information from the context. Be accurate and specific about clinical details.\n"
    "If no relevant info is found, say 'Not available'.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer (max 3 sentences):"
)
QA_CHAIN_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)

# ---------------------------
# Main QA Function
# ---------------------------
vector_database = None

async def load_and_initialize_vector_database():
    global vector_database
    vector_database = initialize_vector_database()

async def question_answer(query, chat_history=None):
    await load_and_initialize_vector_database()
    global vector_database

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    retrieved_docs = document_retrieval(query, vector_database)
    processed_docs = await process_docs_individually(llm, QA_TEMPLATE, query, retrieved_docs)

    # Filter out 'Not available' docs
    def is_valid_answer(text):
        text = text.strip().lower()
        # Define patterns that are not useful
        bad_phrases = [
            "not available",
            "not mentioned",
            "no relevant information",
            "no information found",
            "not found",
            "context not mentioned"
        ]
        return all(phrase not in text for phrase in bad_phrases)

    filtered_docs = [doc for doc in processed_docs if is_valid_answer(doc.page_content)]
    
    if not filtered_docs:
        return {
            "result": "Not available",
            "source_documents": []
        }

    final_context = "\n".join([doc.page_content for doc in filtered_docs])
    final_prompt = QA_CHAIN_PROMPT.format(context=final_context, question=query)
    final_answer = await llm.apredict(final_prompt)

    return {
        "result": final_answer,
        "source_documents": filtered_docs
    }


# ---------------------------
# Entry Point
# ---------------------------
# path="../../../data/evaluation/eval_data/eval_data_r50.json"
# async def main():
#     query = "What is the MELD score of the patient on admission?"
#     result_dict = await question_answer(query)
#     print("\n--- Final QA Response ---")
#     print(result_dict["result"])
#     print("\n--- Source Documents ---")
#     for doc in result_dict["source_documents"]:
#         snippet = re.sub(r'\s+', ' ', doc.page_content.strip())[:300]
#         print(snippet, "\n")

# if __name__ == "__main__":
#     asyncio.run(main())

