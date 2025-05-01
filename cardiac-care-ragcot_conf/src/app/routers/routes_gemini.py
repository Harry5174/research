from fastapi import APIRouter
from pydantic import BaseModel
import re

from app.rag_pipeline.handlers.gemini.chat_operations import (
    question_answer,
)

class GeminiRequest(BaseModel):
    query: str

router = APIRouter()

@router.post("/gemini")
async def gemini_chat(request: GeminiRequest):
    """
    Chat with Gemini model.
    """
    result_dict = await question_answer(request.query)

    # (debug prints…)
    print("\n--- Final QA Response ---")
    print(result_dict["result"])
    print("\n--- Source Documents Snippets ---")
    for doc in result_dict["source_documents"]:
        snippet = re.sub(r"\s+", " ", doc.page_content.strip())[:300]
        print(snippet, "\n")

    # serialize to JSON
    docs = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in result_dict["source_documents"]
    ]

    return {"response": result_dict["result"], "source_documents": docs}
