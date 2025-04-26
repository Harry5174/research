import asyncio
import os
import json
import re
from app.rag_pipeline.handlers.gemini.chat_operations import question_answer

async def evaluate_and_store_responses():
    eval_dataset_path = os.path.join(os.path.dirname(__file__), "..", '..', "..", "data", "evaluation", "eval_data", "eval_data_r50.json")
    queries = []
    original_answers = []

    try:
        with open(eval_dataset_path, "r") as f:
            data = json.load(f)
            queries = [item["query"] for item in data]
            original_answers = [item["original_answer"] for item in data]
            print(f"Loaded {len(queries)} queries from {eval_dataset_path}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading file: {eval_dataset_path}\n{e}")
        return

    responses = []

    for idx, (query, orig_answer) in enumerate(zip(queries, original_answers), 1):
        print(f"\n[{idx}/{len(queries)}] Query: {query}")
        try:
            qa_response = await question_answer(query)
        except Exception as e:
            print(f"Error while processing query: {e}")
            await asyncio.sleep(30)
            continue

        result = qa_response.get("result", "No result")
        source_docs = qa_response.get("source_documents", [])

        print("Gemini Answer:")
        print(result)
        print("Original Answer:")
        print(orig_answer)
        print("Source Documents:")
        for doc in source_docs:
            snippet = re.sub(r'\s+', ' ', doc.page_content.strip())[:300]
            print(snippet + "\n")

        response_entry = {
            "query": query,
            "gemini_answer": result,
            "original_answer": orig_answer,
            "source_documents": [doc.page_content for doc in source_docs]
        }
        responses.append(response_entry)

        # Pause between queries to avoid rate limits
        await asyncio.sleep(30)

    responses_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "evaluation", "eval_result", "model_results_r50.json")
    try:
        with open(responses_path, "w") as f:
            json.dump(responses, f, indent=4)
        print(f"\nResponses saved to {responses_path}")
    except Exception as e:
        print(f"Error saving responses: {e}")

async def main():
    await evaluate_and_store_responses()

if __name__ == "__main__":
    asyncio.run(main())
