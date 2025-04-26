# from app.rag_pipeline.handlers.generic.interaction_utils import (
#     initialize_vector_database,
#     initialize_compression_retriever,
#     retrieve_and_compress_documents,
# )

# # 1. Initialize your vector DB and retriever
# vector_db = initialize_vector_database()
# compression_retriever = initialize_compression_retriever(vector_db)

# # 2. Define a test query
# query = "Why was patient discharged?"

# # 3. Retrieve and compress relevant documents
# compressed_docs, retrieved_docs = retrieve_and_compress_documents(query, vector_db, compression_retriever)

# import logging
# import json

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Log the retrieved documents in a proper format
# logger.info("Retrieved Documents:")
# for doc in retrieved_docs:
#     logger.info(json.dumps({"Document ID": doc['id'], "Text": doc['text'][:200]}, indent=2))

# # Log the compressed documents in a proper format
# logger.info("Compressed Documents:")
# for doc in compressed_docs:
#     logger.info(json.dumps({"Document ID": doc['id'], "Text": doc['text'][:200]}, indent=2))