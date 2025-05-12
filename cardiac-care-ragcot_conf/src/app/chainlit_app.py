import chainlit as cl
# Ensure this path is correct for your project structure and the function name
from app.rag_pipeline.handlers.gemini.chat_operations import Youtube_streaming

@cl.on_message
async def handle_message(msg: cl.Message):
    user_text = msg.content

    # Create an empty message object that will be updated with streamed content.
    # It will be sent to the UI only when the first chunk of the answer arrives.
    response_ui_msg = cl.Message(content="")
    
    has_streamed_main_answer = False  # Flag to track if the main answer has started streaming
    collected_source_documents = []   # To store source documents received from the stream

    # Start an activity to indicate processing, if desired
    # async with cl.Step(name="Processing Query") as step:
    #     step.input = user_text

    # Call the streaming function
    async for event in Youtube_streaming(user_text):
        event_type = event.get("type")
        event_data = event.get("data")

        if event_type == "sources":
            if isinstance(event_data, list):
                collected_source_documents = event_data
            # You could optionally send a quick message like:
            # await cl.Message(content=f"Found {len(collected_source_documents)} potential sources...").send()
            # For now, we'll display them after the main answer.

        elif event_type == "result_chunk":
            chunk_content = event_data
            if chunk_content:  # Make sure there's actual content
                if not has_streamed_main_answer:
                    # This is the first chunk of the main answer
                    response_ui_msg.content = chunk_content
                    await response_ui_msg.send()  # Send the message to the UI
                    has_streamed_main_answer = True
                else:
                    # These are subsequent chunks for the main answer
                    response_ui_msg.content += chunk_content
                    await response_ui_msg.update()  # Update the existing message in the UI
        
        elif event_type == "end_of_stream":
            # The stream from Youtube_streaming has finished
            break 

    # After the stream has finished, handle final UI updates

    # If no main answer content was streamed (e.g., only sources were found, or nothing useful)
    if not has_streamed_main_answer:
        if collected_source_documents:
            # We have sources, but no direct synthesised answer was streamed.
            await cl.Message(content="I couldn't formulate a direct answer based on the information, but here are some relevant sources:").send()
        else:
            # No answer and no sources.
            await cl.Message(content="Sorry, I couldn't find an answer or any relevant information for your query.").send()

    # Now, display the source documents if any were collected
    if collected_source_documents:
        # Optional: Send a header for the sources section
        # await cl.Message(content="--- Retrieved Sources ---").send()

        for i, doc_object in enumerate(collected_source_documents):
            # Assuming doc_object is a LangChain Document or a dict with 'page_content' and 'metadata'
            page_content = ""
            metadata = {}
            source_name_display = f"Source {i+1}" # Default name
            page_num_display = "N/A"

            if hasattr(doc_object, 'page_content') and hasattr(doc_object, 'metadata'):
                page_content = doc_object.page_content
                metadata = doc_object.metadata
                source_name_display = metadata.get("source", f"Document {i+1}")
                page_num_display = metadata.get("page", "N/A") # Or other relevant metadata like 'page_number'
            elif isinstance(doc_object, dict): # Fallback if it's already a dictionary
                page_content = doc_object.get("page_content", "")
                metadata = doc_object.get("metadata", {})
                source_name_display = metadata.get("source", f"Document {i+1}")
                page_num_display = metadata.get("page", "N/A")
            
            # Create a snippet for the message content
            snippet = page_content.strip()[:300] # Show first 300 characters as a snippet

            # Prepare content for the Chainlit message
            source_msg_content = f"**Source:** {source_name_display} (Page: {page_num_display})\n\n*Snippet:*\n{snippet}..."
            
            # Create a Text element to attach the full source content
            # This allows the user to view the full text if they wish
            text_element = cl.Text(
                name=f"Full content: {source_name_display}", # Name for the element tab
                content=page_content, 
                display="inline" # How it's displayed initially (inline, side, page)
            )

            await cl.Message(
                author="Retrieved Document", # Helps distinguish from the main AI response
                content=source_msg_content,
                elements=[text_element] # Attach the full document text
            ).send()
            
    # elif not has_streamed_main_answer and not collected_source_documents:
    #     # This case is already handled by the `if not has_streamed_main_answer:` block above.
    #     # If it gets here and nothing was sent, it means the RAG process yielded no usable information.
    #     pass


# To run this Chainlit app, you would typically use:
# chainlit run your_script_name.py -w