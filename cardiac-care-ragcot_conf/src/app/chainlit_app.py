import chainlit as cl
from app.rag_pipeline.handlers.gemini.chat_operations import Youtube_streaming

@cl.on_chat_start
async def start():
    # Initialize empty sidebar
    await cl.ElementSidebar.set_title("Source References")
    await cl.ElementSidebar.set_elements([])

@cl.on_message
async def handle_message(msg: cl.Message):
    user_text = msg.content

    # Prepare streaming UI message
    response_ui_msg = cl.Message(content="", author="Assistant")
    has_streamed_main_answer = False
    collected_source_documents = []

    # Stream responses
    async for event in Youtube_streaming(user_text):
        etype = event.get("type")
        data = event.get("data")

        if etype == "result_chunk":
            chunk = data or ""
            if not has_streamed_main_answer:
                response_ui_msg.content = chunk
                await response_ui_msg.send()
                has_streamed_main_answer = True
            else:
                response_ui_msg.content += chunk
                await response_ui_msg.update()

        elif etype == "sources":
            if isinstance(data, list):
                collected_source_documents = data

        elif etype == "end_of_stream":
            break

    # Store sources in session
    cl.user_session.set("sources", collected_source_documents)

    # Handle absence of main answer
    if not has_streamed_main_answer:
        if collected_source_documents:
            await cl.Message(content="I couldn't generate a direct answer, but relevant sources are available in the sidebar.").send()
        else:
            await cl.Message(content="Sorry, I couldn't find an answer or relevant information for your query.").send()

    # Update sidebar with sources
    if collected_source_documents:
        md = "### Retrieved Sources\n"
        for i, doc in enumerate(collected_source_documents, start=1):
            if isinstance(doc, dict):
                content = doc.get("page_content", "")
                meta = doc.get("metadata", {})
            else:
                content = getattr(doc, "page_content", "")
                meta = getattr(doc, "metadata", {})
            src = meta.get("source", f"Source {i}")
            page = meta.get("page", "N/A")
            md += f"\n**{i}. {src}** (*{page}*)\n{content}\n"

        # Populate sidebar only
        await cl.ElementSidebar.set_title(f"Sources for: {user_text}")
        await cl.ElementSidebar.set_elements([
            cl.Text(name="sources_md", content=md, display="side")
        ])
