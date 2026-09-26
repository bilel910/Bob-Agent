from io import BytesIO

import chainlit as cl
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ai_companion.graph import graph_builder
from ai_companion.modules.image import ImageToText
from ai_companion.modules.speech import SpeechToText, TextToSpeech
from ai_companion.settings import settings

from langgraph.types import Command

from ai_companion.graph.utils.helpers import get_pending_question


# Global module instances
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()
image_to_text = ImageToText()


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    # thread_id = cl.user_session.get("id")
    cl.user_session.set("thread_id", 1)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle text messages and images"""
    msg = cl.Message(content="")

    # Process any attached images
    content = message.content
    if message.elements:
        for elem in message.elements:
            if isinstance(elem, cl.Image):
                # Read image file content
                with open(elem.path, "rb") as f:
                    image_bytes = f.read()

                # Analyze image and add to message content
                try:
                    # Use global ImageToText instance
                    description = await image_to_text.analyze_image(
                        image_bytes,
                        "Bitte beschreibe auf Deutsch, was du auf diesem Bild siehst, im Kontext unseres Gesprächs.",
                    )
                    content += f"\n[Bildanalyse: {description}]"
                except Exception as e:
                    cl.logger.warning(f"Failed to analyze image: {e}")

    # Process through graph with enriched message content
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    async with cl.Step(type="run"):
        async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
            graph = graph_builder.compile(checkpointer=short_term_memory)

            # Paused at a confirmation? Then this message is the answer, not a new request.
            if get_pending_question(await graph.aget_state(config)):
                graph_input = Command(resume=content)
            else:
                graph_input = {"messages": [HumanMessage(content=content)]}

            async for chunk in graph.astream(graph_input, config, stream_mode="messages"):
                if chunk[1]["langgraph_node"] in ("conversation_node", "action_node") and isinstance(chunk[0], AIMessageChunk):
                    await msg.stream_token(chunk[0].content)

            output_state = await graph.aget_state(config)

            # The checkpointer can leave the interrupt() write uncommitted. Without this
            # commit the pause is lost when the connection closes, and "ja" starts a new run.
            await short_term_memory.conn.commit()

    # The graph paused and is waiting for "ja"/"nein"
    question = get_pending_question(output_state)
    if question:
        await cl.Message(content=question).send()
        return

    if output_state.values.get("workflow") == "audio":
        response = output_state.values["messages"][-1].content
        audio_buffer = output_state.values["audio_buffer"]
        output_audio_el = cl.Audio(
            name="Audio",
            auto_play=True,
            mime="audio/mpeg3",
            content=audio_buffer,
        )
        await cl.Message(content=response, elements=[output_audio_el]).send()
    elif output_state.values.get("workflow") == "image":
        response = output_state.values["messages"][-1].content
        image = cl.Image(
            path=output_state.values["image_path"], display="inline")
        await cl.Message(content=response, elements=[image]).send()
    else:
        await msg.send()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    """Handle incoming audio chunks"""
    if chunk.isStart:
        buffer = BytesIO()
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements):
    """Process completed audio input"""
    # Get audio data
    audio_buffer = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    audio_data = audio_buffer.read()

    # Show user's audio message
    input_audio_el = cl.Audio(mime="audio/mpeg3", content=audio_data)
    await cl.Message(author="Du", content="", elements=[input_audio_el, *elements]).send()

    # Use global SpeechToText instance
    transcription = await speech_to_text.transcribe(audio_data)

    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
        graph = graph_builder.compile(checkpointer=short_term_memory)

        # Paused at a confirmation? Then this voice message is the answer.
        if get_pending_question(await graph.aget_state(config)):
            graph_input = Command(resume=transcription)
        else:
            graph_input = {"messages": [HumanMessage(content=transcription)]}

        await graph.ainvoke(graph_input, config)
        output_state = await graph.aget_state(config)
        await short_term_memory.conn.commit()

    # Either the confirmation question or Bob's normal answer
    response = get_pending_question(output_state) or output_state.values["messages"][-1].content

    # Use global TextToSpeech instance
    audio_buffer = await text_to_speech.synthesize(response)

    output_audio_el = cl.Audio(
        name="Audio",
        auto_play=True,
        mime="audio/mpeg3",
        content=audio_buffer,
    )
    await cl.Message(content=response, elements=[output_audio_el]).send()
