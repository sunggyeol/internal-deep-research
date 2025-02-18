import os
import asyncio
from litellm import completion, embedding
from dotenv import load_dotenv
from text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

def get_ai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("Missing GEMINI_API_KEY in environment")
    # litellm uses the API key from the environment, so simply return the module.
    import litellm
    return litellm

async def generate_completions(client, model, messages, format):
    # Use litellm.completion for generating completions.
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.completion(
            model=model,
            messages=messages,
        ),
    )
    return response

def generate_embedding(client, model, input_text):
    # Synchronous call to litellm.embedding for generating embeddings.
    return client.embedding(
        model=model,
        input=input_text,
    )
