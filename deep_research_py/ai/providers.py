import os
import openai
import asyncio
import tiktoken
from dotenv import load_dotenv
from text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
load_dotenv()

def create_openai_client(api_key: str, base_url: str = "https://api.openai.com/v1"):
    openai.api_key = api_key
    openai.api_base = base_url
    return openai

def get_ai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("Missing OPENAI_API_KEY in environment")
    return create_openai_client(api_key)

# --- Token Counting and Prompt Trimming ---

MIN_CHUNK_SIZE = 140

def get_token_count(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def trim_prompt(prompt: str, context_size: int = 128000) -> str:
    if not prompt:
        return ""
    length = get_token_count(prompt)
    if length <= context_size:
        return prompt
    overflow_tokens = length - context_size
    # Estimate characters to remove (roughly 3 per token)
    chunk_size = len(prompt) - overflow_tokens * 3
    if chunk_size < MIN_CHUNK_SIZE:
        return prompt[:MIN_CHUNK_SIZE]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    splits = splitter.split_text(prompt)
    trimmed_prompt = splits[0] if splits else ""
    if len(trimmed_prompt) == len(prompt):
        return trim_prompt(prompt[:chunk_size], context_size)
    return trim_prompt(trimmed_prompt, context_size)

async def generate_completions(client, model, messages, format):
    # Determine the response format based on whether a format (schema) is provided
    response_format_type = "json_object" if format else "text"

    # Run OpenAI call in thread pool since it's synchronous
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": response_format_type},
        ),
    )
    return response

# --- Optional: Testing the Client ---
if __name__ == "__main__":
    async def main():
        client = get_ai_client()
        model = "gpt-4"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about the effects of climate change."}
        ]
        # Pass 'None' for the format parameter if not using pydantic validation
        response = await generate_completions(client, model, messages, None)
        print(response)
    
    asyncio.run(main())
