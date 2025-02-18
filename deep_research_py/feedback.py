from typing import List
import openai
import json
from prompt import system_prompt
from common.logging import log_error, log_event
from ai.providers import generate_completions
from common.token_consumption import parse_openai_token_consume
from utils import get_service
from pydantic import BaseModel

class FeedbackResponse(BaseModel):
    questions: List[str]

async def generate_feedback(
    query: str,
    client: openai,
    model: str,
    max_feedbacks: int = 5,
) -> List[str]:
    prompt = (
        f"Given this research topic: {query}, generate at most {max_feedbacks} follow-up questions to better understand the user's research needs. "
        "Return the response as a JSON object with a 'questions' array field."
    )
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": prompt},
    ]
    response = await generate_completions(
        client=client, model=model, messages=messages, format=FeedbackResponse.model_json_schema()
    )
    try:
        result = FeedbackResponse.parse_raw(response.choices[0].message.content)
        log_event(f"Generated {len(result.questions)} follow-up questions for query: {query}")
        return result.questions
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        log_error(f"Failed to parse JSON response for query: {query}")
        return []
