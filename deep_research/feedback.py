from typing import List
import json
from prompt import system_prompt
from ai.providers import generate_completions
from pydantic import BaseModel, ValidationError

class FeedbackResponse(BaseModel):
    questions: List[str]

async def generate_feedback(
    query: str,
    client,
    model: str,
    max_feedbacks: int = 5,
) -> List[str]:
    prompt = (
        f"Given this research topic: {query}, generate at most {max_feedbacks} investigative questions that explore key aspects, "
        "gaps, relationships, and implications of this topic. Focus on questions about the knowledge domain itself, "
        "not questions for the user. Return the response as a JSON object with a 'questions' array field."
    )
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": prompt},
    ]
    response = await generate_completions(client=client, model=model, messages=messages, format=FeedbackResponse.model_json_schema())
    try:
        response_text = response.choices[0].message.content.strip()
        # Remove markdown code fences if present
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
        result = FeedbackResponse.parse_raw(response_text)
        return result.questions
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Error parsing JSON response in generate_feedback: {e}")
        return []
