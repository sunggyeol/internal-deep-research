from typing import List
import json
from prompt import system_prompt
from ai.providers import generate_completions
from pydantic import BaseModel, ValidationError

class FeedbackResponse(BaseModel):
    questions: List[str]

async def generate_feedback(query: str, client, model: str, max_questions: int = 5) -> List[str]:
    prompt = (
        f"Given this research query: '{query}'\n\n"
        f"Generate {max_questions} focused research questions that would help guide a comprehensive investigation. "
        f"Return a JSON object with a 'questions' array containing the questions as strings.\n\n"
        f"Example format:\n"
        f'{{"questions": ["Question 1", "Question 2", "Question 3"]}}'
    )
    
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": prompt},
    ]
    
    try:
        response = await generate_completions(client=client, model=model, messages=messages, format=FeedbackResponse.model_json_schema())
        response_text = response.choices[0].message.content.strip()
        
        # Try parsing as pure JSON first
        if response_text.startswith("{") and response_text.endswith("}"):
            try:
                result = FeedbackResponse.model_validate_json(response_text)
                return result.questions
            except Exception as e:
                print(f"Pure JSON parsing failed: {e}")
        
        # Try extracting from code blocks if present
        if "```json" in response_text:
            try:
                json_content = response_text.split("```json")[1].split("```")[0].strip()
                result = FeedbackResponse.model_validate_json(json_content)
                return result.questions
            except Exception as e:
                print(f"Code block parsing failed: {e}")
        
        # Handle dictionary-like responses
        if isinstance(response_text, str) and "'question':" in response_text:
            try:
                import ast
                questions = []
                # Try to evaluate the string as a Python literal
                data = ast.literal_eval(response_text)
                if isinstance(data, dict) and "questions" in data:
                    raw_questions = data["questions"]
                    for q in raw_questions:
                        if isinstance(q, dict) and "question" in q:
                            questions.append(q["question"])
                        elif isinstance(q, str):
                            questions.append(q)
                return questions if questions else ["What are the main aspects of this topic?"]
            except Exception as e:
                print(f"Dictionary parsing failed: {e}")
        
        # Fallback to default question
        return ["What are the main aspects of this topic?"]
        
    except Exception as e:
        print(f"Error in generate_feedback: {e}")
        return ["What are the main aspects of this topic?"]
