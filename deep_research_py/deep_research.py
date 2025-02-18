from typing import List, Dict
import asyncio
import json
from pydantic import BaseModel
import openai

from ai.providers import trim_prompt, generate_completions
from prompt import system_prompt
from common.logging import log_event, log_error
from common.token_consumption import parse_openai_token_consume
from utils import get_service

class FinalReportResponse(BaseModel):
    reportMarkdown: str

async def write_final_report(
    prompt: str,
    learnings: List[str],
    visited_urls: List[str],
    client: openai,
    model: str,
) -> str:
    learnings_string = trim_prompt(
        "\n".join([f"<learning>\n{learning}\n</learning>" for learning in learnings]),
        150_000,
    )
    user_prompt = (
        f"Given the following prompt, write a final report on the topic using the learnings. "
        f"Return a JSON object with a 'reportMarkdown' field containing a detailed markdown report:\n\n"
        f"<prompt>{prompt}</prompt>\n\n"
        f"Here are the learnings:\n\n<learnings>\n{learnings_string}\n</learnings>"
    )
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt},
    ]
    response = await generate_completions(
        client=client, model=model, messages=messages, format=FinalReportResponse.model_json_schema()
    )
    try:
        result = FinalReportResponse.model_validate_json(response.choices[0].message.content)
        parse_openai_token_consume("write_final_report", response)
        report = result.reportMarkdown if result.reportMarkdown else ""
        return report
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        log_error(f"Failed to generate final report. Raw response: {response.choices[0].message.content}")
        return "Error generating report"

async def research_from_directory(
    directory: str,
    query: str,
    client: openai,
    model: str,
) -> Dict[str, str]:
    from aggregator import aggregate_files
    aggregated_content = aggregate_files(directory)
    if not aggregated_content.strip():
        print("No valid content found in the directory.")
        return {"report": "No content found to generate a report."}
    full_prompt = f"Research query: {query}\n\nAggregated content from directory '{directory}':\n\n{aggregated_content}"
    learnings = [aggregated_content]
    visited_urls = []  # Not applicable for file-based research.
    report = await write_final_report(
        prompt=full_prompt,
        learnings=learnings,
        visited_urls=visited_urls,
        client=client,
        model=model,
    )
    return {"report": report}
