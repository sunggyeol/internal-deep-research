from dotenv import load_dotenv
import asyncio
import typer
from prompt_toolkit import PromptSession
from feedback import generate_feedback
from ai.providers import get_ai_client
from utils import set_model
from common.token_consumption import counter
from common.logging import log_event
from deep_research import research_from_directory, write_final_report

load_dotenv()
app = typer.Typer()
session = PromptSession()

def coro(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

async def async_prompt(message: str, default: str = "") -> str:
    return await session.prompt_async(message)

@app.command()
@coro
async def main(
    directory: str = typer.Option("", help="Path to a directory containing text/markdown files for research."),
    concurrency: int = typer.Option(2, help="Number of concurrent tasks (affects API rate limits)."),
    model: str = typer.Option("gpt-4o", help="Which model to use?"),
    max_followup_questions: int = typer.Option(5, help="Maximum follow-up questions to generate (only in interactive mode)."),
    enable_logging: bool = typer.Option(False, help="Enable logging."),
    log_path: str = typer.Option("logs", help="Path to save logs."),
    log_to_stdout: bool = typer.Option(False, help="Log to stdout."),
):
    set_model(model)
    
    if enable_logging:
        from common.logging import initial_logger
        initial_logger(logging_path=log_path, enable_stdout=log_to_stdout)
        print(f"Logging enabled. Logs will be saved to {log_path}")
    
    print("Deep Research Assistant")
    print("An AI-powered research tool")
    print("Using OpenAI as the LLM provider.\n")
    
    client = get_ai_client()
    
    query = await async_prompt("What would you like to research? ")
    print()
    
    if directory:
        print(f"Aggregating files from directory: {directory}")
        print("Generating report from files...")
        result = await research_from_directory(directory, query, client, model)
        report = result.get("report", "No report generated.")
        
        print("\nResearch Complete!")
        print("\nFinal Report:")
        print(report)
        
        with open("output.md", "w", encoding="utf-8") as f:
            f.write(report)
        if enable_logging:
            log_event("Report has been saved to output.md (file-based research mode).")
    
    else:
        print("Creating research plan...")
        follow_up_questions = await generate_feedback(query, client, model, max_followup_questions)
        if follow_up_questions:
            print("Follow-up Questions:")
            answers = []
            for i, question in enumerate(follow_up_questions, 1):
                print(f"Q{i}: {question}")
                answer = await async_prompt("Your answer: ")
                answers.append(answer)
                print()
        else:
            print("No follow-up questions needed!")
            answers = []
    
        combined_query = f"""
Initial Query: {query}
Follow-up Questions and Answers:
{chr(10).join(f"Q: {q} A: {a}" for q, a in zip(follow_up_questions, answers))}
"""
    
        print("Researching your topic...")
        report = await write_final_report(
            prompt=combined_query,
            learnings=[combined_query],
            visited_urls=[],
            client=client,
            model=model,
        )
    
        print("\nResearch Complete!")
        print("\nFinal Report:")
        print(report)
    
        with open("output.md", "w", encoding="utf-8") as f:
            f.write(report)
        if enable_logging:
            log_event("Report has been saved to output.md (interactive mode).")

def run():
    asyncio.run(app())

if __name__ == "__main__":
    asyncio.run(app())
