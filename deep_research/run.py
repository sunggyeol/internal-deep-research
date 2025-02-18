from dotenv import load_dotenv
import asyncio
import typer
from prompt_toolkit import PromptSession
from feedback import generate_feedback
from ai_providers import get_ai_client
from utils import set_model, convert_to_pdf
from deep_research import research_from_directory, write_final_report, iterative_research
import os

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
    model: str = typer.Option("gemini/gemini-2.0-flash", help="Which model to use?"),
    max_followup_questions: int = typer.Option(5, help="Maximum follow-up questions to generate (only in interactive mode)."),
    iterations: int = typer.Option(3, help="Number of iterative research cycles."),
    output_dir: str = typer.Option("outputs", help="Directory to save output files"),
):
    set_model(model)
    
    print("Deep Research Assistant")
    print("An AI-powered research tool")
    print("Using litellm as the LLM provider.\n")
    
    client = get_ai_client()
    
    query = await async_prompt("What would you like to research? ")
    print()
    
    if directory:
        print(f"Aggregating files from directory: {directory}")
        print("Starting iterative research process...")
        result = await iterative_research(directory, query, client, model, iterations=iterations)
        report = result.get("report", "No report generated.")
        
        print("\nResearch Complete!")
        print("\nFinal Report:")
        print(report)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file paths
        md_file = os.path.join(output_dir, "output.md")
        pdf_file = os.path.join(output_dir, "output.pdf")
        
        # Save markdown file
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\nMarkdown report saved to: {md_file}")
        
        # Convert to PDF
        convert_to_pdf(md_file, pdf_file)
        print(f"PDF report saved to: {pdf_file}")
    
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
    
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file paths
        md_file = os.path.join(output_dir, "output.md")
        pdf_file = os.path.join(output_dir, "output.pdf")
        
        # Save markdown file
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\nMarkdown report saved to: {md_file}")
        
        # Convert to PDF
        convert_to_pdf(md_file, pdf_file)
        print(f"PDF report saved to: {pdf_file}")

def run():
    asyncio.run(app())

if __name__ == "__main__":
    asyncio.run(app())
