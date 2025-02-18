from typing import List, Dict
import asyncio
import json
from pydantic import BaseModel
from ai_providers import generate_completions
from prompt import system_prompt

class FinalReportResponse(BaseModel):
    reportMarkdown: str

async def write_final_report(
    prompt: str,
    learnings: List[str],
    visited_urls: List[str],
    client,
    model: str,
) -> str:
    learnings_string = "\n".join([f"<learning>\n{learning}\n</learning>" for learning in learnings])
    user_prompt = (
        f"Given the following prompt, write a comprehensive research paper using ONLY the provided learnings. "
        f"Return a JSON object with a 'reportMarkdown' field containing a research paper formatted in markdown. "
        f"Requirements:\n"
        f"1. Structure the paper following academic conference format:\n"
        f"   - Title\n"
        f"   - Abstract (summarize the key findings and implications)\n"
        f"   - Introduction (background, motivation, research questions)\n"
        f"   - Related Work (literature review from the sources)\n"
        f"   - Methodology (how the research was conducted)\n"
        f"   - Results and Discussion\n"
        f"   - Conclusion\n"
        f"   - References (compiled from citations)\n"
        f"2. The paper should be thorough and detailed, aim for at least 2000 words\n"
        f"3. Every piece of information MUST be directly cited from the source materials\n"
        f"4. Do not make assumptions or add information not present in the learnings\n"
        f"5. For each fact or statement, explicitly reference the source using [Citation: source_path]\n"
        f"6. When multiple sources discuss the same topic, synthesize and compare their information\n"
        f"7. Use direct quotes sparingly and only when particularly important, formatted as '...quote...' with proper citation\n"
        f"8. In the References section, list all cited sources in alphabetical order, using only the source paths from [Citation: source_path]\n\n"
        f"<prompt>{prompt}</prompt>\n\n"
        f"Here are the learnings:\n\n<learnings>\n{learnings_string}\n</learnings>"
    )
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        response = await generate_completions(client=client, model=model, messages=messages, format=FinalReportResponse.model_json_schema())
        response_text = response.choices[0].message.content.strip()
        
        # First try parsing as pure JSON
        if response_text.startswith("{") and response_text.endswith("}"):
            try:
                result = FinalReportResponse.model_validate_json(response_text)
                if result.reportMarkdown:
                    return result.reportMarkdown
            except Exception as json_error:
                print(f"JSON parsing error: {json_error}")
        
        # If that fails, try extracting JSON from markdown code blocks
        if response_text.startswith("```json"):
            try:
                json_content = response_text.split("```json")[1].split("```")[0].strip()
                result = FinalReportResponse.model_validate_json(json_content)
                if result.reportMarkdown:
                    return result.reportMarkdown
            except Exception as block_error:
                print(f"Code block parsing error: {block_error}")
        
        # If all JSON parsing fails, try to extract markdown content directly
        try:
            # Look for markdown content between quotes in the response
            import re
            markdown_match = re.search(r'"reportMarkdown":\s*"(.*?)(?<!\\)"(?=(,|\s*}))', response_text, re.DOTALL)
            if markdown_match:
                # Unescape any escaped quotes and return the markdown
                markdown_content = markdown_match.group(1).replace('\\"', '"')
                return markdown_content
        except Exception as re_error:
            print(f"Regex extraction error: {re_error}")
        
        # Last resort: return the raw response if it looks like markdown
        if "# " in response_text or "## " in response_text:
            return response_text
        
        # If nothing works, return an error message
        return "Error: Unable to parse the report response"
    
    except Exception as e:
        print(f"Error in write_final_report: {str(e)}")
        return f"Error generating report: {str(e)}"

async def research_from_directory(
    directory: str,
    query: str,
    client,
    model: str,
) -> Dict[str, str]:
    from aggregator import aggregate_files
    raw_documents = aggregate_files(directory)
    if not raw_documents:
        print("No valid documents found.")
        return {"report": "No content found to generate a report."}
    
    # Split each file's content into chunks while preserving source metadata.
    from text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for doc in raw_documents:
        file_chunks = splitter.split_text(doc["content"])
        for chunk in file_chunks:
            chunks.append({"page_content": chunk, "source": doc["source"]})
    
    if not chunks:
        print("Failed to split content into chunks.")
        return {"report": "Error processing the content."}
    
    # Create a vector store from the chunks using FAISS and LangChain's OpenAIEmbeddings.
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    import faiss
    embeddings_model = OpenAIEmbeddings(model="gemini/text-embedding-004")
    sample_embedding = embeddings_model.embed_query(chunks[0]["page_content"])
    dim = len(sample_embedding) if sample_embedding else 768
    index = faiss.IndexFlatL2(dim)
    vector_store = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    
    from uuid import uuid4
    try:
        from langchain_core.documents import Document
    except ImportError:
        from collections import namedtuple
        Document = namedtuple("Document", ["page_content", "metadata"])
    documents = [
        Document(page_content=item["page_content"], metadata={"source": item["source"]})
        for item in chunks
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    vector_store.save_local("embeddings_index")
    
    # Perform semantic search over the vector store using the provided query.
    search_results = vector_store.similarity_search(query, k=5)
    context_with_citations = "\n".join([
        f"{doc.page_content} [Citation: {doc.metadata.get('source', 'unknown')}]"
        for doc in search_results
    ])
    
    full_prompt = f"Research query: {query}\n\nRelevant context from documents:\n\n{context_with_citations}"
    learnings = [context_with_citations]
    report = await write_final_report(
        prompt=full_prompt,
        learnings=learnings,
        visited_urls=[],
        client=client,
        model=model,
    )
    return {"report": report}

async def iterative_research(
    directory: str,
    initial_query: str,
    client,
    model: str,
    iterations: int = 3,
) -> Dict[str, str]:
    print("Step 1: Aggregating documents and building FAISS vector store...")
    from aggregator import aggregate_files
    raw_documents = aggregate_files(directory)
    if not raw_documents:
        print("No valid documents found.")
        return {"report": "No content found to generate a report."}
    
    from text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for doc in raw_documents:
        file_chunks = splitter.split_text(doc["content"])
        for chunk in file_chunks:
            chunks.append({"page_content": chunk, "source": doc["source"]})
    
    if not chunks:
        print("Failed to split content into chunks.")
        return {"report": "Error processing the content."}
    
    print("Step 2: Creating FAISS vector store from document chunks...")
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    import faiss
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    sample_embedding = embeddings_model.embed_query(chunks[0]["page_content"])
    dim = len(sample_embedding) if sample_embedding else 768
    index = faiss.IndexFlatL2(dim)
    vector_store = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    from uuid import uuid4
    try:
        from langchain_core.documents import Document
    except ImportError:
        from collections import namedtuple
        Document = namedtuple("Document", ["page_content", "metadata"])
    documents = [
        Document(page_content=item["page_content"], metadata={"source": item["source"]})
        for item in chunks
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    vector_store.save_local("embeddings_index")
    
    current_query = initial_query
    final_report = ""
    
    for i in range(1, iterations + 1):
        print(f"\n=== Iteration {i} ===")
        print("Step 3: Generating question prompts based on current query...")
        try:
            from feedback import generate_feedback
            question_prompts = await generate_feedback(current_query, client, model)
            if not question_prompts:
                question_prompts = ["What are the main aspects of this topic?"]
            
            print("LLM Generated Question Prompts:")
            for idx, prompt_text in enumerate(question_prompts, start=1):
                if isinstance(prompt_text, dict):
                    prompt_text = prompt_text.get('question', str(prompt_text))
                print(f"  Q{idx}: {prompt_text}")
            
            print("Step 4: Retrieving relevant document context for each prompt with citations...")
            contexts = []
            for prompt_question in question_prompts:
                if isinstance(prompt_question, dict):
                    prompt_question = prompt_question.get('question', str(prompt_question))
                search_results = vector_store.similarity_search(prompt_question, k=5)
                context_with_citations = "\n".join([
                    f"{doc.page_content} [Citation: {doc.metadata.get('source', 'unknown')}]"
                    for doc in search_results
                ])
                contexts.append(context_with_citations)
            combined_context = "\n".join(contexts)
            
            print("Step 5: Generating final report for this iteration...")
            full_prompt = f"Research query: {current_query}\n\nRelevant context from documents:\n\n{combined_context}"
            final_report = await write_final_report(
                prompt=full_prompt,
                learnings=[combined_context],
                visited_urls=[],
                client=client,
                model=model,
            )
            print(f"\nFinal Report for iteration {i}:\n{final_report}\n")
            
            print("Step 6: Refining the research query for the next iteration...")
            refine_prompt = (
                f"Based on the final report below, refine the research query for further investigation.\n\n"
                f"Final Report:\n{final_report}\n\n"
                f"Original Query: {current_query}\n\n"
                f"Provide the refined research query."
            )
            messages = [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": refine_prompt},
            ]
            response = await generate_completions(client=client, model=model, messages=messages, format=None)
            refined_query = response.choices[0].message.content.strip()
            print("Refined Query for next iteration:")
            print(refined_query)
            current_query = refined_query
        
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            continue
    
    return {"report": final_report if final_report else "Error: No report generated"}
