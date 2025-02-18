from typing import List, Dict
import asyncio
import json
from pydantic import BaseModel
from ai.providers import generate_completions
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
        f"Given the following prompt, write a final report on the topic using the learnings. "
        f"Return a JSON object with a 'reportMarkdown' field containing a detailed markdown report. "
        f"Ensure that the report clearly cites the source of each piece of information (e.g., [Citation: source_path]).\n\n"
        f"<prompt>{prompt}</prompt>\n\n"
        f"Here are the learnings:\n\n<learnings>\n{learnings_string}\n</learnings>"
    )
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt},
    ]
    response = await generate_completions(client=client, model=model, messages=messages, format=FinalReportResponse.model_json_schema())
    try:
        response_text = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
        result = FinalReportResponse.model_validate_json(response_text)
        report = result.reportMarkdown if result.reportMarkdown else ""
        return report
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return "Error generating report"

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
        from feedback import generate_feedback
        question_prompts = await generate_feedback(current_query, client, model)
        print("LLM Generated Question Prompts:")
        for idx, prompt_text in enumerate(question_prompts, start=1):
            print(f"  Q{idx}: {prompt_text}")
        
        print("Step 4: Retrieving relevant document context for each prompt with citations...")
        contexts = []
        for prompt_question in question_prompts:
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
    
    return {"report": final_report}
