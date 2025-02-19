# Internal Deep Research

Internal Deep Research is an AI-powered tool designed to help organizations generate comprehensive research reports from internal documents. Inspired by deep research methodologies from OpenAI, Google, and Perplexity, this project aggregates content from text and markdown files and synthesizes it into detailed markdown reports.

## Overview

Internal Deep Research helps analysts and researchers quickly generate AI-powered research reports from internal files. The system works by scanning a specified directory for research-related documents, aggregating their content, and then passing a combined prompt—including an iterative refinement process—to an LLM to produce a structured markdown report with clear citations.

## Features

- **File Aggregation:**  
  Recursively aggregates content from `.txt` and `.md` files within a specified directory. Each document’s source is preserved for citation purposes.

- **AI-Powered Report Generation:**  
  Uses an LLM to analyze the aggregated content and generate a comprehensive research report that includes citations indicating the source of each piece of information.

- **Iterative Research Process:**  
  Incorporates an iterative cycle where:
  - The document corpus is processed once into a FAISS vector store.
  - The system generates follow-up question prompts based on the current research query.
  - Each prompt is used to retrieve relevant document chunks with citation markers.
  - A final report is synthesized and then used to refine the research query for the next cycle.
  
- **Modular Architecture:**  
  The project is organized into several modules:
  - **aggregator.py** – Aggregates text content from files in a directory while retaining source metadata.
  - **ai_providers.py** – Configures and manages the LLM API client.
  - **text_splitter.py** – Splits large texts into manageable chunks.
  - **deep_research.py** – Contains the core research logic, including iterative report generation and citation handling.
  - **feedback.py** – Generates follow-up questions to refine research queries.
  - **prompt.py** – Houses the system prompt used to guide the AI.
  - **run.py** – The command-line entry point.
  - **utils.py** – Contains utility functions for configuration.

- **Plain-Text CLI Interface:**  
  The tool operates via a simple text-based interface, making it straightforward to use in various environments.

## How It Works

1. **Initialization:**  
   - The tool loads environment variables (including your API key) using `dotenv`.
   - The LLM client is initialized.

2. **File Aggregation:**  
   - When the `--directory` option is used, the tool recursively scans for `.txt` and `.md` files.
   - The content from these files is aggregated along with metadata indicating each file’s source.

3. **Prompt Generation and Iterative Research:**  
   - The user provides a research query.
   - The system processes the aggregated documents into a FAISS vector store.
   - The LLM generates follow-up question prompts based on the current query.
   - Each prompt is used to retrieve relevant document chunks, with citations showing the file source.
   - A final report is generated for the iteration and is then used to refine the research query for subsequent cycles.

4. **Report Generation:**  
   - The AI synthesizes the provided context and generates a detailed markdown research report that includes clear citations.

5. **Output:**  
   - The generated report is displayed in the terminal.
   - It is also saved as `output.md` for further reference.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/sunggyeol/internal-deep-research.git
   cd internal-deep-research
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**  
   Create a `.env` file in the root directory and add your API key:

   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Usage

### File-Based Research Mode

Run the tool on a directory containing research files:

```bash
python deep_research/run.py --directory "knowledge_base" --iterations 5
```

This command will execute the iterative research process, retrieving document content with citations and refining the query over multiple iterations.

### Interactive Mode

If you omit the `--directory` flag, the tool will prompt you for a research query and follow-up questions interactively:

```bash
python deep_research/run.py
```

## Future Enhancements

- **Expanded File Support:**  
  Planned support for PDFs, image-based text extraction, and additional formats.

- **Enhanced Interactive Experience:**  
  Further refinements in feedback generation and query refinement.

## Acknowledgements

This project is inspired by deep research methodologies developed by OpenAI, Google, and Perplexity. It is adapted from [deep-research-py](https://github.com/epuerta9/deep-research-py) with a focus on internal file-based research.

## License

This project is licensed under the **MIT License**.