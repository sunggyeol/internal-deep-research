# Internal Deep Research

Internal Deep Research is an AI-powered tool designed to help organizations generate comprehensive research reports from internal documents. Inspired by "Deep Research" methodologies from OpenAI, Google, and Gemini, this project aggregates content from text and markdown files and synthesizes it into detailed markdown reports.

This repository's structure and core concept are adapted from [deep-research-py](https://github.com/epuerta9/deep-research-py). However, **this version intentionally removes search functionality** to focus solely on internal file-based research.

For now, the tool supports only `.txt` and `.md` files, with planned support for PDFs, images, and other formats in the future.

## Overview

Internal Deep Research helps analysts and researchers quickly generate AI-powered research reports from internal files. The system works by scanning a specified directory for research-related documents, aggregating their content, and then passing a combined prompt (including a research query) to OpenAI's LLM to produce a structured markdown report.

## Features

- **File Aggregation:**  
  Recursively aggregates content from `.txt` and `.md` files within a specified directory.

- **AI-Powered Report Generation:**  
  Uses OpenAI's LLM to analyze the aggregated content and generate a research report.

- **Modular Architecture:**  
  The project is organized into several modules:
  - **aggregator.py** – Aggregates text content from files in a directory.
  - **ai/providers.py** – Configures and manages the OpenAI API client.
  - **text_splitter.py** – Splits large texts into manageable chunks.
  - **deep_research.py** – Core research logic for synthesizing a final report.
  - **feedback.py** – (Optional) Generates follow-up questions for interactive research mode.
  - **prompt.py** – Houses the system prompt used to guide the AI.
  - **run.py** – The command-line entry point.
  - **utils.py** – Contains utility functions for configuration.
  - **common/logging.py & token_cunsumption.py** – Handle logging and token consumption tracking.

- **Plain-Text CLI Interface:**  
  The tool operates via a simple text-based interface without requiring rich-text UI libraries.

## Project Structure

```
internal-deep-research/
├── README.md
├── deep_research_py/
│   ├── __init__.py
│   ├── aggregator.py              # Aggregates content from text and markdown files.
│   ├── ai/
│   │   ├── __init__.py
│   │   └── providers.py           # Configures and provides access to the OpenAI API.
│   ├── text_splitter.py           # Splits long text into manageable chunks.
│   ├── deep_research.py           # Core research logic and report generation.
│   ├── feedback.py               # Generates follow-up questions (for interactive mode).
│   ├── prompt.py                  # Contains the system prompt for the AI.
│   ├── run.py                     # CLI entry point.
│   ├── utils.py                   # Utility functions for configuration.
│   └── common/
│       ├── logging.py             # Logging utilities.
│       └── token_cunsumption.py   # Token consumption tracking.
└── sample_files/                  # (Optional) Sample files for testing.
    ├── sample1.txt
    ├── sample2.md
    └── sample3.txt
```

## How It Works

1. **Initialization:**  
   - The tool loads environment variables (including your OpenAI API key) using `dotenv`.
   - The OpenAI client is initialized.

2. **File Aggregation:**  
   - If the `--directory` option is used, the tool recursively scans for `.txt` and `.md` files.
   - The content from these files is aggregated into a single text string.

3. **Prompt Generation:**  
   - The user provides a research query.
   - This query is combined with the aggregated file content to form a structured prompt.
   - The prompt is sent to OpenAI's LLM for report generation.

4. **Report Generation:**  
   - The AI processes the prompt and returns a well-structured markdown research report.

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
   Create a `.env` file in the root directory and add your OpenAI API key:

   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### File-Based Research Mode

Run the tool on a directory containing research files:

```bash
python deep_research_py/run.py --directory "sample_files"
```

### Interactive Mode

If you omit the `--directory` flag, the tool will prompt you for a research query and follow-up questions interactively:

```bash
python deep_research_py/run.py
```

## Future Enhancements

- **Expanded File Support:**  
  - Planned support for PDF, image-based text extraction, and other formats.

- **Improved Interactive Mode:**  
  - Enhanced feedback generation for refining research queries.

## Acknowledgements

This project is **inspired by "Deep Research" methodologies** developed by OpenAI, Google, and Gemini.

It is **adapted from [deep-research-py](https://github.com/epuerta9/deep-research-py)**.  
However, **this version removes search engine integration**, focusing entirely on **internal file-based research**.

## License

This project is licensed under the **MIT License**.