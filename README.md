# LangChain Project

## Description

This project provides a setup for developing applications using LangChain with OpenAI integration, featuring comprehensive data ingestion capabilities from diverse sources including text files, PDFs, web pages, and academic papers. It includes environment variable management for API keys and the core LangChain dependency.

## Prerequisites

- Conda (Miniconda or Anaconda)
- macOS (osx-arm64 platform)
- OpenAI API key (obtain from https://platform.openai.com/)
- LangChain API key (obtain from https://www.langchain.com/)

## Installation

1. Create a new Conda environment with Python 3.10:

   ```
   conda create -p venv python==3.10 -y
   ```

2. Activate the environment:

   ```
   conda activate venv/
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:

   ```
   OPEN_API_KEY=your_openai_key
   LANGCHAIN_API_KEY=your_langchain_key
   LANGCHAIN_PROJECT=your_project_name
   ```

## Dependencies

- langchain: The core framework for building LLM applications.
- python-dotenv: For loading environment variables from .env file.
- ipykernel: For running Jupyter notebooks.
- langchain_community: For community-contributed document loaders and integrations.
- pypdf: For loading and processing PDF documents.
- bs4: BeautifulSoup for web scraping and HTML parsing.
- arxiv: For loading documents from ArXiv.
- pymupdf: Alternative PDF processing library.

## Environment Variables

The project uses the following environment variables:

- `OPEN_API_KEY`: Your OpenAI API key for accessing OpenAI services.
- `LANGCHAIN_API_KEY`: Your LangChain API key for LangChain services.
- `LANGCHAIN_PROJECT`: The name of your LangChain project.

Create a `.env` file in the project root and populate these variables. Do not commit `.env` to version control.

## Notes

- **Channel Warning**: You may see warnings about adding 'defaults' to the channel list implicitly. To resolve this, explicitly add channels using `conda config --add channels <name>`, e.g., `conda config --add channels defaults`.
- **Typo in Command**: The initial command had a typo (`pythong==3.10` instead of `python==3.10`), which caused a PackagesNotFoundError. Ensure correct spelling when creating environments.
- The environment is created in the project directory at `./venv`.

## Usage

Add your LangChain code here. For example:

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
# Your code here
```

## Examples

### Data Ingestion

The `1-Langchain/1-DataIngestion/` directory contains comprehensive examples of data ingestion using LangChain's document loaders.

- `DataIngestion.ipynb`: A Jupyter notebook demonstrating various document loaders:
  - TextLoader for loading text files (e.g., `speech.txt`).
  - PyPDFLoader for loading PDF documents (e.g., `attention.pdf`).
  - WebBaseLoader for loading web pages, with optional BeautifulSoup parsing for targeted content extraction.
  - ArxivLoader for loading academic papers from ArXiv.
- `speech.txt`: Sample text file containing information about transformers.
- `attention.pdf`: Sample PDF file (likely the "Attention Is All You Need" paper).

To run the notebook, ensure you have Jupyter installed and run `jupyter notebook` in the project directory.

## Contributing

[Add contributing guidelines if needed]

## License

[Add license information if applicable]
