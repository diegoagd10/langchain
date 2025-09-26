# LangChain Project

## Description

This project provides a setup for developing applications using LangChain with OpenAI and Ollama integration, featuring comprehensive data ingestion, text splitting, embeddings, vector storage, and generative AI capabilities from diverse sources including text files, PDFs, web pages, and academic papers. It includes environment variable management for API keys and the core LangChain dependency.

## Prerequisites

- Conda (Miniconda or Anaconda)
- macOS (osx-arm64 platform)
- OpenAI API key (obtain from https://platform.openai.com/)
- LangSmith API key (obtain from https://smith.langchain.com/)

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
   OPENAI_API_KEY=your_openai_key
   LANGSMITH_API_KEY=your_langsmith_key
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=your_project_name
   LANGSMITH_WORKSPACE_ID=your_workspace_id
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
- langchain-text-splitters: For text splitting utilities.
- langchain-openai: For OpenAI embeddings and integrations.
- langchain-ollama: For Ollama embeddings and local LLM integrations.
- chromadb: Vector database for storing and retrieving embeddings.
- sentence_transformers: For sentence-level embeddings using transformer models.
- langchain-huggingface: For Hugging Face integrations in LangChain.
- faiss-cpu: FAISS vector database for similarity search and clustering (CPU version).
- langchain_chroma: LangChain integration for ChromaDB vector database.
- streamlit: Web framework for creating interactive web applications.

## Environment Variables

The project uses the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key for accessing OpenAI services.
- `LANGSMITH_API_KEY`: Your LangSmith API key for tracing and monitoring.
- `LANGSMITH_TRACING`: Set to "true" to enable tracing.
- `LANGSMITH_PROJECT`: The name of your LangSmith project.
- `LANGSMITH_WORKSPACE_ID`: Your LangSmith workspace ID.

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
  - TextLoader for loading text files (e.g., `../resources/speech.txt`).
  - PyPDFLoader for loading PDF documents (e.g., `../resources/attention.pdf`).
  - WebBaseLoader for loading web pages, with optional BeautifulSoup parsing for targeted content extraction.
  - ArxivLoader for loading academic papers from ArXiv.
- `../resources/speech.txt`: Sample text file containing information about transformers.
- `../resources/attention.pdf`: Sample PDF file (likely the "Attention Is All You Need" paper).

To run the notebook, ensure you have Jupyter installed and run `jupyter notebook` in the project directory.

### Text Splitting

The `1-Langchain/2-TextSplitter/` directory contains comprehensive examples of text splitting using LangChain's text splitters.

- `CharacterTextSplitter.ipynb`: Demonstrates character-based text splitting for well-formatted documents using separators like "\n\n".
- `RecursiveTextSplitter.ipynb`: Shows recursive character text splitting for simple text formats, splitting by common parameters like ["\n", "\n\n", "", " "].
- `HTMLTextSplitter.ipynb`: Examples of splitting HTML content by specified HTML tags.
- `JSONTextSplitter.ipynb`: JSON text splitting by JSON values.

To run the notebooks, ensure you have Jupyter installed and run `jupyter notebook` in the project directory.

### Embeddings

The `1-Langchain/3-Embeddings/` directory contains examples of generating embeddings using different providers.

- `OllamaEmbedding.ipynb`: Demonstrates using Ollama for local embeddings (e.g., gemma:2b model), document and query embedding, text loading and splitting, creating vector stores with Chroma, and performing similarity searches.
- `OpenAIEmbedding.ipynb`: Shows how to use OpenAI embeddings (e.g., text-embedding-3-large), with options for dimensionality reduction, and the same workflow for vector storage and retrieval.

To run the notebooks, ensure you have Jupyter installed and run `jupyter notebook` in the project directory.

### Vector Stores

The `1-Langchain/4-VectorStore/` directory contains examples of vector storage using different providers.

- `Faiss.ipynb`: Demonstrates using FAISS for similarity search and clustering sentence vectors, including saving and loading the index.
- `Chroma.ipynb`: Shows how to use Chroma vector database for storing and retrieving embeddings, with persistence to disk.

To run the notebooks, ensure you have Jupyter installed and run `jupyter notebook` in the project directory.

### Generative AI

The `2-Generative AI/` directory contains examples for generative AI applications using LangChain.

- `2.1 OpenAI/1-GettingStarted.ipynb`: A Jupyter notebook demonstrating getting started with OpenAI's GPT models, including environment setup with LangSmith tracing, creating chat prompts, chaining with output parsers, and answering questions based on provided context.
- `2.1 OpenAI/2-ChatWithWebPage.ipynb`: A Jupyter notebook demonstrating how to load web page content, split text, create embeddings with OpenAI, store in a Chroma vector database, and perform retrieval-based question answering using LangChain.

To run the notebooks, ensure you have Jupyter installed and run `jupyter notebook` in the project directory.

### Ollama Integration

The `2-Generative AI/2.2 Ollama/` directory contains a Streamlit web application demonstrating local LLM integration using Ollama.

- `app.py`: A Streamlit web application that integrates with Ollama's Gemma 3 1B model for conversational AI. Features include:
- Environment setup with LangSmith tracing
- Chat prompt templates using LangChain
- Real-time conversation interface with Streamlit
- Integration with local Ollama models (requires Ollama to be installed and running)

To run the application:

1. Install and start Ollama from https://ollama.com/
2. Pull the Gemma model: `ollama pull gemma3:1b`
3. Run the Streamlit app: `streamlit run 2-Generative\ AI/2.2\ Ollama/app.py`

## Contributing

[Add contributing guidelines if needed]

## License

[Add license information if applicable]
