# LangChain Project

## Description

This project provides a setup for developing applications using LangChain with OpenAI and Ollama integration, featuring comprehensive data ingestion, text splitting, embeddings, vector storage, and generative AI capabilities from diverse sources including text files, PDFs, web pages, and academic papers. It includes environment variable management for API keys and the core LangChain dependency.

## Prerequisites

- Conda (Miniconda or Anaconda)
- macOS (osx-arm64 platform)
- OpenAI API key (obtain from https://platform.openai.com/)
- LangSmith API key (obtain from https://smith.langchain.com/)
- Groq API key (obtain from https://groq.com/)

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
   GROQ_API_KEY=your_groq_key
   ```

## Dependencies

The project dependencies are organized into the following categories:

### Core/Base Libraries

- pydantic==2.9: Data validation and settings management using Python type annotations.
- python-dotenv: For loading environment variables from .env file.

### LangChain Ecosystem

- langchain: The core framework for building LLM applications.
- langchain_community: For community-contributed document loaders and integrations.
- langchain-text-splitters: For text splitting utilities.

### AI/ML Libraries

- sentence_transformers: For sentence-level embeddings using transformer models.
- langchain-huggingface: For Hugging Face integrations in LangChain.

### Vector Databases and Storage

- chromadb: Vector database for storing and retrieving embeddings.
- faiss-cpu: FAISS vector database for similarity search and clustering (CPU version).
- langchain_chroma: LangChain integration for ChromaDB vector database.

### Document Processing

- pypdf: For loading and processing PDF documents.
- bs4: BeautifulSoup for web scraping and HTML parsing.
- pymupdf: Alternative PDF processing library.
- arxiv: For loading documents from ArXiv.
- wikipedia: Python library for accessing Wikipedia's API.

### LLM Providers

- langchain-openai: For OpenAI embeddings and integrations.
- langchain-ollama: For Ollama embeddings and local LLM integrations.
- groq: Direct Groq API client library.
- langchain_groq: For Groq API integrations in LangChain.

### Web Frameworks

- fastapi: Modern, fast web framework for building APIs.
- uvicorn: ASGI web server implementation for Python.
- sse_starlette: Server-sent events implementation for Starlette.
- streamlit: Web framework for creating interactive web applications.

### Development and Runtime Tools

- ipykernel: For running Jupyter notebooks.
- langserve: Framework for deploying LangChain applications as APIs.

## Environment Variables

The project uses the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key for accessing OpenAI services.
- `LANGSMITH_API_KEY`: Your LangSmith API key for tracing and monitoring.
- `LANGSMITH_TRACING`: Set to "true" to enable tracing.
- `LANGSMITH_PROJECT`: The name of your LangSmith project.
- `LANGSMITH_WORKSPACE_ID`: Your LangSmith workspace ID.
- `GROQ_API_KEY`: Your Groq API key for accessing Groq's fast inference services.

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

### Groq Integration

The `2-Generative AI/2.3 Groq/` directory contains a Streamlit web application demonstrating Groq's fast inference API with RAG (Retrieval-Augmented Generation).

- `app.py`: A Streamlit web application that combines Groq's fast inference with document retrieval:
  - **Web Document Loading**: Loads content from Vatican website (Lumen Fidei encyclical)
  - **Document Processing**: Splits documents using RecursiveCharacterTextSplitter
  - **Vector Storage**: Creates ChromaDB vector database with Ollama embeddings
  - **Groq Integration**: Uses Llama 3.1 8B model for fast inference
  - **Retrieval Chain**: Combines document retrieval with Groq responses
  - **Session State**: Caches vector database for improved performance
  - **Interactive UI**: Streamlit interface for asking questions about the loaded content

Key Features:

- Real-time document question answering
- Fast inference using Groq's API
- Context-aware responses based on loaded documents
- Persistent vector storage during session
- LangSmith tracing integration

To run the application:

1. Obtain a Groq API key from https://groq.com/
2. Add `GROQ_API_KEY=your_groq_key` to your `.env` file
3. Ensure Ollama is running with `mxbai-embed-large` model: `ollama pull mxbai-embed-large:latest`
4. Run the Streamlit app: `streamlit run 2-Generative\ AI/2.3\ Groq/app.py`

### Text Classification

The `2-Generative AI/2.4 Classification/` directory contains a Streamlit web application for text classification using structured output with Pydantic models.

- `app.py`: Main Streamlit application that provides two classification modes:

  - **Basic Classification**: Free-form text analysis with open-ended categories
  - **Detailed Classification**: Structured classification with predefined enums and constraints
  - **Groq Integration**: Uses Gemma 2 9B model for fast, accurate classification
  - **Pydantic Models**: Structured output validation and type safety
  - **Interactive UI**: Real-time text classification with immediate results

- `Classification.py`: Basic classification data model with Pydantic:

  - **sentiment**: Free-form sentiment analysis
  - **aggressiveness**: 1-10 scale for text aggressiveness
  - **language**: Free-form language detection

- `DetailedClassification.py`: Advanced classification with constrained enums:
  - **sentiment**: Predefined options (happy, neutral, sad, angry)
  - **aggressiveness**: 1-5 scale with enum constraints
  - **language**: Specific language options (spanish, english, french, german, italian)

Key Features:

- Dual classification modes for different use cases
- Structured output with type validation
- Fast inference using Groq's Gemma 2 9B model
- Real-time classification results
- LangSmith tracing for observability
- Professional UI with separate result sections

To run the application:

1. Obtain a Groq API key from https://groq.com/
2. Add `GROQ_API_KEY=your_groq_key` to your `.env` file
3. Run the Streamlit app: `streamlit run 2-Generative\ AI/2.4\ Classification/app.py`

### API Development

The `3-API/` directory contains a complete API server setup using FastAPI and LangServe for deploying LangChain applications.

- `app.py`: FastAPI server that exposes LangChain models and prompts via REST endpoints:

  - OpenAI model endpoint at `/openai`
  - Essay generation endpoint at `/essay` (templated prompts)
  - Poem generation endpoint at `/poem` (templated prompts)
  - LangSmith tracing integration
  - Support for both OpenAI and local Ollama models

- `client.py`: Streamlit client application that consumes the API:
  - Interactive web interface for testing API endpoints
  - Separate inputs for essay and poem generation
  - Real-time API consumption and result display

To run the API server:

1. Start the server: `python 3-API/app.py`
2. The API will be available at `http://localhost:3000`
3. Available endpoints:
   - `GET /openai`: Direct OpenAI model access
   - `POST /essay/invoke`: Generate essays with custom topics
   - `POST /poem/invoke`: Generate poems with custom topics

To run the client application:

```bash
streamlit run 3-API/client.py
```

### Agents

The `3-Agents/` directory contains examples of autonomous AI agents using LangChain's agent framework with multiple specialized tools.

- `agents.ipynb`: A comprehensive Jupyter notebook demonstrating the creation and usage of AI agents with multiple tools:
  - **Wikipedia Tool**: Query Wikipedia articles with configurable result limits
  - **Web Document Retrieval**: Load web pages and create vector stores using ChromaDB
  - **ArXiv Research Tool**: Search and retrieve academic papers from ArXiv
  - **Custom Retriever Tool**: Specialized tool for LangServe documentation queries
  - **OpenAI Functions Agent**: GPT-4o powered agent that intelligently selects and uses tools
  - **Agent Executor**: Manages agent execution with verbose reasoning output

Key Features:

- Multi-tool integration for comprehensive research capabilities
- Vector-based document retrieval for context-aware responses
- Academic paper analysis and summarization
- Intelligent tool selection based on user queries
- Real-time reasoning and tool execution tracking

Example use cases demonstrated:

- Answering questions about LangServe
- Explaining machine learning concepts
- Analyzing specific academic papers by ArXiv ID

To run the notebook:

1. Ensure you have Jupyter installed and run `jupyter notebook` in the project directory
2. The agent will use OpenAI's GPT-4o model and requires appropriate API keys
3. All tools work with LangSmith tracing for observability

## Contributing

[Add contributing guidelines if needed]

## License

[Add license information if applicable]
