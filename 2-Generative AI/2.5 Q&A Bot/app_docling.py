import streamlit as st
import os
import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_docling.loader import ExportType
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from dotenv import load_dotenv


load_dotenv()

## Load the GROQ and OpenAI API KEY
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

# Validate API keys
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Please add it to your .env file.")
    st.stop()

st.title("Chatgroq With Llama3 Demo")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama-3.3-70b-versatile")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():
    """Create vector embeddings from PDF documents"""

    if "vectors" not in st.session_state:
        try:
            # Check if docs directory exists
            docs_dir = "./docs/Diego Autism Diagnosis.pdf"

            with st.spinner("Loading and processing documents..."):
                embeddings = OpenAIEmbeddings()
                
                tokenizer = OpenAITokenizer(
                    tokenizer=tiktoken.encoding_for_model(embeddings.model),
                    max_tokens=embeddings.dimensions or 0
                )

                st.session_state.loader = DoclingLoader(
                    file_path=docs_dir,
                    export_type=ExportType.MARKDOWN,
                    chunker=HybridChunker(tokenizer=tokenizer),
                )
                st.session_state.docs = st.session_state.loader.load()  # Document Loading

                if not st.session_state.docs:
                    st.error("No documents could be loaded. Please check your PDF files.")
                    return

                splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[
                        ("#", "Header_1"),
                        ("##", "Header_2"),
                        ("###", "Header_3"),
                    ],
                )

                st.session_state.final_documents = [split for doc in st.session_state.docs for split in splitter.split_text(doc.page_content)]

                # Use a persistent Chroma client with collection management
                import chromadb
                persist_directory = "./chroma_db"

                # Initialize Chroma client
                chroma_client = chromadb.PersistentClient(path=persist_directory)

                # Delete existing collection if it exists to avoid dimension mismatch
                try:
                    chroma_client.delete_collection("docling_collection")
                except:
                    pass  # Collection doesn't exist, continue


                # Create Chroma vectorstore with the new collection
                st.session_state.vectors = Chroma(
                    client=chroma_client,
                    collection_name="docling_collection",
                    embedding_function=embeddings
                )

                # Add documents to the collection
                st.session_state.vectors.add_documents(st.session_state.final_documents)

                st.success(f"Successfully processed {len(st.session_state.docs)} documents into {len(st.session_state.final_documents)} chunks!")

        except Exception as e:
            st.error(f"Error during document processing: {str(e)}")
            if "vectors" in st.session_state:
                del st.session_state.vectors





# Input section
prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()

# Question and answer section
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Documents Embedding' first to process the documents.")
    else:
        try:
            with st.spinner("Generating answer..."):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                import time
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start

                print(f"Response time: {response_time}")

                # Display the answer
                st.subheader("Answer:")
                st.write(response['answer'])

                # Display processing time
                st.caption(f"Response generated in {response_time:.2f} seconds")

                # Document similarity search results
                with st.expander("Document Similarity Search"):
                    st.write("Relevant document chunks:")
                    for i, doc in enumerate(response["context"]):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(doc.page_content)
                        st.write("---")

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")