import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma

load_dotenv() # Load all environment variables

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGSMITH_API_KEY' ] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
os.environ['LANGSMITH_WORKSPACE_ID'] = os.getenv("LANGSMITH_WORKSPACE_ID")

groq_api_key = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    st.session_state.loader = WebBaseLoader("https://www.vatican.va/content/francesco/en/encyclicals/documents/papa-francesco_20130629_enciclica-lumen-fidei.html")
    st.session_state.documents = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
    st.session_state.vector_db = Chroma.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("ChatGroq Demo")

llm = ChatGroq(model = "openai/gpt-oss-20b", groq_api_key = groq_api_key)

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
<context>

Question:{input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vector_db.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here")

if prompt:
    response = retriever_chain.invoke({ "input": prompt})
    st.write(response['answer'])
    st.write("--------------------------------")