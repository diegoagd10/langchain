import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_API_KEY' ] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_PROJECT'] = "GenApplicationWithOllama"
os.environ['LANGSMITH_WORKSPACE_ID'] = os.getenv("LANGSMITH_WORKSPACE_ID")

## Prompt Template
prompt = ChatPromptTemplate.from_messages([
   ("system", "You are a helpful assistant. Please respond to the question asked"),
   ("user", "Question: {question}"),
])

## Streamlit Framework
st.title("Langchain Demo With Google Gemma Model")
input_text = st.text_input("What question do you have in mind?")

## Ollama Gemma Model
llm = OllamaLLM(model="gemma3:1b")

## Calling LLM with Langchain
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({ "question": input_text}))
