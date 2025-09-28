import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from Classification import Classification
from DetailedClassification import DetailedClassification

load_dotenv() # Load all environment variables

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGSMITH_API_KEY' ] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
os.environ['LANGSMITH_WORKSPACE_ID'] = os.getenv("LANGSMITH_WORKSPACE_ID")

groq_api_key = os.environ['GROQ_API_KEY']

st.title("Classification Demo")

llm = ChatGroq(temperature=0,model = "gemma2-9b-it", groq_api_key = groq_api_key)

## Structured LLM

structured_llm = llm.with_structured_output(Classification)
detailed_structued_llm = llm.with_structured_output(DetailedClassification)

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

detailed_tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'DetailedClassification' function.

Passage:
{input}
"""
)

prompt = st.text_input("Input your prompt here")

if prompt:
    final_prompt = tagging_prompt.invoke({ "input": prompt})
    response = structured_llm.invoke(final_prompt)
    st.write(f"Sentiment: {response.sentiment}")
    st.write(f"Aggressiveness: {response.agreessiveness}")
    st.write(f"Language: {response.language}")
    st.write("--------------------------------")

    final_detailed_prompt = detailed_tagging_prompt.invoke({ "input": prompt })
    detailed_response = detailed_structued_llm.invoke(final_detailed_prompt)
    st.write(f"Sentiment: {detailed_response.sentiment}")
    st.write(f"Aggressiveness: {detailed_response.agreessiveness}")
    st.write(f"Language: {detailed_response.language}")
    st.write("--------------------------------")
