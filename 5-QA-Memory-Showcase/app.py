import streamlit as st
import os
import tempfile
import sqlite3
import uuid
import json
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

st.title("🦜 Q&A Memory Showcase")
st.subheader("Chat with your documents and remember the conversation!")

# User and Session Setup
user = st.text_input("Enter your user ID", value="default_user", key="user")
session_file = f"{user}_session.txt"
if os.path.exists(session_file):
    with open(session_file, 'r') as f:
        session_id = f.read().strip()
    st.session_state.session_id = session_id
else:
    session_id = str(uuid.uuid4())
    with open(session_file, 'w') as f:
        f.write(session_id)
    st.session_state.session_id = session_id

# Database Setup
conn = sqlite3.connect('chat_history.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY,
    user TEXT,
    session_id TEXT,
    question TEXT,
    answer TEXT,
    context TEXT
)''')
conn.commit()

# Load Chat History
cursor.execute('SELECT question, answer, context FROM history WHERE user = ? AND session_id = ?', (user, session_id))
rows = cursor.fetchall()
st.session_state.chat_history = []
for row in rows:
    context_data = json.loads(row[2]) if row[2] else []
    st.session_state.chat_history.append({"question": row[0], "answer": row[1], "context": context_data})

# API Keys
groq_api_key = os.getenv('GROQ_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in your .env file. 🔑")
    st.stop()

if not openai_api_key:
    st.error("OPENAI_API_KEY not found. Please set it in your .env file. 🔑")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Load or prepare vectors
embeddings = OpenAIEmbeddings()
persist_dir = "./chroma_db"
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    try:
        vectors = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        st.session_state.vectors = vectors
        st.success("Loaded existing embeddings! 🎉")
    except Exception as e:
        st.error(f"Error loading existing embeddings: {e}")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<chat_history>
{chat_history}
</chat_history>
<context>
{context}
</context>
Questions:{input}
"""
)

# File Uploader for PDFs
uploaded_files = st.file_uploader("📄 Upload PDF documents", type="pdf", accept_multiple_files=True)

if st.button("🚀 Embed Documents"):
    if not uploaded_files:
        st.error("Please upload some PDF files first! 📚")
    else:
        try:
            with st.spinner("Processing and embedding documents... 🌟"):
                all_docs = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                    os.unlink(temp_path)  # Clean up

                if not all_docs:
                    st.error("No documents could be loaded. 😔")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    final_documents = text_splitter.split_documents(all_docs)
                    vectors = Chroma.from_documents(final_documents, embeddings, persist_directory=persist_dir)
                    st.session_state.vectors = vectors
                    st.session_state.final_documents = final_documents
                    st.success(f"Successfully embedded {len(final_documents)} chunks! 🎉")

        except Exception as e:
            st.error(f"Error during embedding: {str(e)} 😵")

# Question Input
question = st.text_input("💬 Ask a question about the documents", placeholder="What's the main topic?")

if question:
    if 'vectors' not in st.session_state:
        st.warning("Please embed documents first by clicking 'Embed Documents'! 📚")
    else:
        try:
            with st.spinner("Generating answer... 🤖"):
                chat_history_str = "\n".join([f"Q: {e['question']}\nA: {e['answer']}" for e in st.session_state.chat_history])
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                response = retrieval_chain.invoke({'input': question, 'chat_history': chat_history_str})
                answer = response['answer']
                context = response["context"]

            # Save to database
            context_json = json.dumps([{"content": doc.page_content, "metadata": doc.metadata} for doc in context])
            cursor.execute('INSERT INTO history (user, session_id, question, answer, context) VALUES (?, ?, ?, ?, ?)', (user, session_id, question, answer, context_json))
            conn.commit()

            # Add to session state
            st.session_state.chat_history.append({"question": question, "answer": answer, "context": context})

            # Display latest answer
            st.subheader("Latest Answer:")
            st.write(answer)

            # Relevant chunks
            with st.expander("Relevant Document Chunks 📄"):
                for i, doc in enumerate(context):
                    st.write(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---")

        except Exception as e:
            st.error(f"Error generating answer: {str(e)} 😵")

# Display Chat History
if st.session_state.chat_history:
    st.subheader("Chat History 📜")
    for entry in reversed(st.session_state.chat_history):
        st.write(f"**Q:** {entry['question']}")
        st.write(f"**A:** {entry['answer']}")
        st.write("---")

# Close database connection
conn.close()