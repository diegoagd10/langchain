import streamlit as st
import validators
import tempfile
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader, PyPDFLoader

load_dotenv()

## Streamlit App
st.set_page_config(page_title="🦜 Summarization Showcase", page_icon="🦜")
st.title("🦜 Summarization Showcase")
st.subheader("Summarize Multiple Sources: PDFs, YouTube, Websites")

groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in your .env file. 🔑")
    st.stop()

## File Uploader for PDFs
uploaded_files = st.file_uploader("📄 Upload PDF files", type="pdf", accept_multiple_files=True)

## Dynamic URL Inputs
if 'urls' not in st.session_state:
    st.session_state.urls = ['']

st.write("### 🌐 Add URLs (YouTube or Websites)")
for i, url in enumerate(st.session_state.urls):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.session_state.urls[i] = st.text_input(f"URL {i+1}", value=url, key=f"url_{i}", placeholder="https://...")
    with col2:
        if st.button("❌ Remove", key=f"remove_{i}") and len(st.session_state.urls) > 1:
            st.session_state.urls.pop(i)
            st.rerun()

if st.button("➕ Add URL"):
    st.session_state.urls.append('')
    st.rerun()

## Summarize Button
if st.button("🚀 Summarize All Sources"):
    ## Validate Inputs
    if not uploaded_files and not any(url.strip() for url in st.session_state.urls):
        st.error("Please upload PDFs or add at least one URL! 📚")
    else:
        try:
            all_docs = []
            with st.spinner("Loading and processing sources... 🌟"):
                ## Load PDFs
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                    os.unlink(temp_path)  # Clean up

                ## Load URLs
                for url in st.session_state.urls:
                    if url.strip():
                        if not validators.url(url):
                            st.error(f"Invalid URL: {url}")
                            continue
                        if "youtube.com" in url or "youtu.be" in url:
                            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                        else:
                            loader = UnstructuredURLLoader(
                                urls=[url],
                                ssl_verify=False,
                                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                            )
                        docs = loader.load()
                        all_docs.extend(docs)

            if not all_docs:
                st.error("No content could be loaded from the provided sources. 😔")
            else:
                ## LLM and Chain
                llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
                prompt_template = """
                Provide a comprehensive summary of the following content in 300 words:
                Content: {text}
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

                with st.spinner("Summarizing... 🤖"):
                    output_summary = chain.run(all_docs)

                st.success("Summary Ready! 🎉")
                st.write("### Summary:")
                st.write(output_summary)

        except Exception as e:
            st.exception(f"Oops! An error occurred: {e} 😵")