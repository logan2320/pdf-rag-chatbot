import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please add it to the .env file.")
    st.stop()

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF RAG Chatbot")
st.write("Ask questions based **ONLY** on the content of the PDF document.")

PDF_PATH = "data/Brochure_for_filling_application.pdf"

# --------------------------------------------------
# Load & Embed PDF (Cached)
# --------------------------------------------------
@st.cache_resource
def load_vector_db():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db

vector_db = load_vector_db()

# --------------------------------------------------
# Prompt (Strict document-only)
# --------------------------------------------------
prompt = PromptTemplate.from_template(
    """
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present in the document, say:
"Answer not found in the document."

Context:
{context}

Question:
{input}
"""
)

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# --------------------------------------------------
# RAG Chain (Modern API)
# --------------------------------------------------
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

retrieval_chain = create_retrieval_chain(
    retriever=vector_db.as_retriever(search_kwargs={"k": 6}),
    combine_docs_chain=document_chain
)

# --------------------------------------------------
# UI - Ask Question
# --------------------------------------------------
question = st.text_input("Ask a question from the PDF:")

if question:
    with st.spinner("Searching the document..."):
        response = retrieval_chain.invoke({"input": question})

        st.subheader("Answer")
        st.write(response["answer"])

