import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------
# ENV SETUP
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env")
    st.stop()

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 Retailer Outlet Chatbot")
st.write("Ask questions based **ONLY** on the content of the PDF document.")

# ---------------------------
# LOAD PDF (STATIC FOR NOW)
# ---------------------------
PDF_PATH = "data/Brochure_for_filling_application.pdf"  # <-- put your PDF here

@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

vectorstore = load_vectorstore()

# ---------------------------
# RETRIEVER (SEMANTIC + FRIENDLY)
# ---------------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "lambda_mult": 0.8}
)

# ---------------------------
# PROMPT (STRICT BUT NATURAL)
# ---------------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions strictly based on the provided document context.

Rules:
- Use ONLY the information from the context.
- You may paraphrase or summarize the content.
- Do NOT add external knowledge.
- If the answer cannot be inferred from the context, say:
  "Answer not found in the document."

Context:
{context}

Question:
{input}

Answer:
""")

# ---------------------------
# LLM
# ---------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ---------------------------
# CHAINS (MODERN LANGCHAIN)
# ---------------------------
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# ---------------------------
# USER INPUT
# ---------------------------
query = st.text_input("Ask a question from the PDF:")

if query:
    with st.spinner("Searching the document..."):
        response = retrieval_chain.invoke({"input": query})

    st.subheader("Answer")
    st.write(response["answer"])
