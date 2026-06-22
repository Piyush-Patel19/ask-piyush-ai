import os
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
os.environ["ANONYMIZED_TELEMETRY"] = "False"

BASE_DIR = Path(__file__).resolve().parent
RESUME_PATH = BASE_DIR / "Resume.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "piyush_portfolio"

st.set_page_config(page_title="Ask Piyush AI", page_icon="🤖")

st.title("🤖 Ask Piyush AI")
st.caption("Ask anything about my skills, projects, interests and experience.")

with st.sidebar:
    st.title("Piyush Patel")
    st.write("Computer Science Student")
    st.divider()
    st.markdown("""
### Skills
- Python
- LangChain
- ChromaDB
- Streamlit
- RAG
- FastAPI
- NumPy
- Pandas
""")

@st.cache_resource
def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_db():
    embedding = get_embedding()

    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding
    )

    if db._collection.count() == 0:
        loader = TextLoader(str(RESUME_PATH), encoding="utf-8")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        db.add_documents(chunks)

    return db

@st.cache_resource
def get_llm():
    return ChatOpenRouter(
        model="openai/gpt-oss-120b:free",
        temperature=0.3,
        api_key=st.secrets["OPENROUTER_API_KEY"],
    )

db = get_db()
retriever = db.as_retriever(search_kwargs={"k": 5})
llm = get_llm()

prompt = ChatPromptTemplate.from_template("""
You are the AI version of Piyush Patel.

Act as a portfolio assistant that helps visitors learn about my background, projects, skills, interests, and goals.

STRICT RULES:
- Answer ONLY using the provided context.
- Never use outside knowledge.
- Never guess, infer, assume, or make up information.
- If the answer is not explicitly present in the context, respond exactly:
"I don't currently have that information in my portfolio data."
- Speak in first person ("I", "my", "me").
- Be professional and concise.
- When listing skills, projects, or technologies, use bullet points.

Context:
{context}

Question:
{question}

Answer:
""")

chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

question = st.chat_input("Ask me anything...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                docs = retriever.invoke(question)
                st.write("Debug docs found:", len(docs))
                answer = chain.invoke(question)
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {e}")
