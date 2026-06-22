import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os

os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
import subprocess
import sys
from pathlib import Path

if not Path("chroma_db").exists():
    subprocess.run([sys.executable, "ingest.py"], check=True)


# PAGE 


st.set_page_config(
    page_title="Ask Piyush AI",
    page_icon="🤖"
)

# HEADER


st.title("🤖 Ask Piyush AI")
st.caption(
    "Ask anything about my skills, projects, interests and experience."
)

# SIDEBAR


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

    st.divider()


# VECTOR DB


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    collection_name="piyush_portfolio",
    persist_directory="./chroma_db",
    embedding_function=embedding
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


llm = ChatOpenRouter(
    model="openai/gpt-oss-120b:free",
    temperature=0.3,
    api_key=st.secrets["OPENROUTER_API_KEY"],
)


prompt_template = ChatPromptTemplate.from_template("""
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


# CHAIN


chain = (
    {
        "context": retriever
        | (lambda docs: "\n\n".join(
            doc.page_content
            for doc in docs
        )),
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)


question = st.chat_input(
    "Ask me anything..."
)


# PROCESS QUESTION

if question:

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer = chain.invoke(question)

                st.markdown(answer)

            except Exception as e:

                answer = f"Error: {str(e)}"

                st.error(answer)
