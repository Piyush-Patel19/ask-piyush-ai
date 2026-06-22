from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("Loading AI...")

# VECTOR DB
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# MODEL
llm = init_chat_model(
    "openai/gpt-oss-120b:free",
    model_provider="openrouter",
    temperature=0.3,
)

# PROMPT
prompt_template = ChatPromptTemplate.from_template("""
You are the AI version of Piyush Patel.

Act as a portfolio assistant that helps visitors learn about my background,
projects, skills, interests, and experience.

STRICT RULES:

- Answer ONLY using the provided context.
- Never use outside knowledge.
- Never guess.
- If the answer is not present, respond exactly:

"I don't currently have that information in my portfolio data."

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
            doc.page_content for doc in docs
        )),
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

print("\nAsk Piyush AI")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    try:
        answer = chain.invoke(question)
        print("\nAI:", answer)
        print()

    except Exception as e:
        print(f"\nError: {e}\n")