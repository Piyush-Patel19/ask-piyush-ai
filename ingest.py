# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# loader = TextLoader("Resume.txt")
# docs = loader.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=50
# )

# chunks = splitter.split_documents(docs)

# embedding = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# db = Chroma.from_documents(
#     documents=chunks,
#     embedding=embedding,
#     collection_name="piyush_portfolio",
#     persist_directory="./chroma_db"
# )

# print("Data stored successfully")
