from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv(override=True)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

engine = Chroma(
    embedding_function=embeddings,
    collection_name="PDF_Rag",
    persist_directory="chroma.db",
)
