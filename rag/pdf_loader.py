from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import engine

load_dotenv(override=True)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# pdf extract
loader = PyPDFLoader("company.pdf")
docs = loader.load()

# chunking
data_split = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
all_split = data_split.split_documents(docs)


# added to the chromadb
engine.add_documents(all_split)


