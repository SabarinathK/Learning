# from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma


# load_dotenv(override=True)

# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# # pdf extract
# loader = PyPDFLoader("company.pdf")
# docs = loader.load()

# # chunking
# data_split = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
# all_split = data_split.split_documents(docs)
# print(all_split)

# engine=Chroma(embedding_function=embeddings,
#               collection_name="PDF_Rag",
#               persist_directory="chroma.db")

# engine.add_documents(all_split)


from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader

loader = OpenDataLoaderPDFLoader(
    file_path=["company.pdf"], format="text"
)
documents = loader.load()

for doc in documents:
    print(doc.metadata, doc.page_content[:80])
