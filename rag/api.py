from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv(override=True)

app = FastAPI(title="Ammakase Support API")

# ------------------------
# Embeddings
# ------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# ------------------------
# Vector Store
# ------------------------
engine = Chroma(
    collection_name="PDF_Rag",
    embedding_function=embeddings,
    persist_directory="chroma.db",
)


# ------------------------
# Tool
# ------------------------
@tool
def knowledge_base(query: str):
    """Search Ammakase knowledge base"""
    docs = engine.similarity_search(query=query, k=3)

    return "\n\n".join([doc.page_content for doc in docs])


# ------------------------
# LLM
# ------------------------
model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
)

# ------------------------
# Agent
# ------------------------
prompt = """
You are Sam, a customer support assistant for Ammakase Singapore.

Rules:
- Answer only based on the knowledge base.
- If information is unavailable, say:
  'I could not find that information in our knowledge base.'
- Be concise and professional.
"""

agent = create_agent(
    name="Support Assistant",
    model=model,
    system_prompt=prompt,
    tools=[knowledge_base],
)


# ------------------------
# Request Schema
# ------------------------
class ChatRequest(BaseModel):
    question: str


# ------------------------
# API Endpoint
# ------------------------
@app.post("/answer")
async def answer_question(request: ChatRequest):
    try:
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": request.question,
                    }
                ]
            }
        )

        answer = response["messages"][-1].content

        return {
            "success": True,
            "question": request.question,
            "answer": answer,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ------------------------
# Health Check
# ------------------------
@app.get("/health")
async def health():
    return {"status": "healthy"}
