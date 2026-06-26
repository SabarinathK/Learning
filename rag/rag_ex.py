from langchain_groq import ChatGroq
import os
from config import engine

def knowledge_base(query):
    "knowledge base for the user questions"
    retrived_doc = engine.similarity_search(query=query, k=3)
    return retrived_doc


model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)


def chatbot(user_question):
    context=knowledge_base(user_question)
    prompt =f"""your are support assistant,
            your name is sam
            answer precise and answer like a support system

            her is the user question:
            {user_question}

            and here is the context you have for the question:
            {context}

            """
    resposne=model.invoke(prompt)
    return resposne

result=chatbot(user_question="what is the name of the company")
print(result.content)