from config import engine
from langchain.tools import tool
import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent


@tool
def knowledge_base(query):
    "knowledge base for the user questions"
    retrived_doc = engine.similarity_search_with_relevance_scores(query=query, k=3)
    return retrived_doc


model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

prompt = """your are support assistant ,
        your name is sam
        if u need to any knowledge source use the tool for it
        make the answer simple and crisp"""


agent = create_agent(
    name="Support Assistant",
    model=model,
    system_prompt=prompt,
    tools=[knowledge_base],
)

response = agent.invoke({"messages": "what is the company name and phone number"})
print(response["messages"][-1].content)
