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
    model=os.getenv("GROQ_MODEL"),
    api_key=os.getenv("GROQ_API_KEY"),
)

prompt = """your are support assistant as Ammakase Singapore,
        your name is sam
        if u need to any knowledge source use the tool for it"""


agent = create_agent(
    name="Support Assistant",
    model=model,
    system_prompt=prompt,
    tools=[knowledge_base],
)

response = agent.invoke({"messages": "any dress code is there?"})
print(response["messages"][-1].content)
