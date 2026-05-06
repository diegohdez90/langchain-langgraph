"""
Module: main.py
Description: This module demonstrates how to create
a simple agent using LangChain that can perform a search
"""


from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic as Anthropic
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Search for the given query and return the results.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """

    print(f"Searching for: {query}")
    return tavily.search(query=query, limit=5)


llm = Anthropic(model="claude-2", temperature=0.7)
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-langgraph!")
    result = agent.invoke(
        {
            "messaages": [
                HumanMessage(
                    content="What is the weather in Mexico City today?"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
