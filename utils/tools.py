from langchain.agents import Tool
from langchain_experimental.tools import PythonREPLTool
from .rag import search_knowledge_base

def load_tools():
    return [
        #PythonREPLTool(),
        Tool(
            name="Knowledge Base - Query Building",
            func=lambda query: search_knowledge_base(query, tag="query"),
            description="Search for and retrieve documents tagged as 'query' from the knowledge base to help construct SQL queries."
        ),
        Tool(
            name="Knowledge Base - Specific Definitions",
            func=lambda query: search_knowledge_base(query, tag="definition"),
            description="Search for and retrieve documents tagged as 'definition' from the knowledge base to provide indicator definitions."
        )
    ]