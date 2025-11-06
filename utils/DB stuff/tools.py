from langchain.agents import Tool
from langchain_experimental.tools import PythonREPLTool
#from .oracle_db import get_db_connection
#from .self_check import self_check_sql
from .rag import search_knowledge_base

#db = get_db_connection()

#def safe_query_runner(query: str):
#    verdict = self_check_sql(query)
#    result = db.run(query)
#    return {"verdict": verdict, "query": query, "result": result}

def load_tools():
    return [
 #       Tool(
 #           name="SQL Database",
 #           func=safe_query_runner,
 #           description="Query the SQL database to retrieve and manipulate structured data"
 #       ),
        PythonREPLTool(),
        Tool(
            name="Knowledge Base",
            func=search_knowledge_base,
            description="Search for and retrieve documents or policies from the knowledge base containing special instructions or metrics"
        )
    ]
