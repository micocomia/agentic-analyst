import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase

load_dotenv()

def get_db_connection():
    server = os.getenv("AZURE_SQL_SERVER")
    database = os.getenv("AZURE_SQL_DATABASE")
    user = os.getenv("AZURE_SQL_USER")
    password = os.getenv("AZURE_SQL_PASSWORD")
    driver = os.getenv("AZURE_SQL_DRIVER", "ODBC Driver 17 for SQL Server")
    uri = f"mssql+pyodbc://{user}:{password}@{server}/{database}?driver={driver.replace(' ', '+')}"
    return SQLDatabase.from_uri(uri)
