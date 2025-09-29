import os
import oracledb
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase

load_dotenv()

oracledb.init_oracle_client()

def get_db_connection():
    user = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    tns_alias = os.getenv("ORACLE_TNS_ALIAS")
    
    uri = f"oracle+oracledb://{user}:{password}@{tns_alias}"
    
    return SQLDatabase.from_uri(uri)