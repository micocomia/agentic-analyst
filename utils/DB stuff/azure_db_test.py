import pyodbc
import os
from dotenv import load_dotenv

load_dotenv()

def check_db_connection():
    server = os.getenv("AZURE_SQL_SERVER")
    database = os.getenv("AZURE_SQL_DATABASE")
    user = os.getenv("AZURE_SQL_USER")
    password = os.getenv("AZURE_SQL_PASSWORD")
    driver = os.getenv("AZURE_SQL_DRIVER", "SQL Server")  

    # Create the connection string
    try:
        connection_string = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={user};PWD={password}'
        # Connect to the database
        with pyodbc.connect(connection_string) as conn:
            print("Successfully connected to the database.")
            # Additional operations can be performed here
    except Exception as e:
        print("Failed to connect to the database.")
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db_connection()