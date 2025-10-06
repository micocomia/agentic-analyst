import oracledb
import os

load_dotenv()
oracledb.init_oracle_client()

user = os.getenv("ORACLE_USER")
password = os.getenv("ORACLE_PASSWORD")
dsn = os.getenv("ORACLE_TNS_ALIAS")

try:
    connection = oracledb.connect(user=user, password=password, dsn=dsn)
    print('Connected successfully')
    
    cursor = connection.cursor()
    cursor.execute("SELECT agg_date, total_advertised FROM adm_tm_ops_pi_advertised_di")
    for row in cursor:
        print("Output:", row[0])
        
except Exception as e:
    print("Connection failed", e)
    
finally:
    try:
        cursor.close()
        connection.close()
    except:
        pass