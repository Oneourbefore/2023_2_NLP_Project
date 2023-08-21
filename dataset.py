## dbconfig 사용 예시
import mysql.connector
from dbconfig import Settings

settings = Settings()

db_host = settings.db_host
db_port = settings.db_port
db_user = settings.db_user
db_password = settings.db_password
db_db = settings.db_db

connection = mysql.connector.connect(
    host = db_host,
    port = db_port,   
    database = db_db,  
    user = db_user,
    password = db_password
)
