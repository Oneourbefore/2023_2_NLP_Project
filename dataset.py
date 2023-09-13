import pymysql
from dbconfig import Settings

settings = Settings()

db_host = settings.db_host
db_port = settings.db_port
db_user = settings.db_user
db_password = settings.db_password
db_db = settings.db_db

class MysqlConnection :
    def __init__(self):
        self.connection = pymysql.connect(
            host = db_host,
            port = db_port,
            database = db_db,
            user = db_user,
            password = db_password
        )
