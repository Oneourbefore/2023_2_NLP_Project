# pip install python-dotenv
from dotenv import load_dotenv
import os

load_dotenv() # take environment variables from .env.

# database
DB_DB = os.getenv("DB_DB")
DB_HOST = os.getenv("DB_HOST")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DB_PORT = int(os.getenv("DB_PORT")) # 테스트용
DB_USER = os.getenv("DB_USER")


class Settings():
    db_db: str = DB_DB
    db_host: str = DB_HOST
    db_password: str = DB_PASSWORD
    db_port: int = DB_PORT
    db_user: str = DB_USER