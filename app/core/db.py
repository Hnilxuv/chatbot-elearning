import mysql.connector
from app.core.settings import DATABASE_CONFIG

def get_db_connection():
    return mysql.connector.connect(**DATABASE_CONFIG)
