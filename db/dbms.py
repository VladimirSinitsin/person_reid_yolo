import sqlite3

from pathlib import Path
from typing import Dict, List

from config import ROOTPATH


# Connect to DB.
db_path = Path.joinpath(ROOTPATH, "db/person_images.db")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()


# Check if the database was not created.
cursor.execute("SELECT name FROM sqlite_master "
               "WHERE type='table' AND name='PersonImages'")
table_exists = cursor.fetchall()
if not table_exists:
    with open(Path.joinpath(ROOTPATH, "db/createdb.sql"), "r") as f:
        sql = f.read()
    cursor.executescript(sql)
    conn.commit()
