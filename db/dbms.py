import sqlite3
import numpy as np

from pathlib import Path

from config import ROOTPATH
from config import RECREATE_DB
from tools import bytes_to_image
from tools import get_date_now_formatted


# DBMS debugging.
if RECREATE_DB and Path.joinpath(ROOTPATH, "db/person_images.db").exists():
    Path.joinpath(ROOTPATH, "db/person_images.db").unlink()

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


def insert_image_data(person_id: int, image: np.ndarray) -> None:
    """ Insert image data on DB. """
    insert({"person_id": person_id,
            "date": get_date_now_formatted(),
            "image": image.tobytes()})


def insert(column_values: dict) -> None:
    """
    Insert values in DB.
    :param column_values: dictionary with columns(keys) and values.
    """
    columns = ", ".join(column_values.keys())
    values = [tuple(column_values.values())]
    placeholders = ", ".join("?" * len(column_values.keys()))
    cursor.executemany(f"INSERT INTO PersonImages ({columns}) VALUES ({placeholders})", values)
    conn.commit()


def select_all_images_data() -> list:
    """ Get all images data from DB. """
    cursor.execute(f"SELECT * FROM PersonImages")
    rows = cursor.fetchall()
    result = []
    for row in rows:
        dict_row = {"person_id": row[1],
                    "prediction": row[2],
                    "date": row[3],
                    "image": bytes_to_image(row[4])}
        result.append(dict_row)
    return result


def select_curr_images_data(person_id: int) -> list:
    """
    Get current images data from DB.
    :param person_id: current person id.
    :return: list with images data.
    """
    cursor.execute(f"SELECT * FROM PersonImages WHERE person_id={person_id}")
    rows = cursor.fetchall()
    result = []
    for row in rows:
        dict_row = {"person_id": row[1],
                    "prediction": row[2],
                    "date": row[3],
                    "image": bytes_to_image(row[4])}
        result.append(dict_row)
    # Sorting by prediction.
    sorted_result = sorted(result, key=lambda d: float(d["prediction"]))
    return sorted_result


def select_max_person_id() -> int:
    """ Select max value of person_id column. """
    cursor.execute(f"SELECT person_id FROM PersonImages")
    rows = np.array(cursor.fetchall())
    return np.max(rows)
