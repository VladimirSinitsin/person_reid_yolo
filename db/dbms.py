import sqlite3
import numpy as np

from pathlib import Path

from config import ROOTPATH
from tools import get_date_now_formatted


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


def insert_image_data(person_id: int, prediction: float, image: np.ndarray) -> None:
    """ Insert image data on DB. """
    insert({"person_id": person_id,
            "prediction": f'{prediction:.3f}',  # 0.999, 0.700, 0.101, 0.005 etc.
            "date": get_date_now_formatted(),
            "image": image})  # TODO: convert to bytes.


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
                    "image": row[4]}  # TODO: convert from bytes
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
                    "image": row[4]}  # TODO: convert from bytes
        result.append(dict_row)
    # Sorting by prediction.
    sorted_result = sorted(result, key=lambda d: float(d["prediction"]))
    return sorted_result
