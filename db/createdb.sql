CREATE TABLE PersonImages(
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    date TEXT,
    image BLOB
);