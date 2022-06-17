CREATE TABLE PersonImages(
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    prediction TEXT,
    date TEXT,
    image BLOB
);