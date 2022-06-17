CREATE TABLE PersonImages(
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    prediction REAL,
    data TEXT,
    image BLOB
);