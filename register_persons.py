import os
import cv2
import psycopg2
from insightface.app import FaceAnalysis

DB_NAME = "surveillance2_db"
DB_USER = "postgres"
DB_PASS = "moushikta@123"
DB_HOST = "localhost"

MAIN_FOLDER = "criminal_faces"

conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS known_persons (
    id SERIAL PRIMARY KEY,
    person_id VARCHAR(50),
    person_name VARCHAR(100),
    image_name TEXT,
    image_path TEXT,
    embedding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
conn.commit()

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(1280, 1280), det_thresh=0.20)

if not os.path.exists(MAIN_FOLDER):
    print("❌ Folder not found:", os.path.abspath(MAIN_FOLDER))
    exit()

stored = 0
failed = 0

for folder_name in os.listdir(MAIN_FOLDER):
    folder_path = os.path.join(MAIN_FOLDER, folder_name)

    if not os.path.isdir(folder_path):
        continue

    parts = folder_name.split("_", 1)

    if len(parts) == 2:
        person_id = parts[0]
        person_name = parts[1]
    else:
        person_id = folder_name
        person_name = folder_name

    print("\nPerson:", person_id, person_name)

    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print("❌ Cannot read:", image_name)
            failed += 1
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print("❌ No face found:", image_name)
            failed += 1
            continue

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        embedding_str = ",".join(map(str, face.embedding))

        try:
            cursor.execute("""
                INSERT INTO known_persons
                (person_id, person_name, image_name, image_path, embedding)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                person_id,
                person_name,
                image_name,
                image_path,
                embedding_str
            ))

            conn.commit()
            stored += 1
            print("✅ Stored:", image_name)

        except Exception as e:
            conn.rollback()
            failed += 1
            print("❌ DB insert failed:", image_name)
            print("Error:", e)

cursor.execute("SELECT COUNT(*) FROM known_persons;")
total = cursor.fetchone()[0]

cursor.close()
conn.close()

print("\n==============================")
print("Stored now:", stored)
print("Failed:", failed)
print("Total rows in PostgreSQL:", total)
print("==============================")