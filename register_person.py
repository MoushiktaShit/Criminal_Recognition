import os
import zipfile
import cv2
import numpy as np
import psycopg2
from insightface.app import FaceAnalysis


# ==============================
# ZIP / FOLDER PATH
# ==============================
ZIP_PATH = r"E:\Station Unattended dataset1\criminal_faces.zip"
EXTRACT_PATH = r"E:\Station Unattended dataset1\criminal_faces_extracted"

MAIN_FOLDER = os.path.join(EXTRACT_PATH, "criminal_faces")


# ==============================
# DATABASE CONNECTION
# ==============================
conn = psycopg2.connect(
    host="localhost",
    database="surveillance2_db",
    user="postgres",
    password="moushikta@123"
)

cursor = conn.cursor()


# ==============================
# CREATE TABLE
# ==============================
cursor.execute("""
CREATE TABLE IF NOT EXISTS criminal_faceperson (
    id SERIAL PRIMARY KEY,
    person_id VARCHAR(50),
    person_name VARCHAR(100),
    image_path TEXT,
    embedding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()


# ==============================
# EXTRACT ZIP
# ==============================
if not os.path.exists(EXTRACT_PATH):
    os.makedirs(EXTRACT_PATH)

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

print("Zip extracted successfully")


# ==============================
# INSIGHTFACE SETUP
# ==============================
face_app = FaceAnalysis(name="buffalo_l")

face_app.prepare(
    ctx_id=-1,
    det_size=(960, 960),
    det_thresh=0.25
)


# ==============================
# STORE EMBEDDINGS
# ==============================
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

stored_count = 0
skipped_count = 0

for person_folder in os.listdir(MAIN_FOLDER):

    person_folder_path = os.path.join(MAIN_FOLDER, person_folder)

    if not os.path.isdir(person_folder_path):
        continue

    # Example folder: CR001_BINOD
    parts = person_folder.split("_", 1)

    if len(parts) == 2:
        person_id = parts[0]
        person_name = parts[1]
    else:
        person_id = person_folder
        person_name = person_folder

    print("\nProcessing:", person_id, person_name)

    for image_name in os.listdir(person_folder_path):

        if not image_name.lower().endswith(image_extensions):
            continue

        image_path = os.path.join(person_folder_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            print("Image not readable:", image_path)
            skipped_count += 1
            continue

        faces = face_app.get(image)

        if len(faces) == 0:
            print("No face found:", image_name)
            skipped_count += 1
            continue

        # take biggest face
        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )

        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        embedding_str = ",".join(map(str, embedding))

        cursor.execute("""
            INSERT INTO criminal_faceperson
            (person_id, person_name, image_path, embedding)
            VALUES (%s, %s, %s, %s)
        """, (
            person_id,
            person_name,
            image_path,
            embedding_str
        ))

        conn.commit()

        stored_count += 1

        print("Stored:", image_name)


cursor.close()
conn.close()

print("\n================================")
print("Embedding storing completed")
print("Stored images:", stored_count)
print("Skipped images:", skipped_count)
print("================================")