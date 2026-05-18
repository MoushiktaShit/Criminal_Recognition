import cv2
import numpy as np
import psycopg2
import warnings
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore", message="`rcond` parameter will change")

# ==========================================
# PostgreSQL Connection & Load Data
# ==========================================
conn = psycopg2.connect(
    host="localhost", database="surveillance_db", user="postgres", password="moushikta@123"
)
cursor = conn.cursor()
cursor.execute("SELECT person_name, embedding FROM known_persons")
rows = cursor.fetchall()
known_embeddings = [(row[0], np.array(list(map(float, row[1].split(","))))) for row in rows]
print("Known persons loaded:", len(known_embeddings))

model = YOLO("yolov8n.pt")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.4) # Slightly lower thresh for video

video_path = r"E:\Station Unattended dataset\12752510_3840_2160_60fps.mp4"
cap = cv2.VideoCapture(video_path)

threshold = 0.55
track_name_memory = {}
detection_logs = [] # List to store dicts for the HUD

def draw_hud(frame, logs):
    """Draws a transparent black HUD with the latest detection logs."""
    if not logs:
        return frame
        
    hud_width = 400
    hud_height = 50 + (len(logs) * 30)
    
    # Create transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_width, 10 + hud_height), (0, 0, 0), -1)
    
    # Blend overlay with original frame (0.6 alpha for transparency)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Add HUD Header
    cv2.putText(frame, "SUSPICIOUS ACTIVITY LOG", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.line(frame, (20, 45), (390, 45), (0, 0, 255), 1)
    
    # Add Log Entries
    y_offset = 70
    for log in reversed(logs[-5:]): # Show only last 5 entries
        text = f"[{log['time']}] {log['name']} Detected (ID:{log['id']})"
        cv2.putText(frame, text, (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 30
        
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            # Default to tracking memory if it exists
            name = track_name_memory.get(track_id, "Unknown")

            # Only attempt new face recognition if we haven't locked in an identity yet
            # OR if you want it to continually verify, remove the `if name == "Unknown":`
            if name == "Unknown":
                faces = face_app.get(person_crop)

                if len(faces) > 0:
                    test_embedding = faces[0].embedding
                    best_similarity = -1
                    best_match_name = "Unknown"

                    for person_name, db_embedding in known_embeddings:
                        similarity = 1 - cosine(test_embedding, db_embedding)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_name = person_name

                    if best_similarity >= threshold:
                        name = best_match_name
                        track_name_memory[track_id] = name
                        
                        # Add to HUD log when a new person is positively identified
                        current_time = datetime.now().strftime("%H:%M:%S")
                        detection_logs.append({"name": name, "time": current_time, "id": track_id})

            # Draw bounding box and text
            color = (0, 255, 255) if name == "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"ID:{track_id} | {name}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw HUD before resizing so text stays crisp
    frame = draw_hud(frame, detection_logs)
    frame = cv2.resize(frame, (1280, 720))

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()