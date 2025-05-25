import cv2
import face_recognition
import os
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import numpy as np
face_encodings = []
face_names = []
log_file = "mismatch_log.txt"
face_folder = "faces"
outfit_folder = "outfits"

def load_faces():
    for filename in os.listdir(face_folder):
        if filename.lower().endswith((".jpg", ".png")):
            path = os.path.join(face_folder, filename)
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                face_encodings.append(encoding[0])
                face_names.append(filename.split(".")[0])

def load_outfits():
    outfit_images = []
    for filename in os.listdir(outfit_folder):
        if filename.lower().endswith((".jpg", ".png")):
            path = os.path.join(outfit_folder, filename)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (150, 150))  # daha büyük boyut
                outfit_images.append(img)
    return outfit_images

def log_mismatch(name):
    with open(log_file, "a") as f:
        f.write(f"{name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def is_outfit_match(cropped_region, outfit_images, threshold=0.45):
    if cropped_region.size == 0:
        return False
    cropped_region = cv2.resize(cropped_region, (150, 150))
    cropped_gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)

    for i, outfit_img in enumerate(outfit_images):
        outfit_gray = cv2.cvtColor(outfit_img, cv2.COLOR_BGR2GRAY)
        score = ssim(cropped_gray, outfit_gray)
        print(f"[DEBUG] Eşleşme Skoru #{i+1}: {score:.2f}")
        if score > threshold:
            return True
    return False

load_faces()
outfit_images = load_outfits()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings_current = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings_current, face_locations):
        matches = face_recognition.compare_faces(face_encodings, face_encoding)
        name = "Bilinmiyor"
        color = (0, 0, 255)

        if True in matches:
            idx = matches.index(True)
            name = face_names[idx]

        face_height = bottom - top
        y1 = bottom
        y2 = bottom + face_height * 2  # Daha geniş kıyafet alanı
        x1 = left - face_height // 2
        x2 = right + face_height // 2

        # Ekran dışına taşmayı engelle
        y1 = max(0, y1)
        y2 = min(frame.shape[0], y2)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)

        outfit_region = frame[y1:y2, x1:x2]

        matched_outfit = False
        if outfit_region.size != 0:
            matched_outfit = is_outfit_match(outfit_region, outfit_images)
            if matched_outfit:
                color = (0, 255, 0)

        if not matched_outfit and name != "Bilinmiyor":
            log_mismatch(name)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
