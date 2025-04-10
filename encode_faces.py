import face_recognition
import os
import pickle

ENCODINGS_FILE = "face_recognition_data.pkl"
DATASET_DIR = "dataset"

known_encodings = []
known_names = []

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
        else:
            print(f"Face not found in {img_path}")

# Save encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("Face encodings saved successfully!")
