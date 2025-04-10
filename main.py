import face_recognition
import cv2
import os
import pickle
import numpy as np

ENCODINGS_FILE = "face_recognition_data.pkl"
TEST_IMAGES_DIR = "test_images"

# Load encodings
with open(ENCODINGS_FILE, "rb") as f:
    known_encodings, known_names = pickle.load(f)

images_to_show = []

for test_img_name in os.listdir(TEST_IMAGES_DIR):
    img_path = os.path.join(TEST_IMAGES_DIR, test_img_name)
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Convert to BGR for OpenCV display
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    
    # Resize image to a standard height to combine easily (optional)
    image_resized = cv2.resize(image_bgr, (400, int(image_bgr.shape[0] * 400 / image_bgr.shape[1])))
    images_to_show.append(image_resized)

# Combine images vertically if there's more than one
if images_to_show:
    collage = np.vstack(images_to_show)
    cv2.imshow("Collage of Test Images", collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No images to show.")
