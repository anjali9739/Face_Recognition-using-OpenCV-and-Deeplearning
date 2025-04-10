import cv2
import face_recognition

# Try different indexes if 0 doesn't work
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video stream from the webcam. Trying index 1...")
    video_capture = cv2.VideoCapture(1)
    if not video_capture.isOpened():
        print("Error: Could not open video stream from the webcam at index 1 either.")
        exit()

print("Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame from the camera. Exiting loop.")
        break

    # Process frame for face detection (if needed)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    print("Faces found:", len(face_locations))
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Webcam Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
