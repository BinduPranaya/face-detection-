import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Try webcam, else video file
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not found, falling back to video file...")
    cap = cv2.VideoCapture("sample_video.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    out.write(frame)  # Save processed frame to video
    frame_count += 1

cap.release()
out.release()
print(f"Processed {frame_count} frames. Saved to output_video.mp4")
