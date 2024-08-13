import cv2
from deepface import DeepFace
import os
import numpy as np

def load_known_faces(known_faces_dir='known_faces'):
    known_face_embeddings = []
    known_face_names = []

    for file_name in os.listdir(known_faces_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(known_faces_dir, file_name)
            try:
                embedding = DeepFace.represent(img_path=file_path, model_name='Facenet', enforce_detection=False)
                if embedding:
                    known_face_embeddings.append(np.array(embedding[0]['embedding']))
                    known_face_names.append(os.path.splitext(file_name)[0])
                else:
                    print(f"No embedding found for {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return known_face_embeddings, known_face_names

def start_recognition():
    known_face_embeddings, known_face_names = load_known_faces()
    webcam = cv2.VideoCapture(0)

    while True:
        success, frame = webcam.read()

        try:
            detected_faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False, detector_backend='opencv')
            if not detected_faces:
                print("No faces detected")
                continue

            for face in detected_faces:
                facial_area = face['facial_area']
                try:
                    detected_embedding = np.array(
                        DeepFace.represent(img_path=face['face'], model_name='Facenet', enforce_detection=False)[0]['embedding'])

                    best_match_name = None
                    smallest_distance = float("inf")

                    for known_embedding, name in zip(known_face_embeddings, known_face_names):
                        distance = np.linalg.norm(known_embedding - detected_embedding)
                        if distance < smallest_distance:
                            smallest_distance = distance
                            best_match_name = name
                    
                    label = best_match_name if best_match_name else "Unknown"
                except Exception as e:
                    label = "Error"
                    print(f"Error processing : {e}")

                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        except Exception as e:
            print(f"Error detecting face: {e}")

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_recognition()
