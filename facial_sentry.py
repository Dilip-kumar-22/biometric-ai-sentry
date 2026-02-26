import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

# --- CONFIGURATION ---
# Create a folder named 'authorized' and put photos of yourself/team there
AUTHORIZED_DIR = "authorized"
tolerance = 0.6 # Lower is stricter, higher is more loose

class FacialSentry:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_authorized_personnel()

    def load_authorized_personnel(self):
        """Loads and encodes all images in the authorized directory."""
        if not os.path.exists(AUTHORIZED_DIR):
            os.makedirs(AUTHORIZED_DIR)
            print(f"[!] Please add authorized photos to the '{AUTHORIZED_DIR}' folder.")
            return

        print("[*] Initializing Neural Encodings...")
        for filename in os.listdir(AUTHORIZED_DIR):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(AUTHORIZED_DIR, filename)
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)[0]
                
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(os.path.splitext(filename)[0])
        print(f"[+] {len(self.known_face_names)} Profiles Loaded.")

    def run_sentry(self):
        video_capture = cv2.VideoCapture(0)

        print("[*] SENTRY ACTIVE. Monitoring Entrance...")
        while True:
            ret, frame = video_capture.read()
            if not ret: break

            # Shrink frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find faces in current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance)
                name = "UNKNOWN INTRUDER"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    color = (0, 255, 0) # Green for Authorized
                else:
                    color = (0, 0, 255) # Red for Intruder

                # Draw identity box
                top, right, bottom, left = face_locations[0]
                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                if name != "UNKNOWN INTRUDER":
                    print(f"[ACCESS GRANTED] {name} identified at {datetime.now().strftime('%H:%M:%S')}")

            cv2.imshow('NEURAL FORTRESS - Facial Recognition Sentry', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sentry = FacialSentry()
    sentry.run_sentry()
