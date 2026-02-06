# ultra_simple.py - Minimum working version
import cv2
import face_recognition
import os
from datetime import datetime

# Create faces folder
os.makedirs("faces", exist_ok=True)

# Load known faces
known_faces = []
known_names = []

for file in os.listdir("faces"):
    if file.endswith(".jpg"):
        name = file.replace(".jpg", "")
        image = face_recognition.load_image_file(f"faces/{file}")
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_faces.append(encodings[0])
            known_names.append(name)

print(f"Loaded {len(known_names)} faces")
print("Press 't' to train, 'q' to quit")

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Find faces
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locs = face_recognition.face_locations(rgb)
    face_encs = face_recognition.face_encodings(rgb, face_locs)
    
    for (top, right, bottom, left), face_enc in zip(face_locs, face_encs):
        # Compare
        matches = face_recognition.compare_faces(known_faces, face_enc)
        name = "Unknown"
        
        if True in matches:
            name = known_names[matches.index(True)]
            print(f"✅ {name} detected")
        
        # Draw box
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Show
    cv2.imshow("Face Recognition", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        name = input("Enter name: ")
        cv2.imwrite(f"faces/{name}.jpg", frame)
        print(f"Saved {name}.jpg")

cap.release()
cv2.destroyAllWindows()