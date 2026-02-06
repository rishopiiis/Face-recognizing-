# test_camera.py - Test camera and face detection
import cv2

def test_camera():
    """Test if camera and face detection works"""
    print("📷 Testing camera and face detection...")
    
    # Test camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return
    
    print("✅ Camera detected")
    
    # Test face cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    if face_cascade.empty():
        print("❌ Face cascade not loaded")
        cap.release()
        return
    
    print("✅ Face detector loaded")
    
    print("\n👤 Look at the camera. Press 'Q' to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Camera Test - Press Q to exit", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_camera()