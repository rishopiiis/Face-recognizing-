# quick_test.py - Test if everything works
import cv2
print("OpenCV version:", cv2.__version__)

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera works!")
    cap.release()
else:
    print("❌ Camera error!")

try:
    import face_recognition
    print("✅ Face recognition works!")
except:
    print("❌ Face recognition error!")

print("\n✅ All checks passed! Run main.py")