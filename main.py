# face_attendance_full.py - COMPLETE WORKING SYSTEM
import cv2
import numpy as np
import os
import pickle
import time
from datetime import datetime
import csv

class FaceAttendanceSystem:
    def __init__(self):
        print("="*60)
        print("🤖 FACE ATTENDANCE SYSTEM - LIVE CAMERA")
        print("="*60)
        
        # Setup directories
        self.dataset_dir = "face_dataset"
        self.model_dir = "trained_models"
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ ERROR: Cannot open camera!")
            exit()
        
        # Camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Face database
        self.name_to_id = {}
        self.id_to_name = {}
        self.next_id = 0
        
        # Load existing model
        self.load_model()
        
        print(f"✅ System Initialized")
        print(f"📊 Known faces: {len(self.name_to_id)}")
        print(f"📷 Camera ready")
        print("="*60)
    
    def load_model(self):
        """Load trained face model"""
        model_file = os.path.join(self.model_dir, "face_model.yml")
        labels_file = os.path.join(self.model_dir, "labels.pickle")
        
        if os.path.exists(model_file) and os.path.exists(labels_file):
            try:
                self.recognizer.read(model_file)
                with open(labels_file, 'rb') as f:
                    self.name_to_id = pickle.load(f)
                
                self.id_to_name = {v: k for k, v in self.name_to_id.items()}
                if self.name_to_id:
                    self.next_id = max(self.name_to_id.values()) + 1
                
                print(f"📂 Loaded {len(self.name_to_id)} faces: {list(self.name_to_id.keys())}")
                return True
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
        
        print("ℹ️ Starting fresh - no trained faces yet")
        return False
    
    def save_model(self):
        """Save trained model"""
        model_file = os.path.join(self.model_dir, "face_model.yml")
        labels_file = os.path.join(self.model_dir, "labels.pickle")
        
        try:
            self.recognizer.write(model_file)
            with open(labels_file, 'wb') as f:
                pickle.dump(self.name_to_id, f)
            print(f"💾 Model saved with {len(self.name_to_id)} faces")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    # ====================
    # OPTION 1: TRAIN NEW FACE
    # ====================
    def train_new_face(self):
        """Train a new face with live camera"""
        print("\n" + "="*50)
        print("👤 TRAIN NEW FACE")
        print("="*50)
        
        # Get person's name
        name = input("Enter person's name: ").strip()
        if not name:
            print("❌ No name entered")
            return
        
        print(f"\n📸 Training {name}...")
        print("- Look straight at camera")
        print("- Vary expressions slightly")
        print("- Press 'S' to capture samples")
        print("- Press 'Q' when done (need 15+ samples)")
        print("-"*40)
        
        # Create directory for this person
        person_dir = os.path.join(self.dataset_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Collect samples
        samples = []
        sample_count = 0
        max_samples = 20
        
        print("Opening camera for training...")
        
        while sample_count < max_samples:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Camera error!")
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            display = frame.copy()
            
            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display, f"Sample {sample_count+1}/{max_samples}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display info
            cv2.putText(display, f"Training: {name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display, f"Samples: {sample_count}/{max_samples}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display, "Press 'S' to capture | 'Q' to finish", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            
            cv2.imshow("Train New Face", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Extract and preprocess face
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (200, 200))
                    face_equalized = cv2.equalizeHist(face_resized)
                    
                    # Save sample
                    samples.append(face_equalized)
                    
                    # Save image file
                    sample_file = os.path.join(person_dir, f"sample_{sample_count+1}.jpg")
                    cv2.imwrite(sample_file, face_equalized)
                    
                    sample_count += 1
                    
                    # Visual feedback
                    cv2.putText(display, "✅ CAPTURED!", (x, y+h+30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Train New Face", display)
                    cv2.waitKey(200)
                    
                    print(f"  ✓ Captured sample {sample_count}")
            
            elif key == ord('q'):
                break
        
        cv2.destroyWindow("Train New Face")
        
        if len(samples) < 10:
            print(f"❌ Need at least 10 samples, got {len(samples)}")
            return
        
        # Train the model
        print(f"\n🎓 Training model with {len(samples)} samples...")
        success = self.train_with_samples(name, samples)
        
        if success:
            print(f"✅ {name} successfully trained!")
            print(f"   Total faces in system: {len(self.name_to_id)}")
        else:
            print(f"❌ Failed to train {name}")
    
    def train_with_samples(self, name, samples):
        """Train recognizer with collected samples"""
        if len(samples) == 0:
            return False
        
        # Assign ID to new person
        if name not in self.name_to_id:
            self.name_to_id[name] = self.next_id
            self.id_to_name[self.next_id] = name
            self.next_id += 1
        
        label_id = self.name_to_id[name]
        
        # Prepare training data
        labels = [label_id] * len(samples)
        samples_array = np.array(samples, dtype=np.uint8)
        labels_array = np.array(labels, dtype=np.int32)
        
        try:
            # Check if we need to update or train fresh
            model_file = os.path.join(self.model_dir, "face_model.yml")
            
            if os.path.exists(model_file) and len(self.name_to_id) > 1:
                print("  🔄 Updating existing model...")
                self.recognizer.update(samples_array, labels_array)
            else:
                print("  🆕 Training fresh model...")
                self.recognizer.train(samples_array, labels_array)
            
            # Save the model
            self.save_model()
            
            # Test recognition immediately
            self.test_trained_face(name, samples[0])
            
            return True
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def test_trained_face(self, expected_name, test_sample):
        """Test if the trained face can be recognized"""
        try:
            label_id, confidence = self.recognizer.predict(test_sample)
            confidence_percent = max(0, 100 - confidence)
            
            predicted_name = self.id_to_name.get(label_id, "Unknown")
            
            print(f"🧪 Test: Expected {expected_name}, Got {predicted_name} ({confidence_percent:.1f}%)")
            
            if predicted_name == expected_name and confidence_percent > 60:
                print("✅ Recognition test PASSED!")
                return True
            else:
                print("⚠️ Recognition test needs improvement")
                return False
                
        except Exception as e:
            print(f"❌ Test error: {e}")
            return False
    
    # ====================
    # OPTION 2: IDENTIFY FACES (REAL-TIME)
    # ====================
    def identify_faces(self):
        """Real-time face identification"""
        if len(self.name_to_id) == 0:
            print("❌ No faces trained yet! Train some faces first.")
            return
        
        print("\n" + "="*50)
        print("🔍 IDENTIFY FACES - REAL-TIME")
        print("="*50)
        print(f"Looking for {len(self.name_to_id)} known faces")
        print("Press 'Q' to stop identification")
        print("-"*40)
        
        attendance_marked = {}
        
        print("Opening camera for identification...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Camera error!")
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            recognized_names = []
            
            for (x, y, w, h) in faces:
                # Extract and preprocess face
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))
                face_equalized = cv2.equalizeHist(face_resized)
                
                # Predict
                try:
                    label_id, confidence = self.recognizer.predict(face_equalized)
                    confidence_percent = max(0, 100 - confidence)
                    
                    # Get name
                    name = self.id_to_name.get(label_id, "Unknown")
                    
                    # Determine if recognized
                    if confidence_percent > 65 and name != "Unknown":
                        # Known face
                        color = (0, 255, 0)  # Green
                        text = f"{name} ({confidence_percent:.1f}%)"
                        recognized_names.append(name)
                        
                        # Mark attendance (once per session)
                        if name not in attendance_marked:
                            self.mark_attendance(name, confidence_percent)
                            attendance_marked[name] = True
                    else:
                        # Unknown face
                        color = (0, 0, 255)  # Red
                        text = f"Unknown ({confidence_percent:.1f}%)"
                    
                    # Draw on frame
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                except Exception as e:
                    print(f"Recognition error: {e}")
                    continue
            
            # Display statistics
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Recognized: {len(recognized_names)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Database: {len(self.name_to_id)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show recognized names
            if recognized_names:
                names_text = "Identified: " + ", ".join(recognized_names)
                cv2.putText(frame, names_text, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Press 'Q' to quit", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            
            cv2.imshow("Face Identification - Press Q to stop", frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("\n⏹️  Identification stopped")
    
    def mark_attendance(self, name, confidence):
        """Mark attendance in CSV file"""
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")
        time_str = current_time.strftime("%H:%M:%S")
        
        # Create attendance file if it doesn't exist
        attendance_file = "attendance.csv"
        file_exists = os.path.exists(attendance_file)
        
        try:
            with open(attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Date', 'Name', 'Time', 'Confidence'])
                writer.writerow([date_str, name, time_str, f"{confidence:.1f}%"])
            
            print(f"📝 Attendance marked: {name} at {time_str}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving attendance: {e}")
            return False
    
    # ====================
    # OPTION 3: VIEW TRAINED FACES
    # ====================
    def view_trained_faces(self):
        """Show all trained faces"""
        print("\n" + "="*50)
        print("📋 TRAINED FACES DATABASE")
        print("="*50)
        
        if not self.name_to_id:
            print("No faces trained yet")
            return
        
        print(f"Total trained faces: {len(self.name_to_id)}\n")
        
        for i, (name, face_id) in enumerate(self.name_to_id.items(), 1):
            person_dir = os.path.join(self.dataset_dir, name)
            
            if os.path.exists(person_dir):
                samples = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
                sample_count = len(samples)
            else:
                sample_count = 0
            
            print(f"{i:2d}. {name:20} (ID: {face_id:2d}) - {sample_count:2d} samples")
        
        print("\n" + "-"*50)
        print("To retrain all faces, choose option 4")
    
    # ====================
    # OPTION 4: RETRAIN ALL FACES
    # ====================
    def retrain_all_faces(self):
        """Retrain model from all saved images"""
        print("\n" + "="*50)
        print("🔄 RETRAIN ALL FACES")
        print("="*50)
        
        all_samples = []
        all_labels = []
        
        for name in self.name_to_id.keys():
            person_dir = os.path.join(self.dataset_dir, name)
            
            if not os.path.exists(person_dir):
                continue
            
            label_id = self.name_to_id[name]
            
            # Load all images for this person
            image_count = 0
            for img_file in os.listdir(person_dir):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(person_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Resize if needed
                        if img.shape != (200, 200):
                            img = cv2.resize(img, (200, 200))
                        
                        all_samples.append(img)
                        all_labels.append(label_id)
                        image_count += 1
            
            print(f"  Loaded {image_count} samples for {name}")
        
        if len(all_samples) == 0:
            print("❌ No training images found!")
            return
        
        print(f"\n📊 Total: {len(all_samples)} samples from {len(self.name_to_id)} people")
        
        # Train fresh model
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(np.array(all_samples), np.array(all_labels))
            
            # Save model
            self.save_model()
            
            print("✅ Retraining complete!")
            print("🎯 Model is now ready for identification")
            
        except Exception as e:
            print(f"❌ Retraining error: {e}")
    
    # ====================
    # OPTION 5: TEST CAMERA
    # ====================
    def test_camera_only(self):
        """Just test camera without recognition"""
        print("\n" + "="*50)
        print("📷 TEST CAMERA ONLY")
        print("="*50)
        print("Testing camera feed...")
        print("Press 'Q' to exit")
        
        for i in range(100):  # Test for 100 frames
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Camera error!")
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces to test
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'Q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            
            cv2.imshow("Camera Test - Press Q to exit", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("✅ Camera test complete")
    
    # ====================
    # OPTION 6: VIEW ATTENDANCE
    # ====================
    def view_attendance(self):
        """View today's attendance records"""
        print("\n" + "="*50)
        print("📊 ATTENDANCE RECORDS")
        print("="*50)
        
        attendance_file = "attendance.csv"
        
        if not os.path.exists(attendance_file):
            print("No attendance records found")
            return
        
        try:
            with open(attendance_file, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if len(rows) <= 1:
                print("No attendance data")
                return
            
            today = datetime.now().strftime("%Y-%m-%d")
            print(f"Today's date: {today}")
            print("-"*40)
            
            # Show today's attendance
            today_count = 0
            for row in rows[1:]:  # Skip header
                if len(row) >= 3 and row[0] == today:
                    print(f"✅ {row[1]:20} - {row[2]:8}")
                    today_count += 1
            
            if today_count == 0:
                print("No attendance marked today")
            else:
                print(f"\nTotal today: {today_count} people")
            
            # Show overall stats
            total_count = len(rows) - 1
            unique_dates = set(row[0] for row in rows[1:] if len(row) > 0)
            
            print(f"\n📈 Overall Statistics:")
            print(f"   Total records: {total_count}")
            print(f"   Days recorded: {len(unique_dates)}")
            
        except Exception as e:
            print(f"❌ Error reading attendance: {e}")
    
    # ====================
    # MAIN MENU
    # ====================
    def show_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("🎯 FACE ATTENDANCE SYSTEM - MAIN MENU")
        print("="*60)
        print("1. 👤 Train a new face")
        print("2. 🔍 Identify faces (Real-time)")
        print("3. 📋 View trained faces")
        print("4. 🔄 Retrain all faces")
        print("5. 📷 Test camera only")
        print("6. 📊 View attendance records")
        print("7. 🚪 Exit system")
        print("="*60)
    
    def run(self):
        """Main program loop"""
        while True:
            self.show_menu()
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                self.train_new_face()
            elif choice == '2':
                self.identify_faces()
            elif choice == '3':
                self.view_trained_faces()
                input("\nPress Enter to continue...")
            elif choice == '4':
                self.retrain_all_faces()
            elif choice == '5':
                self.test_camera_only()
            elif choice == '6':
                self.view_attendance()
                input("\nPress Enter to continue...")
            elif choice == '7':
                print("\n👋 Thank you for using Face Attendance System!")
                break
            else:
                print("❌ Invalid choice! Please enter 1-7")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

# ====================
# SIMPLE VERSION - For Quick Testing
# ====================

def simple_face_system():
    """Ultra-simple version for testing"""
    print("="*60)
    print("🎯 SIMPLE FACE SYSTEM - CHOOSE OPTION")
    print("="*60)
    print("1. Train new face")
    print("2. Identify faces")
    print("3. Exit")
    print("="*60)
    
    # Create directories
    os.makedirs("faces", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Try to load existing model
    try:
        recognizer.read("models/face_model.yml")
        print("✅ Loaded existing model")
    except:
        print("⚠️  Starting fresh - no model found")
    
    while True:
        choice = input("\nChoose (1=Train, 2=Identify, 3=Exit): ").strip()
        
        if choice == '1':
            # TRAIN
            name = input("Enter your name: ")
            print(f"\nTraining {name}... Look at camera")
            print("Press 'S' to save face, 'Q' to finish")
            
            samples = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.imshow(f"Training {name} - Press S to capture", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s') and len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face = gray[y:y+h, x:x+w]
                        face = cv2.resize(face, (200, 200))
                        samples.append(face)
                        print(f"✓ Captured sample {len(samples)}")
                
                elif key == ord('q'):
                    break
            
            cv2.destroyAllWindows()
            
            if samples:
                # Train
                labels = [0] * len(samples)
                recognizer.train(samples, np.array(labels))
                recognizer.write("models/face_model.yml")
                print(f"✅ Trained {name} with {len(samples)} samples")
        
        elif choice == '2':
            # IDENTIFY
            print("\nIdentifying faces... Press 'Q' to stop")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))
                    
                    label, confidence = recognizer.predict(face)
                    confidence = 100 - confidence
                    
                    if confidence > 60:
                        text = f"Known ({confidence:.0f}%)"
                        color = (0, 255, 0)
                    else:
                        text = f"Unknown ({confidence:.0f}%)"
                        color = (0, 0, 255)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.imshow("Face Identification - Press Q to stop", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cv2.destroyAllWindows()
            print("Identification stopped")
        
        elif choice == '3':
            print("👋 Goodbye!")
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ====================
# QUICK START GUIDE
# ====================

def quick_start():
    """Quick start instructions"""
    print("="*70)
    print("🚀 QUICK START GUIDE")
    print("="*70)
    print("\nSTEP 1: First time setup")
    print("   pip install opencv-python opencv-contrib-python numpy")
    print("\nSTEP 2: Run the system")
    print("   python face_attendance_full.py")
    print("\nSTEP 3: Train your first face")
    print("   Choose option 1 from menu")
    print("   Enter your name")
    print("   Look at camera, press 'S' 15-20 times")
    print("   Press 'Q' when done")
    print("\nSTEP 4: Test identification")
    print("   Choose option 2 from menu")
    print("   Look at camera - should see your name in green box")
    print("   Press 'Q' to stop")
    print("\nSTEP 5: Add more people")
    print("   Repeat Step 3 for each person")
    print("="*70)
    input("\nPress Enter to start the system...")

# ====================
# MAIN EXECUTION
# ====================

if __name__ == "__main__":
    # Show quick start guide
    quick_start()
    
    # Create and run the system
    try:
        system = FaceAttendanceSystem()
        system.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf camera doesn't work, try these fixes:")
        print("1. Make sure camera is connected")
        print("2. Try: cv2.VideoCapture(1) instead of 0")
        print("3. Check permissions")
        input("\nPress Enter to exit...")