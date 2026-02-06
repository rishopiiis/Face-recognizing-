# setup.py - Initial setup script
import os
import sys
import subprocess

def check_and_install():
    """Check and install required packages"""
    required = {
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'openpyxl': 'openpyxl'
    }
    
    print("🔧 Setting up Face Attendance System...")
    print("="*50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher required")
        return False
    
    print("✅ Python version OK")
    
    # Create directories
    directories = [
        "face_dataset",
        "trained_models",
        "reports",
        "face_images"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created: {directory}/")
    
    print("\n📦 Checking packages...")
    
    # Install missing packages
    for package, import_name in required.items():
        try:
            __import__(import_name)
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"⬇️  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\n" + "="*50)
    print("🚀 Setup complete!")
    print("\n📁 Directory structure:")
    print("  face_dataset/    - Face training data")
    print("  trained_models/  - Saved models")
    print("  reports/         - Generated reports")
    print("  face_images/     - Pre-existing face images")
    print("\n👉 Run: python face_attendance_improved.py")
    print("="*50)
    
    return True

if __name__ == "__main__":
    check_and_install()