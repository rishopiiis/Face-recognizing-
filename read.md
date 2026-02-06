pip install opencv-python face-recognition numpy pandas


# Create dummy test data first
python main.py test

# Then run the system
python main.py




requirement error occurs 
# 1. UNINSTALL problematic packages first
pip uninstall dlib face-recognition cmake -y

# 2. Install Visual C++ Build Tools (QUICKEST WAY)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Or run this in PowerShell as Administrator:
# winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools"

# 3. ALTERNATIVE: Use pre-built wheel (Recommended)
# Download from: https://github.com/zjcmwx/dlib-wheel
# Or use this direct link for Python 3.11:
pip install https://github.com/zjcmwx/dlib-wheel/raw/master/dlib-19.24.2-cp311-cp311-win_amd64.whl

# 4. Then install face-recognition
pip install face-recognition