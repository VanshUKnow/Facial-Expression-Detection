"""
Setup script for facial expression recognition project
"""
import os
import sys
import subprocess
import urllib.request
from src.utils import create_directory_structure, download_haar_cascade

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Requirements installed successfully!")

def setup_project():
    """Setup the complete project structure"""
    print("Setting up Facial Expression Recognition project...")
    
    # Create directory structure
    create_directory_structure()
    
    # Download Haar cascade
    download_haar_cascade()
    
    print("\\nProject setup completed!")
    print("\\nNext steps:")
    print("1. Download the FER2013 dataset from: https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Extract the dataset to the 'data/fer2013/' folder")
    print("3. Run 'python src/train_model.py' to train the model")
    print("4. Run 'python src/real_time_detection.py' for real-time detection")
    
    print("\\nAlternatively, you can use the pre-trained model:")
    print("Run 'python src/real_time_detection.py --mode pretrained'")

def download_sample_model():
    """Download a pre-trained model (placeholder)"""
    print("Note: For a pre-trained model, you can:")
    print("1. Train your own using train_model.py")
    print("2. Use the DeepFace pre-trained model (recommended for quick start)")
    print("3. Download models from the GitHub repositories mentioned in the README")

def check_system_requirements():
    """Check if system has required components"""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Warning: Python 3.7+ is recommended")
    else:
        print(f"✓ Python {sys.version.split()[0]} is supported")
    
    # Check for camera
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera detected")
            cap.release()
        else:
            print("⚠ Camera not detected or not accessible")
    except ImportError:
        print("⚠ OpenCV not installed yet")
    
    print("System check completed!")

def main():
    print("="*60)
    print("FACIAL EXPRESSION RECOGNITION - PROJECT SETUP")
    print("="*60)
    
    try:
        # Check system requirements
        check_system_requirements()
        
        print("\\n" + "="*60)
        
        # Install requirements
        install_requirements()
        
        print("\\n" + "="*60)
        
        # Setup project structure
        setup_project()
        
        print("\\n" + "="*60)
        print("SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\\nError during setup: {str(e)}")
        print("Please check the error message and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())