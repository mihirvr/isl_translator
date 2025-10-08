#!/usr/bin/env python3
"""
ISL Translator - Installation Verification Script

This script verifies that all dependencies are properly installed
and the system is ready to run the ISL Translator project.

Run this script after installation to ensure everything works correctly.
"""

import sys
import importlib
import subprocess
import os

def check_python_version():
    """Check if Python version is 3.7 or higher"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not supported. Requires Python 3.7+")
        return False

def check_dependency(package_name, import_name=None):
    """Check if a Python package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        else:
            version = "unknown"
        print(f"✅ {package_name} ({version}) - OK")
        return True
    except ImportError:
        print(f"❌ {package_name} - NOT FOUND")
        return False

def check_camera():
    """Check if camera is accessible"""
    print("\n📹 Checking camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera is accessible and working")
                cap.release()
                return True
            else:
                print("❌ Camera found but cannot read frames")
                cap.release()
                return False
        else:
            print("❌ Cannot open camera (check connections and permissions)")
            return False
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False

def check_audio():
    """Check if text-to-speech is working"""
    print("\n🔊 Checking text-to-speech...")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        # Test without actually speaking (to avoid noise during tests)
        voices = engine.getProperty('voices')
        if voices:
            print(f"✅ Text-to-speech engine initialized with {len(voices)} voices")
            return True
        else:
            print("⚠️  Text-to-speech engine initialized but no voices found")
            return True
    except Exception as e:
        print(f"❌ Text-to-speech test failed: {e}")
        return False

def check_project_files():
    """Check if essential project files exist"""
    print("\n📁 Checking project files...")
    required_files = [
        'collect_data.py',
        'train_model.py', 
        'recognize_sign.py',
        'gui_recognizer.py',
        'model.pkl',
        'gestures.txt',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    dataset_path = 'dataset'
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        gestures = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        print(f"✅ dataset/ directory found with {len(gestures)} gesture categories: {gestures}")
    else:
        print("❌ dataset/ directory not found")
        missing_files.append('dataset/')
    
    return len(missing_files) == 0, missing_files

def main():
    """Main verification function"""
    print("🤟 ISL Translator - Installation Verification")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 7
    
    # Check Python version
    if check_python_version():
        checks_passed += 1
    
    print("\n📦 Checking Python dependencies...")
    
    # Required dependencies
    dependencies = [
        ('opencv-python', 'cv2'),
        ('mediapipe', 'mediapipe'),
        ('scikit-learn', 'sklearn'),
        ('numpy', 'numpy'),
        ('pyttsx3', 'pyttsx3'),
        ('Pillow', 'PIL')
    ]
    
    for package, import_name in dependencies:
        if check_dependency(package, import_name):
            checks_passed += 1
    
    # Check camera
    if check_camera():
        checks_passed += 1
    
    # Check audio
    if check_audio():
        checks_passed += 1
    
    # Check project files
    files_ok, missing_files = check_project_files()
    if files_ok:
        checks_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Verification Summary: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("\n🎉 All checks passed! Your system is ready to run ISL Translator.")
        print("\n🚀 Quick start commands:")
        print("   python gui_recognizer.py      # GUI interface")
        print("   python recognize_sign.py      # Command line interface")
        print("   python collect_data.py        # Collect new training data")
        print("   python train_model.py         # Train new model")
    else:
        print(f"\n⚠️  {total_checks - checks_passed} issues found. Please fix them before running the project.")
        
        if not files_ok:
            print(f"\n📋 Missing files: {', '.join(missing_files)}")
            if 'model.pkl' in missing_files:
                print("   To generate model.pkl: run 'python train_model.py'")
        
        print("\n🔧 Installation help:")
        print("   pip install -r requirements.txt")
        print("   Check README.md for detailed setup instructions")
    
    return checks_passed == total_checks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)