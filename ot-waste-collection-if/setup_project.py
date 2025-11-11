#!/usr/bin/env python3
"""
Setup script for the OT Project: Municipal Waste Collection with ALNS

This script helps set up the project environment and run initial validation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported.")
        print("✅ Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor} is supported.")
    return True


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    requirements = [
        "numpy>=1.24",
        "matplotlib>=3.6",
        "pillow>=9.0"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True


def check_imports():
    """Check if all required modules can be imported"""
    print("\n🔍 Checking imports...")
    
    modules_to_check = [
        "numpy",
        "matplotlib",
        "matplotlib.pyplot"
    ]
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    return True


def setup_environment():
    """Setup environment variables for headless operation if needed"""
    print("\n🌍 Setting up environment...")
    
    # For headless servers
    if platform.system() != "Windows":
        os.environ["MPLBACKEND"] = "Agg"
        print("✅ Set MPLBACKEND=Agg for headless operation")
    
    return True


def copy_fixed_files():
    """Copy fixed files over original versions"""
    print("\n📁 Setting up project files...")
    
    files_to_copy = {
        "problem_fixed.py": "problem.py",
        "utils_fixed.py": "utils.py",
        "test_all_fixed.py": "test_all.py"
    }
    
    for source, target in files_to_copy.items():
        if os.path.exists(source):
            try:
                import shutil
                shutil.copy2(source, target)
                print(f"✅ Copied {source} to {target}")
            except Exception as e:
                print(f"❌ Failed to copy {source}: {e}")
                return False
        else:
            print(f"⚠️  {source} not found, skipping")
    
    return True


def run_basic_tests():
    """Run basic validation tests"""
    print("\n🧪 Running basic validation...")
    
    try:
        # Test basic imports
        from problem import ProblemInstance, Location
        from solution import Solution, Route
        from alns import ALNS
        from data_generator import DataGenerator
        
        # Test problem creation
        problem = DataGenerator.generate_instance("Test", 6, 1, seed=42)
        print("✅ Problem instance creation")
        
        # Test ALNS initialization
        solver = ALNS(problem)
        print("✅ ALNS initialization")
        
        # Test solution generation
        initial_solution = solver._generate_initial_solution()
        print("✅ Initial solution generation")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic validation failed: {e}")
        return False


def create_demo_outputs_directory():
    """Create directory for demo outputs"""
    print("\n📂 Creating output directories...")
    
    directories = ["demo_outputs", "outputs", "tests"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created {directory}/")
        except Exception as e:
            print(f"❌ Failed to create {directory}: {e}")
            return False
    
    return True


def run_demo():
    """Run a quick demo to verify everything works"""
    print("\n🎮 Running quick demo...")
    
    try:
        from demo_complete import main as demo_main
        
        # Run with reduced parameters for quick validation
        print("Running comprehensive demo (this may take a few minutes)...")
        success = demo_main()
        
        if success:
            print("✅ Demo completed successfully!")
            return True
        else:
            print("❌ Demo failed")
            return False
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 OT Project Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check imports
    if not check_imports():
        return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Copy fixed files
    if not copy_fixed_files():
        return False
    
    # Create output directories
    if not create_demo_outputs_directory():
        return False
    
    # Run basic tests
    if not run_basic_tests():
        return False
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("1. Run comprehensive demo: python demo_complete.py")
    print("2. Run test suite: python test_all_fixed.py")
    print("3. Basic usage: python main.py --demo basic --save-results")
    print("4. Check outputs in: demo_outputs/ directory")
    
    # Ask if user wants to run demo
    response = input("\nWould you like to run the comprehensive demo now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        return run_demo()
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Project is ready to use!")
        sys.exit(0)
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)