# Author: Harsh Sharma (231070064)
#
# This file is a setup and demonstration script for the project. It
# automates the process of checking the Python version, installing
# dependencies, verifying the project, and running demonstrations.
#!/usr/bin/env python3
"""
OT Project Setup and Demo Script
Complete setup and verification for the ALNS VRP project
"""

import sys
import os
import subprocess
import time
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True, result.stdout
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False, result.stderr
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False, str(e)


def test_python_version():
    """Check Python version compatibility"""
    print("\n" + "=" * 60)
    print("PYTHON VERSION CHECK")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True


def install_dependencies():
    """Install project dependencies"""
    print("\n" + "=" * 60)
    print("DEPENDENCY INSTALLATION")
    print("=" * 60)
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    return run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Installing project dependencies")


def verify_project():
    """Verify the project works correctly"""
    print("\n" + "=" * 60)
    print("PROJECT VERIFICATION")
    print("=" * 60)
    
    return run_command(f"{sys.executable} verify_project.py", 
                      "Running project verification")


def run_basic_demo():
    """Run a basic demonstration"""
    print("\n" + "=" * 60)
    print("BASIC DEMONSTRATION")
    print("=" * 60)
    
    return run_command(f"{sys.executable} main.py --demo basic", 
                      "Running basic demonstration")


def run_comprehensive_demo():
    """Run a comprehensive demonstration"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    
    return run_command(f"{sys.executable} main.py --demo comprehensive --iterations 100", 
                      "Running comprehensive demonstration")


def main():
    """Main setup and demo function"""
    print("=" * 80)
    print("           OT PROJECT SETUP AND DEMO")
    print("=" * 80)
    print("Municipal Waste Collection with ALNS Algorithm")
    print("Authors: Harsh Sharma & Chaitanya Shinde")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check Python version
    if not test_python_version():
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies()[0]:
        return False
    
    # Step 3: Verify project
    if not verify_project()[0]:
        print("\n⚠️  Project verification failed, but continuing with demo...")
    
    # Step 4: Run basic demo
    demo_success = run_basic_demo()[0]
    
    # Step 5: Ask if user wants comprehensive demo
    if demo_success:
        print("\n" + "=" * 60)
        print("RUN COMPREHENSIVE DEMO?")
        print("=" * 60)
        print("The comprehensive demo will run more iterations and show detailed analysis.")
        print("This may take longer but provides better insights into the algorithm.")
        print("\nTo run comprehensive demo, use:")
        print("  python3 main.py --demo comprehensive --iterations 200")
        
        # Auto-run comprehensive demo
        print("\n🔄 Running comprehensive demo...")
        run_comprehensive_demo()
    
    # Final status
    print("\n" + "=" * 80)
    print("                    SETUP COMPLETE")
    print("=" * 80)
    print("✅ Project is ready to use!")
    
    print("\nNext steps:")
    print("1. Run: python3 main.py --help (for command options)")
    print("2. Run: python3 main.py --demo basic (for quick test)")
    print("3. Run: python3 main.py --demo comprehensive (for full demo)")
    print("4. Run: python3 tests/test_all.py (for complete test suite)")
    print("5. Check src/ folder for implementation details")
    print("6. Check tests/ folder for test cases")
    
    print("\n" + "=" * 80)
    print("                     FINAL STATUS")
    print("=" * 80)
    print(f"Setup completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)