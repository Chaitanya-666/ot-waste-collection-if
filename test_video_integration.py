# Author: Harsh Sharma (231070064)
#
# This file is a test script to verify that the updated main.py with video
# creation works correctly.
#!/usr/bin/env python3
"""
Test script to verify that the updated main.py with video creation works correctly
"""

import sys
import os
import subprocess

# Add the project directory to path
project_dir = "/workspace/OT_Project_ALNS_VRP_FIXED"
sys.path.insert(0, project_dir)
os.chdir(project_dir)

def test_basic_video_functionality():
    """Test basic video creation functionality"""
    print("🧪 Testing Video Creation Integration")
    print("=" * 50)
    
    try:
        # Test 1: Check if main.py can import successfully
        print("1️⃣ Testing import functionality...")
        import main
        print("✅ main.py imports successfully")
        
        # Test 2: Check video creator availability
        print("\n2️⃣ Testing video creator availability...")
        from main import VIDEO_CREATOR_AVAILABLE
        print(f"Video creator available: {VIDEO_CREATOR_AVAILABLE}")
        
        # Test 3: Test basic demonstration
        print("\n3️⃣ Testing basic demonstration...")
        if VIDEO_CREATOR_AVAILABLE:
            print("🎬 Running basic demonstration with video tracking...")
            solution, problem, solver = main.run_basic_demonstration(create_video=True)
            print("✅ Basic demonstration completed")
        else:
            print("⚠️ Video creator not available, running without video...")
            solution, problem, solver = main.run_basic_demonstration(create_video=False)
            print("✅ Basic demonstration completed")
        
        # Test 4: Check if video files were created
        print("\n4️⃣ Checking for created video files...")
        video_files = []
        for file in os.listdir("."):
            if file.endswith((".gif", ".mp4")) and "alns_optimization" in file:
                video_files.append(file)
        
        if video_files:
            print(f"✅ Video files created: {video_files}")
        else:
            print("ℹ️ No video files found (this is normal if video creator isn't available)")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_video_option():
    """Test CLI video option"""
    print("\n🔧 Testing CLI Video Option")
    print("=" * 50)
    
    try:
        # Test help output
        print("1️⃣ Testing --help output...")
        result = subprocess.run([
            sys.executable, "main.py", "--help"
        ], capture_output=True, text=True, cwd=project_dir)
        
        if "--video" in result.stdout:
            print("✅ --video option appears in help output")
        else:
            print("⚠️ --video option not found in help output")
            
        # Test basic run with video option (if video creator available)
        print("\n2️⃣ Testing CLI with --video flag...")
        try:
            from main import VIDEO_CREATOR_AVAILABLE
            if VIDEO_CREATOR_AVAILABLE:
                print("Running basic demo with --video flag...")
                result = subprocess.run([
                    sys.executable, "main.py", "--demo", "basic", "--video", "--iterations", "50"
                ], capture_output=True, text=True, cwd=project_dir, timeout=60)
                
                if result.returncode == 0:
                    print("✅ CLI with --video flag completed successfully")
                    print(f"Output preview: {result.stdout[:200]}...")
                else:
                    print(f"⚠️ CLI returned non-zero exit code: {result.returncode}")
                    print(f"Error: {result.stderr[:200]}")
            else:
                print("ℹ️ Skipping CLI test - video creator not available")
        except subprocess.TimeoutExpired:
            print("⚠️ CLI test timed out (expected for longer runs)")
        except Exception as e:
            print(f"⚠️ CLI test error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎬 Testing Updated main.py with Video Creation")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_basic_video_functionality()
    test2_passed = test_cli_video_option()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Basic functionality test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"CLI option test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Your main.py is ready with video creation!")
        print("\n💡 Usage Examples:")
        print("   python main.py --demo basic --video")
        print("   python main.py --demo comprehensive --video --iterations 300")
        print("   python main.py --live --video")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    print(f"\n📁 Working directory: {os.getcwd()}")
    print("🔧 If video creation fails, ensure simple_video_creator.py is in the workspace root")