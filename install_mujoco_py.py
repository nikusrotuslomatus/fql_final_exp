#!/usr/bin/env python3
"""
Script to install mujoco-py for Google Colab and other environments.
This is much simpler than complex API bridging patches.
"""

import subprocess
import sys
import os

def install_mujoco_binaries():
    """Install MuJoCo binary files."""
    print("📦 Installing MuJoCo binaries...")
    
    import urllib.request
    import tarfile
    import shutil
    
    try:
        # Create mujoco directory
        mujoco_dir = os.path.expanduser("~/.mujoco")
        os.makedirs(mujoco_dir, exist_ok=True)
        
        # Check if already installed
        mujoco210_path = os.path.join(mujoco_dir, "mujoco210")
        if os.path.exists(mujoco210_path):
            print("✅ MuJoCo 210 already installed")
            return True
        
        # Download MuJoCo 2.1.0
        print("⬬ Downloading MuJoCo 2.1.0...")
        mujoco_url = "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
        tar_path = "/tmp/mujoco210.tar.gz"
        
        urllib.request.urlretrieve(mujoco_url, tar_path)
        
        # Extract
        print("📂 Extracting MuJoCo...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(mujoco_dir)
        
        # Verify installation
        if os.path.exists(mujoco210_path):
            print("✅ MuJoCo binaries installed successfully")
            
            # Set up library path
            lib_path = os.path.join(mujoco210_path, "bin")
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_path not in current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld_path}"
                print(f"📝 Updated LD_LIBRARY_PATH: {lib_path}")
            
            return True
        else:
            print("❌ MuJoCo extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ MuJoCo installation failed: {e}")
        return False

def install_mujoco_py():
    """Install mujoco-py with proper configuration."""
    print("🔧 Installing mujoco-py for D4RL compatibility...")
    
    try:
        # First install MuJoCo binaries
        if not install_mujoco_binaries():
            return False
        
        # Install required system packages for compilation
        print("📦 Installing system dependencies...")
        try:
            subprocess.check_call([
                "apt-get", "update", "-qq"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call([
                "apt-get", "install", "-y", "-qq",
                "libgl1-mesa-dev", "libgl1-mesa-glx", "libglew-dev",
                "libosmesa6-dev", "software-properties-common"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ System dependencies installed")
        except:
            print("⚠️ Could not install system dependencies (may already be installed)")
        
        # Install mujoco-py
        print("📦 Installing mujoco-py...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "mujoco-py<2.2,>=2.0", 
            "--no-cache-dir"
        ])
        
        # Test import
        print("🧪 Testing mujoco-py import...")
        import mujoco_py
        print(f"✅ mujoco-py installed successfully (version: {mujoco_py.__version__})")
        return True
        
    except Exception as e:
        print(f"❌ mujoco-py installation failed: {e}")
        print("🔄 Falling back to compatibility patches...")
        return False

def setup_environment():
    """Set up environment variables for mujoco-py."""
    print("🔧 Setting up environment variables...")
    
    # Suppress D4RL warnings
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    
    # Set GL backend for headless environments
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    
    # Disable CUDA warnings if not needed
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU if available
    
    print("✅ Environment configured")

def main():
    """Main installation process."""
    print("🚀 MuJoCo-py Installation for FQL")
    print("=" * 40)
    
    # Setup environment first
    setup_environment()
    
    # Try to install mujoco-py
    success = install_mujoco_py()
    
    if success:
        print("\n🎉 Installation successful!")
        print("You can now run FQL training directly:")
        print("  python main.py --env_name=antmaze-medium-play-v0 --agent.kl_coeff=0.3")
    else:
        print("\n⚠️ Installation failed - using compatibility patches")
        print("Run with patches:")
        print("  python run_with_patches.py --env_name=antmaze-medium-play-v0 --agent.kl_coeff=0.3")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 