"""
One-liner setup for Google Colab - run this in a notebook cell.
This installs everything needed for FQL training.
"""

def setup_fql_colab():
    """Complete setup for FQL in Google Colab."""
    import subprocess
    import sys
    import os
    
    print("🚀 Setting up FQL for Google Colab...")
    print("=" * 40)
    
    # Step 1: Set environment variables
    print("🔧 Setting environment variables...")
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    
    # Step 2: Update system packages
    print("📦 Updating system packages...")
    try:
        subprocess.run(["apt-get", "update", "-qq"], check=True, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([
            "apt-get", "install", "-y", "-qq",
            "libgl1-mesa-dev", "libgl1-mesa-glx", "libglew-dev", 
            "libosmesa6-dev", "patchelf", "libglfw3", "libglfw3-dev"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ System packages updated")
    except:
        print("⚠️ System update skipped (may not have permissions)")
    
    # Step 3: Install MuJoCo 2.1.0
    print("📦 Installing MuJoCo 2.1.0...")
    try:
        import urllib.request
        import tarfile
        
        mujoco_dir = "/root/.mujoco"
        os.makedirs(mujoco_dir, exist_ok=True)
        
        if not os.path.exists(f"{mujoco_dir}/mujoco210"):
            print("⬬ Downloading MuJoCo...")
            urllib.request.urlretrieve(
                "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz",
                "/tmp/mujoco210.tar.gz"
            )
            
            print("📂 Extracting...")
            with tarfile.open("/tmp/mujoco210.tar.gz", "r:gz") as tar:
                tar.extractall(mujoco_dir)
        
        # Set library path
        lib_path = f"{mujoco_dir}/mujoco210/bin"
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        # Fix missing glewosmesa library issue
        try:
            import subprocess
            subprocess.run([
                "ln", "-sf", 
                "/usr/lib/x86_64-linux-gnu/libGLEW.so", 
                f"{lib_path}/libglewosmesa.so"
            ], check=True)
            print("🔗 Fixed glewosmesa library linking")
        except:
            print("⚠️ Could not fix glewosmesa linking")
        
        print("✅ MuJoCo installed")
        
    except Exception as e:
        print(f"❌ MuJoCo installation failed: {e}")
        print("🔄 Will use compatibility patches instead")
    
    # Step 4: Install Python packages
    print("📦 Installing Python packages...")
    packages = [
        "mujoco-py<2.2,>=2.0",
        "d4rl>=1.1", 
        "gymnasium>=0.29.0",
        "jax[cuda12]",
        "flax",
        "optax",
        "wandb"
    ]
    
    for package in packages:
        try:
            print(f"  Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ], check=True)
        except:
            print(f"  ⚠️ Failed to install {package}")
    
    print("✅ Package installation complete")
    
    # Step 5: Test imports
    print("🧪 Testing imports...")
    success = True
    
    try:
        import mujoco_py
        print(f"✅ mujoco-py: {mujoco_py.__version__}")
    except:
        print("⚠️ mujoco-py import failed - will use patches")
        success = False
    
    try:
        import d4rl
        print("✅ d4rl imported")
    except Exception as e:
        print(f"⚠️ d4rl import issues: {e}")
    
    try:
        import jax
        print(f"✅ jax: {jax.__version__}")
    except:
        print("❌ jax import failed")
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Setup complete! You can run:")
        print("  !python main.py --env_name=antmaze-medium-play-v0 --agent.kl_coeff=0.3")
    else:
        print("⚠️ Setup partially complete.")
        print("💡 IMPORTANT: Restart Python kernel (Runtime -> Restart runtime) then try:")
        print("  !python run_with_patches.py --env_name=antmaze-medium-play-v0 --agent.kl_coeff=0.3")
        print("  OR")
        print("  !python main.py --env_name=antmaze-medium-play-v0 --agent.kl_coeff=0.3")
    
    return success

# For Jupyter/Colab usage
if __name__ == "__main__":
    setup_fql_colab()
else:
    # Auto-run when imported
    setup_fql_colab() 