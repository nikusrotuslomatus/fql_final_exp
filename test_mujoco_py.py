#!/usr/bin/env python3
"""
Quick test script to verify mujoco-py installation after kernel restart.
Run this in Google Colab after restarting the runtime.
"""

import os

def test_mujoco_py():
    """Test if mujoco-py works correctly."""
    print("🧪 Testing mujoco-py installation...")
    
    # Set environment variables
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    
    # Set library path
    mujoco_dir = "/root/.mujoco/mujoco210"
    if os.path.exists(mujoco_dir):
        lib_path = f"{mujoco_dir}/bin"
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        print(f"✅ MuJoCo directory found: {mujoco_dir}")
    else:
        print("❌ MuJoCo directory not found - run colab_setup.py first")
        return False
    
    try:
        print("🔍 Testing mujoco-py import...")
        import mujoco_py
        print(f"✅ mujoco-py version: {mujoco_py.__version__}")
        
        print("🔍 Testing D4RL import...")
        import d4rl
        print("✅ D4RL imported successfully")
        
        print("🔍 Testing environment creation...")
        import gymnasium as gym
        env = gym.make('GymV21Environment-v0', env_id='antmaze-medium-play-v0')
        obs, _ = env.reset()
        print(f"✅ Environment created - observation shape: {obs.shape}")
        env.close()
        
        print("\n🎉 All tests passed! You can run FQL training:")
        print("  !python main.py --env_name=antmaze-medium-play-v0 --agent.kl_coeff=0.3")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n🔧 Fallback options:")
        print("  !python run_with_patches.py --env_name=antmaze-medium-play-v0 --agent.kl_coeff=0.3")
        return False

if __name__ == "__main__":
    success = test_mujoco_py()
    exit(0 if success else 1) 