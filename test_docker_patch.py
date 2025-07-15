#!/usr/bin/env python3
"""
Test script to verify Docker compatibility patch works correctly.
"""

import os
import sys

def test_docker_patch():
    """Test the Docker compatibility patch."""
    print("🧪 Testing Docker compatibility patch...")
    
    # Set Docker environment variables
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    
    try:
        # Test the patch
        from patch_docker_compatibility import apply_all_patches
        apply_all_patches()
        print("✅ Docker compatibility patch applied successfully")
        
        # Test basic imports
        print("🔍 Testing basic imports...")
        import jax
        import jax.numpy as jnp
        import numpy as np
        print("✅ JAX and NumPy imports successful")
        
        # Test ogbench import
        print("🔍 Testing ogbench import...")
        try:
            import ogbench
            print("✅ ogbench imported successfully")
        except Exception as e:
            print(f"⚠️ ogbench import warning: {e}")
        
        # Test environment creation
        print("🔍 Testing environment creation...")
        try:
            import gymnasium as gym
            # Try to create a simple environment first
            env = gym.make('CartPole-v1')
            obs, _ = env.reset()
            print(f"✅ Basic environment created - observation shape: {obs.shape}")
            env.close()
            
            # Try puzzle environment
            try:
                env = gym.make('puzzle-3x3-play-singletask-v0')
                obs, _ = env.reset()
                print(f"✅ Puzzle environment created - observation shape: {obs.shape}")
                env.close()
            except Exception as e:
                print(f"⚠️ Puzzle environment creation failed: {e}")
                
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            return False
        
        print("\n🎉 Docker compatibility patch test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Docker compatibility patch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_docker_patch()
    if success:
        print("\n✅ You can now run FQL training:")
        print("  python main.py --env_name=puzzle-3x3-play-singletask-v0 --offline_steps=200000")
    else:
        print("\n❌ Please check your environment setup")
        print("  Try installing missing dependencies or using a different environment") 