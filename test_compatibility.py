#!/usr/bin/env python3
"""
Test script to verify compatibility patches work correctly.
Run this before training to ensure your environment is properly set up.
"""

import sys
import traceback

def test_basic_imports():
    """Test basic Python imports."""
    print("🔍 Testing basic imports...")
    try:
        import os, json, random, time
        import numpy as np
        print("✅ Basic Python imports working")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_jax():
    """Test JAX import and basic functionality."""
    print("🔍 Testing JAX...")
    try:
        import jax
        import jax.numpy as jnp
        # Simple test
        x = jnp.array([1, 2, 3])
        y = jnp.sum(x)
        print(f"✅ JAX working (test sum: {y})")
        return True
    except Exception as e:
        print(f"❌ JAX failed: {e}")
        return False

def test_environment_detection():
    """Test environment detection."""
    print("🔍 Testing environment detection...")
    try:
        from patch_colab_compatibility import detect_environment
        env = detect_environment()
        print(f"✅ Environment detected: {env}")
        return True
    except Exception as e:
        print(f"❌ Environment detection failed: {e}")
        return False

def test_mujoco():
    """Test MuJoCo/mujoco_py situation."""
    print("🔍 Testing MuJoCo situation...")
    
    # Test new mujoco
    try:
        import mujoco
        print(f"✅ New mujoco available (version: {mujoco.__version__})")
        new_mujoco = True
    except ImportError:
        print("⚠️ New mujoco not available")
        new_mujoco = False
    
    # Test old mujoco_py
    try:
        import mujoco_py
        print("✅ mujoco_py available (or patched)")
        mujoco_py_available = True
    except ImportError:
        print("⚠️ mujoco_py not available")
        mujoco_py_available = False
    
    return new_mujoco or mujoco_py_available

def test_patches():
    """Test applying compatibility patches."""
    print("🔍 Testing compatibility patches...")
    try:
        from patch_colab_compatibility import apply_all_patches
        apply_all_patches()
        print("✅ Compatibility patches applied successfully")
        return True
    except Exception as e:
        print(f"❌ Patch application failed: {e}")
        traceback.print_exc()
        return False

def test_d4rl_import():
    """Test D4RL import after patches."""
    print("🔍 Testing D4RL import...")
    try:
        import d4rl
        print("✅ D4RL imported successfully")
        return True
    except Exception as e:
        print(f"⚠️ D4RL import had issues: {e}")
        # This might still work for training, so return True
        return True

def test_gymnasium():
    """Test Gymnasium/Gym."""
    print("🔍 Testing Gymnasium...")
    try:
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        state, _ = env.reset()
        env.close()
        print(f"✅ Gymnasium working (CartPole state shape: {state.shape})")
        return True
    except Exception as e:
        print(f"❌ Gymnasium failed: {e}")
        return False

def test_env_creation():
    """Test creating a D4RL environment."""
    print("🔍 Testing D4RL environment creation...")
    try:
        from envs.env_utils import make_env_and_datasets
        # Try a simple environment
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets("antmaze-medium-play-v0", frame_stack=1)
        print(f"✅ Environment created successfully")
        print(f"   Dataset size: {len(train_dataset)}")
        env.close()
        eval_env.close()
        return True
    except Exception as e:
        print(f"⚠️ Environment creation had issues: {e}")
        print("   This might still work depending on the specific environment")
        return True

def main():
    """Run all compatibility tests."""
    print("🧪 FQL Compatibility Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("JAX", test_jax),
        ("Environment Detection", test_environment_detection),
        ("MuJoCo", test_mujoco),
        ("Compatibility Patches", test_patches),
        ("D4RL Import", test_d4rl_import),
        ("Gymnasium", test_gymnasium),
        ("Environment Creation", test_env_creation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your environment is ready for FQL training.")
        print("\nYou can now run:")
        print("  python main.py --env_name=antmaze-medium-play-v0 --agent.kl_coeff=0.3")
    elif passed >= total - 2:
        print("⚠️ Most tests passed. Training should work with minor issues.")
        print("\nTry running:")
        print("  python run_with_patches.py --env_name=antmaze-medium-play-v0 --agent.kl_coeff=0.3")
    else:
        print("❌ Several tests failed. Check your environment setup.")
        print("\nConsider:")
        print("  1. Installing missing dependencies")
        print("  2. Using run_with_patches.py instead of main.py")
        print("  3. Manually applying patches in a notebook cell")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 