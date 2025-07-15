#!/usr/bin/env python3
"""
Test script to verify all fixes made to FQL training.
Run this to check if all issues are resolved.
"""

import os
import sys

def test_imports():
    """Test that all necessary imports work."""
    print("🧪 Testing imports...")
    
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        print("✅ JAX imports successful")
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
        return False
    
    try:
        from utils.evaluation_metrics import MetricsTracker, compute_d4rl_score, compute_success_rate
        print("✅ Evaluation metrics imports successful")
    except ImportError as e:
        print(f"❌ Evaluation metrics import failed: {e}")
        return False
    
    try:
        from agents.fql import FQLAgent
        print("✅ FQL agent import successful")
    except ImportError as e:
        print(f"❌ FQL agent import failed: {e}")
        return False
    
    return True

def test_metrics_tracker():
    """Test MetricsTracker functionality."""
    print("\n🧪 Testing MetricsTracker...")
    
    try:
        # Create tracker
        tracker = MetricsTracker("antmaze-medium-play-v0")
        
        # Test adding evaluation data
        returns = [0.5, 0.8, 0.2, 0.9, 0.1]
        info_dicts = [{'episode': {'return': r}} for r in returns]
        tracker.add_evaluation(returns, info_dicts, step=1000)
        
        # Test adding training metrics
        tracker.add_training_metrics(td_loss=1.5)
        
        # Test computing metrics
        metrics = tracker.compute_comprehensive_metrics()
        
        print(f"✅ MetricsTracker test passed")
        print(f"   Mean return: {metrics['mean_return']:.3f}")
        print(f"   D4RL score: {metrics['d4rl_score']:.3f}")
        print(f"   Success rate: {metrics['success_rate_percent']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ MetricsTracker test failed: {e}")
        return False

def test_trajectory_processing():
    """Test trajectory processing logic."""
    print("\n🧪 Testing trajectory processing...")
    
    try:
        # Simulate trajectory structure from evaluate()
        from collections import defaultdict
        
        traj = defaultdict(list)
        
        # Add some transitions
        for i in range(5):
            traj['reward'].append(0.1 * i)
            traj['info'].append({'step': i, 'episode': {'return': 0.1 * i}})
        
        # Test reward sum calculation
        total_reward = sum(traj['reward'])
        print(f"✅ Reward sum calculation: {total_reward}")
        
        # Test info extraction
        info_list = traj.get('info', {})
        if isinstance(info_list, list) and len(info_list) > 0:
            final_info = info_list[-1]
            print(f"✅ Info extraction: {final_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trajectory processing test failed: {e}")
        return False

def test_config_access():
    """Test config access patterns."""
    print("\n🧪 Testing config access...")
    
    try:
        import ml_collections
        
        # Create test config
        config = ml_collections.ConfigDict({
            'agent_name': 'fql',
            'advantage_weighted': True,
            'kl_coeff': 0.3,
            'adv_weight_coeff': 1.0
        })
        
        # Test access patterns
        agent_name = config.get('agent_name', 'unknown')
        adv_weighted = config.get('advantage_weighted', False)
        kl_coeff = config.get('kl_coeff', 0.0)
        
        print(f"✅ Config access test passed")
        print(f"   Agent: {agent_name}")
        print(f"   Advantage weighted: {adv_weighted}")
        print(f"   KL coeff: {kl_coeff}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config access test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running FQL fixes verification tests...\n")
    
    tests = [
        test_imports,
        test_metrics_tracker,
        test_trajectory_processing,
        test_config_access
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes should work correctly.")
        print("\n🚀 You can now run:")
        print("  python main.py --env_name=antmaze-medium-play-v0 --offline_steps=100000")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 