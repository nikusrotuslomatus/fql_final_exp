#!/usr/bin/env python3
"""
Standalone script to apply compatibility patches and run FQL training.
Useful for environments where automatic patching in main.py doesn't work.

Usage:
    python run_with_patches.py --env_name=antmaze-medium-play-v0 --agent.kl_coeff=0.3
"""

import sys
import subprocess

def main():
    print("🚀 Starting FQL with compatibility patches...")
    
    # Apply patches first
    try:
        from patch_colab_compatibility import apply_all_patches
        apply_all_patches()
    except Exception as e:
        print(f"Warning: Could not apply patches: {e}")
    
    # Run main.py with all passed arguments
    cmd = [sys.executable, "main.py"] + sys.argv[1:]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("🛑 Training interrupted by user")
        return 130

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 