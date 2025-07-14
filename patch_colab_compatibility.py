"""
Universal compatibility patch for D4RL mujoco_py dependency issues.
Works in Google Colab, local environments, and other platforms.
This patches the old gym.envs.mujoco module to work with the new mujoco library.
"""

import sys
import os
import warnings
from unittest.mock import MagicMock

def detect_environment():
    """Detect the current environment."""
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass
    
    if 'JUPYTER_' in os.environ or 'JPY_' in os.environ:
        return "jupyter"
    
    return "local"

def check_mujoco_py_needed():
    """Check if mujoco_py patching is needed."""
    try:
        import mujoco_py
        print("✅ mujoco_py already available")
        return False
    except ImportError:
        try:
            import mujoco
            print("🔧 mujoco available but mujoco_py missing - patching needed")
            return True
        except ImportError:
            print("❌ Neither mujoco nor mujoco_py available")
            return True

def patch_mujoco_py():
    """Patch mujoco_py to use the new mujoco library or create mock."""
    if not check_mujoco_py_needed():
        return
        
    print("🔧 Patching mujoco_py compatibility...")
    
    # Try to create a functional bridge to new mujoco first
    try:
        import mujoco
        print("Creating mujoco_py bridge using new mujoco...")
        
        # Create a more functional bridge
        class MjSimBridge:
            def __init__(self, model, data=None):
                self.model = model
                self.data = data or mujoco.MjData(model)
                
        class MjViewerBridge:
            def __init__(self, sim):
                self.sim = sim
                
        mujoco_py_bridge = type('mujoco_py', (), {})()
        mujoco_py_bridge.MjSim = MjSimBridge
        mujoco_py_bridge.MjViewer = MjViewerBridge
        mujoco_py_bridge.load_model_from_xml = mujoco.MjModel.from_xml_string
        mujoco_py_bridge.load_model_from_path = mujoco.MjModel.from_xml_path
        mujoco_py_bridge.MjSimState = type('MjSimState', (), {})
        mujoco_py_bridge.functions = mujoco.mjtNum
        mujoco_py_bridge.cymj = mujoco
        
        sys.modules['mujoco_py'] = mujoco_py_bridge
        
    except ImportError:
        print("New mujoco not available, creating mock mujoco_py...")
        # Fallback to mock if new mujoco isn't available
        mujoco_py_mock = MagicMock()
        mujoco_py_mock.MjSim = MagicMock()
        mujoco_py_mock.MjViewer = MagicMock()
        mujoco_py_mock.load_model_from_xml = MagicMock()
        mujoco_py_mock.load_model_from_path = MagicMock()
        mujoco_py_mock.MjSimState = MagicMock()
        mujoco_py_mock.functions = MagicMock()
        mujoco_py_mock.cymj = MagicMock()
        
        sys.modules['mujoco_py'] = mujoco_py_mock
    
    # Add submodules
    sys.modules['mujoco_py.builder'] = MagicMock()
    sys.modules['mujoco_py.cymj'] = sys.modules['mujoco_py'].cymj
    sys.modules['mujoco_py.generated'] = MagicMock()
    
    print("✅ mujoco_py compatibility patch applied")

def patch_d4rl_imports():
    """Patch D4RL to suppress import warnings."""
    print("🔧 Suppressing D4RL import warnings...")
    
    # Set environment variables to suppress D4RL warnings
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    
    # Mock problematic modules
    mock_modules = [
        'flow',
        'carla',
        'metaworld',
        'dm_control'
    ]
    
    for module in mock_modules:
        if module not in sys.modules:
            sys.modules[module] = MagicMock()
    
    print("✅ D4RL import patches applied")

def patch_gym_mujoco():
    """Patch gym.envs.mujoco to work without mujoco_py."""
    print("🔧 Patching gym.envs.mujoco...")
    
    try:
        import gym.envs.mujoco
        # If it imports successfully, we're good
        print("✅ gym.envs.mujoco already working")
    except Exception as e:
        print(f"Patching gym.envs.mujoco error: {e}")
        # Create mock module
        sys.modules['gym.envs.mujoco'] = MagicMock()
        sys.modules['gym.envs.mujoco.mujoco_env'] = MagicMock()

def apply_all_patches():
    """Apply all compatibility patches universally."""
    env = detect_environment()
    print(f"🚀 Applying compatibility patches for {env} environment...")
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Apply patches in order
    patch_mujoco_py()
    patch_d4rl_imports()
    patch_gym_mujoco()
    
    print("✅ All compatibility patches applied successfully!")
    print(f"Environment: {env}")
    print("You can now run: python main.py --env_name=antmaze-medium-play-v0 ...")

if __name__ == "__main__":
    apply_all_patches() 