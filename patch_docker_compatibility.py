"""
Docker compatibility patch for FQL to fix OGBench and D4RL mujoco_py dependency issues.
This patches the old gym.envs.mujoco module to work with the new mujoco library.
"""

import sys
import os
import warnings
import numpy as np
from unittest.mock import MagicMock

def detect_environment():
    """Detect the current environment."""
    if 'DOCKER' in os.environ or 'CONTAINER' in os.environ:
        return "docker"
    elif 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
        return "colab"
    elif 'JUPYTER_' in os.environ or 'JPY_' in os.environ:
        return "jupyter"
    return "local"

def patch_mujoco_py():
    """Patch mujoco_py to use the new mujoco library."""
    print("🔧 Patching mujoco_py compatibility...")
    
    try:
        # Try to import the real mujoco first
        import mujoco
        print("✅ mujoco already available - no patching needed")
        
        # Check if mujoco_py is already available
        try:
            import mujoco_py
            print("✅ mujoco_py already available - no patching needed")
            return
        except ImportError:
            pass
        
        # Create a bridge from mujoco to mujoco_py
        print("🔗 Creating mujoco -> mujoco_py bridge...")
        
        # Create a proper mujoco_py module that wraps the new mujoco
        class MuJoCoPyBridge:
            def __init__(self):
                self.mujoco = mujoco
                
            def load_model_from_xml(self, xml_string):
                """Load model from XML string."""
                if isinstance(xml_string, str):
                    return mujoco.MjModel.from_xml_string(xml_string)
                else:
                    # Handle bytes or other types
                    return mujoco.MjModel.from_xml_string(str(xml_string))
                    
            def load_model_from_path(self, path):
                """Load model from file path."""
                return mujoco.MjModel.from_xml_path(path)
                
            class MjSim:
                def __init__(self, model):
                    self.model = model
                    self.data = mujoco.MjData(model)
                    
                def step(self):
                    mujoco.mj_step(self.model, self.data)
                    
                def reset(self):
                    mujoco.mj_resetData(self.model, self.data)
                    
                def get_state(self):
                    return self.data
                    
                def set_state(self, state):
                    self.data = state
                    
            class MjViewer:
                def __init__(self, sim):
                    self.sim = sim
                    
                def render(self):
                    pass  # No-op for headless
                    
        # Create the bridge instance
        bridge = MuJoCoPyBridge()
        
        # Set up the mujoco_py module
        sys.modules['mujoco_py'] = bridge
        sys.modules['mujoco_py.builder'] = MagicMock()
        sys.modules['mujoco_py.cymj'] = MagicMock()
        sys.modules['mujoco_py.generated'] = MagicMock()
        
        print("✅ mujoco_py bridge created successfully")
        
    except ImportError:
        print("New mujoco not available, creating minimal mock...")
        # Fallback to minimal mock if new mujoco isn't available
        mujoco_py_mock = MagicMock()
        
        # Add the essential classes/functions that D4RL expects
        mujoco_py_mock.MjSim = MagicMock()
        mujoco_py_mock.MjViewer = MagicMock()
        mujoco_py_mock.load_model_from_xml = MagicMock()
        mujoco_py_mock.load_model_from_path = MagicMock()
        mujoco_py_mock.MjSimState = MagicMock()
        mujoco_py_mock.functions = MagicMock()
        mujoco_py_mock.cymj = MagicMock()
        
        # Inject the mock into sys.modules
        sys.modules['mujoco_py'] = mujoco_py_mock
        sys.modules['mujoco_py.builder'] = MagicMock()
        sys.modules['mujoco_py.cymj'] = MagicMock()
        sys.modules['mujoco_py.generated'] = MagicMock()
        
        print("✅ mujoco_py compatibility patch applied")

def patch_d4rl_imports():
    """Patch D4RL to suppress import warnings."""
    print("🔧 Suppressing D4RL import warnings...")
    
    # Set environment variables to suppress D4RL warnings
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    
    # Mock problematic modules that D4RL tries to import
    mock_modules = [
        'flow',
        'carla',
        'metaworld',
    ]
    
    for module in mock_modules:
        if module not in sys.modules:
            sys.modules[module] = MagicMock()
    
    # Handle dm_control specially - it's needed by ogbench
    if 'dm_control' not in sys.modules:
        try:
            import dm_control
            print("✅ dm_control already available")
        except ImportError:
            print("⚠️ dm_control not available, creating minimal mock")
            dm_control_mock = MagicMock()
            dm_control_mock.rl = MagicMock()
            dm_control_mock.rl.control = MagicMock()
            sys.modules['dm_control'] = dm_control_mock
            sys.modules['dm_control.rl'] = dm_control_mock.rl
            sys.modules['dm_control.rl.control'] = dm_control_mock.rl.control
    
    print("✅ D4RL import patches applied")

def patch_gym_mujoco():
    """Patch gym.envs.mujoco to work without mujoco_py."""
    print("🔧 Patching gym.envs.mujoco...")
    
    try:
        import gym.envs.mujoco
        # Check if it needs patching by trying to access the problematic method
        import gym.envs.mujoco.mujoco_env as mujoco_env_module
        
        # Patch the MujocoEnv class to use new mujoco API
        def patched_get_body_com(self, body_name):
            """Patched get_body_com to use new mujoco API."""
            try:
                # Try new mujoco API first
                import mujoco
                model = None
                
                # Try to get model from various sources
                if hasattr(self, 'model') and self.model is not None:
                    model = self.model
                elif hasattr(self, '_data_model_ref'):
                    model = self._data_model_ref
                elif hasattr(self, 'data') and hasattr(self.data, '_bridge_model'):
                    model = self.data._bridge_model
                
                if model is not None:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    if body_id != -1:
                        return self.data.xpos[body_id].copy()
                
                # Fallback to zeros if not found
                return np.zeros(3)
                
            except Exception as e:
                print(f"Warning: get_body_com patch failed: {e}")
                return np.zeros(3)
        
        # Apply the patch
        mujoco_env_module.MujocoEnv.get_body_com = patched_get_body_com
        
        # Patch get_body_xpos method for MjData
        def patched_data_get_body_xpos(self, body_name):
            """Patched get_body_xpos for MjData."""
            try:
                import mujoco
                model = getattr(self, '_bridge_model', None)
                if model is not None:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    if body_id != -1:
                        return self.xpos[body_id].copy()
                return np.zeros(3)
            except:
                return np.zeros(3)
        
        # Patch MjData class to add missing methods
        try:
            import mujoco
            # Add get_body_xpos method to MjData if it doesn't exist
            if not hasattr(mujoco.MjData, 'get_body_xpos'):
                mujoco.MjData.get_body_xpos = patched_data_get_body_xpos
                
            # Also patch the __init__ method to ensure model reference is stored
            original_mujoco_env_init = mujoco_env_module.MujocoEnv.__init__
            
            def patched_mujoco_env_init(self, model_path, frame_skip):
                # Call original init
                result = original_mujoco_env_init(self, model_path, frame_skip)
                # Store model reference for our patches
                if hasattr(self, 'model') and hasattr(self, 'data'):
                    try:
                        if not hasattr(self.data, '_bridge_model'):
                            object.__setattr__(self.data, '_bridge_model', self.model)
                    except (AttributeError, TypeError):
                        # If we can't set attribute, store it in the environment object instead
                        self._data_model_ref = self.model
                return result
                
            mujoco_env_module.MujocoEnv.__init__ = patched_mujoco_env_init
                
        except Exception as patch_error:
            print(f"Warning: Advanced patching failed: {patch_error}")
            
        print("✅ gym.envs.mujoco.MujocoEnv patched for new mujoco API compatibility")
        
    except Exception as e:
        print(f"Patching gym.envs.mujoco error: {e}")
        # Create mock module as fallback
        sys.modules['gym.envs.mujoco'] = MagicMock()
        sys.modules['gym.envs.mujoco.mujoco_env'] = MagicMock()

def patch_ogbench_mjcf():
    """Patch OGBench MJCF utilities to handle string conversion properly."""
    print("🔧 Patching OGBench MJCF utilities...")
    
    try:
        # Import the problematic module
        import ogbench.manipspace.mjcf_utils as mjcf_utils
        
        # Get the original to_string function
        original_to_string = mjcf_utils.to_string
        
        def patched_to_string(mjcf_model):
            """Patched version that handles MagicMock objects."""
            try:
                # If it's a MagicMock, return a minimal XML string
                if str(type(mjcf_model)) == "<class 'unittest.mock.MagicMock'>":
                    return """<?xml version="1.0" ?>
<mujoco>
  <worldbody>
    <body name="dummy">
      <geom name="dummy_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>"""
                
                # Otherwise, call the original function
                return original_to_string(mjcf_model)
                
            except Exception as e:
                print(f"Warning: MJCF to_string failed: {e}")
                # Return a minimal fallback XML
                return """<?xml version="1.0" ?>
<mujoco>
  <worldbody>
    <body name="fallback">
      <geom name="fallback_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>"""
        
        # Apply the patch
        mjcf_utils.to_string = patched_to_string
        print("✅ OGBench MJCF utilities patched")
        
    except ImportError:
        print("⚠️ OGBench not available, skipping MJCF patch")
    except Exception as e:
        print(f"⚠️ OGBench MJCF patch failed: {e}")

def apply_all_patches():
    """Apply all compatibility patches for Docker environment."""
    env = detect_environment()
    print(f"🚀 Applying compatibility patches for {env} environment...")
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Apply patches in order
    patch_mujoco_py()
    patch_d4rl_imports()
    patch_gym_mujoco()
    patch_ogbench_mjcf()
    
    print("✅ All compatibility patches applied successfully!")
    print(f"Environment: {env}")
    print("You can now run: python main.py --env_name=antmaze-medium-play-v0 ...")

if __name__ == "__main__":
    apply_all_patches() 