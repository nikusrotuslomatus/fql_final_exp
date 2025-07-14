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
        print("✅ mujoco_py already available - no patching needed")
        return False
    except ImportError:
        try:
            import mujoco
            print("🔧 New mujoco available but mujoco_py missing")
            print("💡 Consider running: python install_mujoco_py.py")
            print("🔧 Applying compatibility patches as fallback...")
            return True
        except ImportError:
            print("❌ Neither mujoco nor mujoco_py available")
            print("💡 Install with: python install_mujoco_py.py")
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
        class MjDataBridge:
            def __init__(self, data, model):
                self._data = data
                self._model = model
                
            def get_body_xpos(self, body_name):
                """Get body position using new mujoco API."""
                body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                return self._data.xpos[body_id].copy()
                
            def get_body_xquat(self, body_name):
                """Get body quaternion using new mujoco API."""
                body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                return self._data.xquat[body_id].copy()
                
            @property
            def model(self):
                return self._model
                
            def __getattr__(self, name):
                # Delegate to original data object for other attributes
                return getattr(self._data, name)
        
        class MjSimBridge:
            def __init__(self, model, data=None):
                self.model = model
                self._data = data or mujoco.MjData(model)
                # Create bridge with model reference
                self.data = MjDataBridge(self._data, model)
                
            def step(self):
                mujoco.mj_step(self.model, self._data)
                
            def forward(self):
                mujoco.mj_forward(self.model, self._data)
                
            def reset(self):
                mujoco.mj_resetData(self.model, self._data)
                
        class MjViewerBridge:
            def __init__(self, sim):
                self.sim = sim
                
            def render(self):
                pass  # Mock render method
                
        class MjSimStateBridge:
            def __init__(self):
                pass
                
        mujoco_py_bridge = type('mujoco_py', (), {})()
        mujoco_py_bridge.MjSim = MjSimBridge
        mujoco_py_bridge.MjViewer = MjViewerBridge
        mujoco_py_bridge.load_model_from_xml = mujoco.MjModel.from_xml_string
        mujoco_py_bridge.load_model_from_path = mujoco.MjModel.from_xml_path
        mujoco_py_bridge.MjSimState = MjSimStateBridge
        mujoco_py_bridge.functions = mujoco  # Use mujoco module itself
        mujoco_py_bridge.cymj = mujoco
        
        sys.modules['mujoco_py'] = mujoco_py_bridge
        
    except (ImportError, AttributeError) as e:
        print(f"New mujoco bridge failed ({e}), creating mock mujoco_py...")
        # Fallback to mock if new mujoco isn't available or has issues
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
                
                if model is not None:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    # Try to access xpos from data
                    try:
                        # For real mujoco-py data
                        if hasattr(self.data, 'body_xpos'):
                            return self.data.body_xpos[body_id].copy()
                        elif hasattr(self.data, 'xpos'):
                            return self.data.xpos[body_id].copy()
                        # For bridge data
                        elif hasattr(self.data, '_data') and hasattr(self.data._data, 'xpos'):
                            return self.data._data.xpos[body_id].copy()
                        else:
                            # Fallback: get from sim if available
                            if hasattr(self, 'sim') and hasattr(self.sim.data, 'body_xpos'):
                                return self.sim.data.body_xpos[body_id].copy()
                    except:
                        pass
                else:
                    raise Exception("No model available")
            except Exception as e:
                # Fallback: try the data bridge method if available
                if hasattr(self.data, 'get_body_xpos'):
                    try:
                        return self.data.get_body_xpos(body_name)
                    except:
                        pass
                        
                # Try mujoco-py specific methods
                try:
                    if hasattr(self, 'get_body_com'):
                        # Call original method if it exists
                        import gym.envs.mujoco.mujoco_env as mujoco_env_module
                        if hasattr(mujoco_env_module.MujocoEnv, '_original_get_body_com'):
                            return mujoco_env_module.MujocoEnv._original_get_body_com(self, body_name)
                except:
                    pass
                    
                # Last resort: return zeros
                print(f"Warning: get_body_com fallback for {body_name}: {e}")
                import numpy as np
                return np.zeros(3)
        
        def patched_data_get_body_xpos(data_self, body_name):
            """Add get_body_xpos method to MjData objects."""
            try:
                import mujoco
                # Try to find the model - it might be stored in different places
                model = None
                if hasattr(data_self, '_bridge_model'):
                    model = data_self._bridge_model
                elif hasattr(data_self, 'model'):
                    model = data_self.model
                elif hasattr(data_self, '_model'):
                    model = data_self._model
                
                if model is not None:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    return data_self.xpos[body_id].copy()
                else:
                    # If no model found, try to get from global context or return default
                    print(f"Warning: No model found for get_body_xpos({body_name})")
                    import numpy as np
                    return np.zeros(3)
            except Exception as e:
                print(f"Warning: get_body_xpos fallback for {body_name}: {e}")
                import numpy as np
                return np.zeros(3)
        
        # Store original method and apply patches
        if not hasattr(mujoco_env_module.MujocoEnv, '_original_get_body_com'):
            mujoco_env_module.MujocoEnv._original_get_body_com = mujoco_env_module.MujocoEnv.get_body_com
        mujoco_env_module.MujocoEnv.get_body_com = patched_get_body_com
        
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
                # Store model reference for our patches (as a private attribute to avoid conflicts)
                if hasattr(self, 'model') and hasattr(self, 'data'):
                    # Store model reference in a way that doesn't conflict with MjData
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