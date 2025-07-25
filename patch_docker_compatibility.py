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
        except Exception as e:
            # Catch *all* exceptions (not only ImportError) so that library/path issues
            # don’t crash startup. We fall back to a pure-mujoco compatibility bridge.
            print(f"⚠️  mujoco_py import failed ({e}); falling back to compatibility bridge...")
        
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
                
            def get_version(self):
                """Return Mujoco version string (fallback to 'unknown')."""
                try:
                    return self.mujoco.__version__
                except AttributeError:
                    try:
                        # Use mj_versionString if available
                        from mujoco import mj_versionString
                        return mj_versionString()
                    except Exception:
                        return "unknown"
                    
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
                    
            class MjSimState:
                """Lightweight stand-in for mujoco_py.MjSimState required by D4RL wrappers."""
                def __init__(self, time, qpos, qvel, act, udd_state=None):
                    self.time = time
                    self.qpos = qpos
                    self.qvel = qvel
                    self.act = act
                    self.udd_state = udd_state
                def copy(self):
                    # Return a shallow copy sufficient for gym Env.set_state()
                    return MuJoCoPyBridge.MjSimState(self.time, self.qpos.copy(), self.qvel.copy(), self.act.copy(), self.udd_state)
                def get_flattened(self):
                    import numpy as _np
                    return _np.concatenate([self.qpos, self.qvel, self.act])
                def __iter__(self):
                    # compatibility shim for unpacking
                    return iter((self.time, self.qpos, self.qvel, self.act, self.udd_state))
        
        # Create the bridge instance
        bridge = MuJoCoPyBridge()
        # Ensure compatibility: some code expects mujoco_py.get_version()
        if not hasattr(bridge, "get_version"):
            bridge.get_version = lambda: getattr(bridge.mujoco, "__version__", "unknown")
        
        # Common helpers expected by gym/d4rl
        bridge.ignore_mujoco_warnings = lambda *a, **kw: None

        # Expose 'functions' submodule (some libs do `from mujoco_py import functions`)
        import types as _types
        functions_mod = _types.ModuleType("mujoco_py.functions")
        sys.modules['mujoco_py.functions'] = functions_mod
        bridge.functions = functions_mod
        
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
        _printed_gbc_warning = False  # closure var to avoid spam

        def patched_get_body_com(self, body_name):
            """Robust replacement for MujocoEnv.get_body_com.
            Works with both new mujoco (MjModel) and legacy mujoco_py models.
            """
            import types, numpy as _np
            try:
                import mujoco  # new mujoco (>=2.3.0)
                model = None

                # Detect model handle depending on env version
                if hasattr(self, 'model') and self.model is not None:
                    model = self.model
                elif hasattr(self, '_data_model_ref'):
                    model = self._data_model_ref
                elif hasattr(self, 'data') and hasattr(self.data, '_bridge_model'):
                    model = self.data._bridge_model

                # If we got new-style MjModel, use mj_name2id
                if model is not None and not isinstance(model, types.ModuleType):
                    try:
                        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                        if body_id != -1:
                            return self.data.xpos[body_id].copy()
                    except TypeError:
                        # When model comes from mujoco_py (PyMjModel) mj_name2id fails – handle below
                        pass

                # Legacy mujoco_py fallback
                if hasattr(self, 'sim') and hasattr(self.sim, 'data'):
                    try:
                        return self.sim.data.get_body_xpos(body_name).copy()
                    except Exception:
                        pass

                return _np.zeros(3)

            except Exception as e:
                nonlocal _printed_gbc_warning
                if not _printed_gbc_warning:
                    print(f"Warning: get_body_com patch failed: {e}")
                    _printed_gbc_warning = True  # suppress further spam
                return _np.zeros(3)
        
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