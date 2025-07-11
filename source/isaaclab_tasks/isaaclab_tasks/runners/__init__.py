from __future__ import annotations

# Note: This import is from Isaac Lab tasks and should be available when Isaac Sim is installed
# # Try to import from IsaacLab workspace first, then fall back to Isaac Sim
try:
    from isaaclab_tasks.runners.joint_monitor_runner import JointMonitorRunner
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    # Isaac Lab not available, create dummy class for syntax checking
    ISAAC_LAB_AVAILABLE = False
    class JointMonitorRunner:
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            pass
 
__all__ = ["JointMonitorRunner"] 