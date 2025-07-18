# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../source')))
import tripedal_robot_tasks.direct.tripedal_robot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../IsaacLab/scripts/reinforcement_learning/rsl_rl')))


# Try to import from IsaacLab workspace first, then fall back to Isaac Sim
try:
    from isaaclab.app import AppLauncher
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    # Isaac Lab not available, create dummy class for syntax checking
    ISAAC_LAB_AVAILABLE = False
    class AppLauncher:
        def __init__(self, args):
            pass
        @staticmethod
        def add_app_launcher_args(parser):
            pass

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--save_joint_data", action="store_true", default=False, 
                    help="Use JointMonitorRunner to save joint data charts during training.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

# Try to import from IsaacLab workspace first, then fall back to Isaac Sim
try:
    from isaaclab.utils.dict import class_to_dict, print_dict
    from isaaclab.utils.io import dump_pickle, dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    # Isaac Lab not available, create dummy classes for syntax checking
    ISAAC_LAB_AVAILABLE = False
    def class_to_dict(obj):
        return {}
    def print_dict(d, nesting=0):
        pass
    def dump_pickle(data, path):
        pass
    def dump_yaml(data, path):
        pass
    class RslRlOnPolicyRunnerCfg:
        pass
    class RslRlVecEnvWrapper:
        def __init__(self, env):
            self.env = env
    class DirectMARLEnv:
        pass
    class DirectMARLEnvCfg:
        pass
    class DirectRLEnvCfg:
        pass
    class ManagerBasedRLEnvCfg:
        pass
    def multi_agent_to_single_agent(env):
        return env
    def get_checkpoint_path(log_root_path, load_run, load_checkpoint):
        return ""
    def hydra_task_config(task, entry_point):
        def decorator(func):
            return func
        return decorator

from rsl_rl.runners import OnPolicyRunner

# 导入JointMonitorRunner（如果需要记录关节数据）
if args_cli.save_joint_data:
    try:
        from isaaclab_tasks.runners import JointMonitorRunner
    except ImportError:
        JointMonitorRunner = OnPolicyRunner  # Fallback to standard runner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env_kwargs = {
        "cfg": env_cfg,
        "render_mode": "rgb_array" if args_cli.video else None
    }
    
    # 对于TLR6环境，添加实时可视化参数
    if "TLR6" in args_cli.task:
        env_kwargs.update({
            "enable_joint_visualization": True,  # 保留实时可视化
            "max_datapoints": 1000
        })
        print("[INFO] Added joint visualization parameters")
        
    # 创建环境
    env = gym.make(args_cli.task, **env_kwargs)
    print(f"[INFO] Environment created: {type(env)}")

    # Try to safely get environment attributes for diagnostic
    try:
        if hasattr(env, 'unwrapped'):
            print(f"[INFO] Unwrapped env type: {type(env.unwrapped)}")
            
            # 检查是否是我们的TLR6环境
            if "TLR6" in str(type(env.unwrapped)):
                # 直接检查常见的属性
                attrs_to_check = [
                    "_enable_joint_visualization", 
                    "_save_plots", 
                    "_plot_save_dir",
                    "_joint_visualizer",
                    "_joint_names",
                    "_debug_visualization"
                ]
                
                print("[INFO] Checking TLR6 environment attributes:")
                for attr in attrs_to_check:
                    if hasattr(env.unwrapped, attr):
                        value = getattr(env.unwrapped, attr)
                        print(f"  - {attr}: {value}")
                    else:
                        print(f"  - {attr}: Not found")
                
                # 特别检查joint_visualizer是否存在且已初始化
                if hasattr(env.unwrapped, "_joint_visualizer") and env.unwrapped._joint_visualizer is not None:
                    print("[INFO] Joint visualizer is initialized")
                elif hasattr(env.unwrapped, "_joint_visualizer"):
                    print("[WARN] Joint visualizer attribute exists but is None")
                else:
                    print("[WARN] Joint visualizer attribute not found")
                    
                # 检查是否有set_current_iteration方法
                if hasattr(env.unwrapped, "set_current_iteration"):
                    print("[INFO] set_current_iteration method is available")
                else:
                    print("[WARN] set_current_iteration method not found")
    except Exception as e:
        print(f"[WARN] Error inspecting environment: {e}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # 创建runner
    agent_cfg_dict = class_to_dict(agent_cfg)
    
    # 根据命令行参数选择使用的Runner
    if args_cli.save_joint_data:
        print("[INFO] Using JointMonitorRunner to save joint data during training")
        runner = JointMonitorRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    else:
        print("[INFO] Using standard OnPolicyRunner")
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
        
    # write git state to logs (if method exists)
    if hasattr(runner, "add_git_repo_to_log"):
        runner.add_git_repo_to_log(__file__)
        
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    
    # 运行训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
