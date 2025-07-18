# Copyright (c) 2023-2025, 三足机器人项目开发者.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class TLR6FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 100
    experiment_name = "tlr6_flat_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 64],  # 由于TLR6更简单，使用更小的网络
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TLR6RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000  # 比ANYmal少一些迭代
    save_interval = 100
    experiment_name = "tlr6_rough_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],  # 由于TLR6更简单，使用更小的网络
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TLR6JumpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """跳跃模态的PPO配置"""
    num_steps_per_env = 24
    max_iterations = 6000  # 跳跃任务可能需要更多迭代来学习
    save_interval = 100
    experiment_name = "tlr6_jump_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7,  # 增加初始噪声，提高探索性
        actor_hidden_dims=[256, 128, 64],  # 使用更大的网络
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # 增加熵系数，鼓励更多探索
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,  # 使用较小的学习率
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,  # 允许更大的KL散度，增加探索
        max_grad_norm=1.0,
    ) 