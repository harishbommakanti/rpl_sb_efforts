# general libraries
import os
from gym.logger import info
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import tensorflow as tf

# robosuite libraries
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite import load_controller_config

# RL framework libraries
from stable_baselines.common import make_vec_env
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.tf_util import TensorboardCallback
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.ppo2 import PPO2

def train(seed):
    """
    The central training function for PPO evaluations
    on the robosuite environments.
    """

    # local variables
    robot_name = "Panda"
    env_name = "Lift"
    log_dir = "log"
    experiment_description = "DVE_1Envs_100Horizon_10MSteps_3Seeds_trimmed_obsspace_1.0rewardscale_1mb_10nopte_0ent_1024st"
    folder_path = "/".join([log_dir, env_name+"-"+robot_name, experiment_description])
    seed_desc = f"Seed{seed}"

    # make a multiprocessed environment to train
    n_envs = 1      # simplify it to 1 env for now, w/ multiprocessing it'd be 4

    vec_env_cls = DummyVecEnv   # avoid SubprocVecEnv to simplify problem
    train_env = vec_env_cls([make_env(env_name, robot_name, i, folder_path, seed) for i in range(n_envs)])
    # train_env = VecNormalize(train_env)

    # print(f"observation space dimensions for training env: {train_env.observation_space.shape}")
    # print(f"action space dimensions for training env: {train_env.action_space.shape}")

    # initialize PPO2 model
    tensorboard_path = folder_path + "/tensorboard/"
    ppo2_model = PPO2(
        policy = MlpPolicy,
        env = train_env,
        verbose = 1,
        tensorboard_log = tensorboard_path,
        max_grad_norm = 0.5,
        learning_rate = 0.0003,
        n_steps = 1024,         # try 2048 later
        nminibatches = 1,       # 16 previously
        noptepochs = 10,        # 6 previously
        vf_coef = 0.5,
        lam = 0.95,
        gamma = 0.99,
        cliprange = 0.2,
        ent_coef = 0.0          # change to 0.01 for second exp
    )

    # print(f"observation space dimensions for model: {ppo2_model.observation_space.shape}")
    # print(f"action space dimensions for model: {ppo2_model.action_space.shape}")

    # log the model files every 100k steps. This is based off of the
    # number of vectorized envs since each of their steps contribute towards 100k
    checkpoint_cb = CheckpointCallback(
        save_freq = 1e6 // n_envs,
        save_path = folder_path,
        name_prefix = seed_desc
    )

    # to show additional task-specific data
    tensorboard_cb = TensorboardCallback()

    # train the model
    total_timesteps = int(1e7)
    ppo2_model.learn(
        total_timesteps = total_timesteps,
        log_interval = 1,
        tb_log_name = seed_desc,
        callback = [tensorboard_cb, checkpoint_cb]
    )

    train_env.close()

    # save model data
    model_filepath = folder_path + f"/{seed_desc}.pkl"
    ppo2_model.save(model_filepath)

    # method calls in case of using VecNormalize, am not using currently

    # env_filepath = folder_path + "/vec_normalize8envs500Horizon.pkl"
    # stats_path = os.path.join(log_dir, env_filepath)
    # train_env.save(stats_path)

    # visualize_data(ppo2_model)

def make_env(env_name, robot_name, rank, folder_path, seed):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # use OSC_POSITION controller instead of OSC_POSE
        control_config = load_controller_config(default_controller="OSC_POSITION")

        #gym.make(env_id)
        suite_env = suite.make(
            env_name = env_name,
            robots = robot_name,
            controller_configs = control_config,
            use_camera_obs = False,
            use_object_obs = True,
            reward_scale = 1.0,                      # make max possible reward == horizon
            reward_shaping = True,                  # if false, only successes give reward
            has_renderer = False,
            has_offscreen_renderer = False,
            render_camera = "frontview",
            control_freq = 20,
            horizon = 100,                          # may change to 500 later to match SAC benchmark
            ignore_done = False,                    # setting this to true makes tensorboard/terminal logging go away
            hard_reset = False,
            camera_names = "agentview",
            camera_heights = 48,
            camera_widths = 48,
            camera_depths = False
        )

        # may use VecNormalize outside GymWrapper but had little success with that so far

        # wrap env in a gym wrapper, then a Monitor wrapper to see logger results (None for no Monitor filepath)
        monitor_path = None #folder_path + f"/Seed{seed}_Env{rank}"
        env = Monitor(
            GymWrapper(
                suite_env,
                keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                      'robot0_gripper_qvel', 'cube_pos', 'cube_quat']
            ),
            monitor_path
        )

        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


# need an if __name__ == '__main__' block for SubprovVecEnv as shown on docs
if __name__ == '__main__':
    # from the robosuite-benchmarking runs
    seeds = np.random.randint(0, 1000, 3)
    for seed in seeds:
        train(seed)

    print("Training completed.")
