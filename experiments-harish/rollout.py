# general libraries
import numpy as np
import matplotlib.pyplot as plt

# robosuite libraries
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite import load_controller_config


# RL framework libraries
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
# maybe evaluate_policy does a lot of the code I wrote already?
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common import set_global_seeds

def make_rollout_env(env_name, robot_name, rank, seed):
    """making of a rollout env mimics the make_env() function in train.py"""

    # use OSC_POSITION controller instead of OSC_POSE
    control_config = load_controller_config(default_controller="OSC_POSITION")
    
    suite_env = suite.make(
        controller_configs=control_config,
        env_name=env_name,
        robots=robot_name,
        use_camera_obs=False,
        use_object_obs=True,
        has_renderer=True,              # set to True for video rollouts
        has_offscreen_renderer=False,
        reward_shaping = True,          # if set to True, rewards will be very small
        reward_scale=1.0
    )

    env = GymWrapper(
            suite_env,
            keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
            'robot0_gripper_qvel', 'cube_pos', 'cube_quat']
    )

    return env

def load_model(model_fpath: str):
    """
    returns a model from the saved filepath,
    whether it is a .zip or .pkl file
    """

    return PPO2.load(model_fpath)


def evaluate_rollouts(experiment_desc: str, model_name: str, horizon: int, num_episodes: int):
    """
    Loads a model file and performs/evaluates the video rollouts.

    - experiment_desc: the experiment description, the overarching folder where data is stored
    - horizon: the length of an episode (in timesteps)
    - num_episodes: the number of episodes to include in the rollout
    """

    # create a lift-panda env with a random seed from 0 to 500
    env_name = "Lift"
    robot_name = "Panda"
    env = make_rollout_env(env_name, robot_name, 0, np.random.randint(0, 500))


    # load the model
    log_dir = "log"
    folder_filepath = "/".join([log_dir, env_name +"-"+robot_name, experiment_desc])
    model_filepath = f"{folder_filepath}/{model_name}"
    model = load_model(model_filepath)


    # run the trained model and visualize results
    
    # log all the summed rewards
    rewards = []
    # 1 if an episode was checked as successfuly completed atleast once
    episode_successes = []
    # max cube height over episode
    max_cube_heights = []
    # 1 if the cube was successfuly grasped atleast once
    cube_grasps = []
    # min eef to tgt dist over episode (0 is optimal)
    eef_to_tgt_dists = []

    for episode in range(num_episodes):
        obs = env.reset()

        episode_stats = {
            "reward": 0,
            "max_cube_height": -10,
            "min_eef_to_tgt_dist": 20,
            "cube_grasp": 0,
            "cube_success": 0
        }

        for i in range(horizon):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # update episode stats using `get_env_info()`
            episode_stats["reward"] += reward
            env_info = env.get_env_info()
            
            episode_stats["max_cube_height"] = max(env_info['cube_height'], episode_stats["max_cube_height"])
            
            episode_stats["min_eef_to_tgt_dist"] = min(env_info['cube_to_gripper_dist'], episode_stats["min_eef_to_tgt_dist"])

            # if a cube was grasped or task was completed atleast once in episode,
            # set to true (in case cube is dropped or something like that)
            if env_info['cube_grasped'] == 1:
                episode_stats["cube_grasp"] = 1

            if env_info['cube_success'] == 1:
                episode_stats["cube_success"] = 1

            # show visualizations
            env.render()

        # show logging for each of the env stats
        print("-"*20)
        print(f"Episode {episode}")
        for stat in episode_stats.keys():
            print(f"{stat}: {episode_stats[stat]}")
        print("-"*20)

        # add episode stats to cumulative stats arrays
        rewards.append(episode_stats["reward"])
        episode_successes.append(episode_stats["cube_success"])
        max_cube_heights.append(episode_stats["max_cube_height"])
        eef_to_tgt_dists.append(episode_stats["min_eef_to_tgt_dist"])
        cube_grasps.append(episode_stats["cube_grasp"])


    env.close()



    print("\n\n")
    print("SUMMARY:")
    print(f"Percentage of sucessful episodes: {100*np.mean(episode_successes)}")

    # print out the average for every stat we care about
    print(f"Average reward over rollouts: {np.mean(rewards)}")
    print(f"Average cube success over rollouts: {np.mean(episode_successes)}")
    print(f"Average max cube height over rollouts: {np.mean(max_cube_heights)}")
    print(f"Average min eef to tgt distance over rollouts: {np.mean(eef_to_tgt_dists)}")
    print(f"Average cube grasp over rollouts: {np.mean(cube_grasps)}")

    # plot results for each of the cumulative episode stats
    num_stats = 5
    fig, axs = plt.subplots(num_stats, sharex=True)
    plt.xlabel("episode number")

    x = np.linspace(0, num_episodes, num_episodes)

    # ax[0]: all rewards
    axs[0].plot(x, rewards, 'ro', label='rewards')

    # ax[1]: successes
    axs[1].plot(x, episode_successes, 'bo', label='successes: {0,1}')

    # ax[2]: max cube heights
    axs[2].plot(x, max_cube_heights, 'go', label='max_cube_heights')

    # ax[3]: cube grasp
    axs[3].plot(x, cube_grasps, 'mo', label='cube_grasp: {0,1}')

    # ax[4]: min eef to tgt dist
    axs[4].plot(x, eef_to_tgt_dists, 'co', label='min_eef_tgt_dist')
    

    fig.legend(loc="upper center", ncol = num_stats, fontsize='x-small')
    fig.savefig(f'{experiment_desc}_{model_name}_rollout_evaluation_stats.png')
    # plt.show()


def main():

    # change these to specify what to evaluate. here are test files
    experiment_desc = "DVE_1Envs_500Horizon_10MSteps_3Seeds_trimmed_obsspace_changing_hyperparams_1.0scale"
    seed = 970
    evaluate_checkpts = False

    if evaluate_checkpts:
        # do evaluate_rollout for each checkpoint stored for that seed
        for checkpt in range(int(1e6), int(9e6), int(1e6)):
            model_name = f"Seed{seed}_{checkpt}_steps.zip"
            evaluate_rollouts(
                experiment_desc = experiment_desc,
                model_name = model_name,
                horizon = 100,
                num_episodes = 1
            )
    else:
        # evaluate final model pkl only over 5 episodes
        model_name = f"Seed{seed}_5000000.pkl"
        evaluate_rollouts(
            experiment_desc = experiment_desc,
            model_name = model_name,
            horizon = 500,
            num_episodes = 5
        )
    
    print("Finished evaluating rollouts.")

if __name__ == "__main__":
    main()
