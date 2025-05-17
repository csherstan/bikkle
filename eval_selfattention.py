import numpy as np
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.vector import AutoresetMode
from tensordict import TensorDict
from tqdm import tqdm

from env import BikkleGymEnvironment
from leanrl_ppo_selfattention import make_env, restore_models
from model import BikklePolicy, BikklePolicyParams, BikkleValueFunctionParams

# Load the saved policy model
model_path = Path(
    "/home/sherstancraig/work/maincode/data/BikkleFakeEyeTracking-v0/leanrl_ppo_selfattention/1747399171_piacxpqy/checkpoint_1996800.pth")
device = "cpu"

env_name = "BikkleFakeEyeTracking-v0"


def run_one_eval(env):
    envs = gym.vector.SyncVectorEnv([lambda: env], autoreset_mode=AutoresetMode.SAME_STEP)
    obs_space = env.observation_space
    action_space = env.action_space

    # Load the policy model
    agent, *(_) = restore_models(observation_space=obs_space,
                                 action_space=action_space, device=device, checkpoint_to_load=model_path)
    agent.eval()
    policy = agent.get_action

    # Run through 500 episodes and collect returns
    num_episodes = 20
    returns = []

    for episode in tqdm(range(num_episodes)):
        obs, _ = envs.reset()
        episode_return = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Get the action from the policy
            with torch.no_grad():
                action, _, _ = policy(TensorDict(obs, device=device, batch_size=(1,)), greedy=True)
            action = action.cpu().numpy()

            # Step the environment
            obs, reward, terminated, truncated, info = envs.step(action)
            episode_return += reward

        returns.append(episode_return)

    # Calculate and report the mean return
    mean_return = np.mean(returns)
    # print(f"Mean return over {num_episodes} episodes: {mean_return}")
    return mean_return


fake_eye_tracking_env = make_env(env_name, 0,
                                 capture_video=False,
                                 run_name="",
                                 gamma=0.99,
                                 render_mode=None,
                                 continuing=True,
                                 num_blocks=2,
                                 )()
print("Fake Eye Tracking Environment")
run_one_eval(fake_eye_tracking_env)
fake_eye_tracking_env.close()

fake_eye_tracking_env = make_env(env_name, 0,
                                 capture_video=False,
                                 run_name="",
                                 gamma=0.99,
                                 render_mode=None,
                                 continuing=True,
                                 num_blocks=2,
                                 uniform=True,
                                 )()
print("Fake Eye Tracking Environment-Uniform")
run_one_eval(fake_eye_tracking_env)
fake_eye_tracking_env.close()

self_attention_env = make_env("BikkleSelfAttention-v0", 0,
                              capture_video=False,
                              run_name="",
                              gamma=0.99,
                              continuing=True,
                              num_blocks=2,
                              fps=10,
                              # render_mode="human",
                              )()
print("Self Attention Environment")
run_one_eval(self_attention_env)
self_attention_env.close()
