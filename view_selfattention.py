import math
from pathlib import Path

import gymnasium as gym
import pygame
import torch
from gymnasium.vector import AutoresetMode
from tensordict import TensorDict

from env import BikkleGymEnvironment
from leanrl_ppo_selfattention import make_env, restore_models
from model import BikklePolicy, BikklePolicyParams, BikkleValueFunctionParams

# Load the saved policy model
# model_path = "/home/sherstancraig/work/maincode/data/BikkleFakeEyeTracking-v0/leanrl_ppo_selfattention/1747399171_piacxpqy/checkpoint_614400.pth"
# model_path = "/home/sherstancraig/work/maincode/data/BikkleFakeEyeTracking-v0/leanrl_ppo_selfattention/1747399171_piacxpqy/checkpoint_51200.pth"
# model_path = "/home/sherstancraig/work/maincode/data/BikkleFakeEyeTracking-v0/leanrl_ppo_selfattention/1747370163_80dcr9mx/checkpoint_1792000.pth"
# model_path = "/home/sherstancraig/work/maincode/data/BikkleFakeEyeTracking-v0/leanrl_ppo_selfattention/1747399171_piacxpqy/checkpoint_1996800.pth"
model_path = "/home/sherstancraig/work/maincode/data/BikkleFakeEyeTracking-v0/leanrl_ppo_selfattention/1747370163_80dcr9mx/checkpoint_1382400.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

model_path = Path(model_path)

env = make_env("BikkleEyeTracking-v0", 0,
               capture_video=False,
               run_name="",
               gamma=0.99,
               continuing=True,
               num_blocks=4,
               fps=10,
               )()
# env = make_env("BikkleSelfAttention-v0", 0,
#                capture_video=False,
#                run_name="",
#                gamma=0.99,
#                continuing=True,
#                num_blocks=4,
#                fps=10,
#                render_mode="human",
#                )()

# Initialize the environment
# env = make_env(env_name, 0, capture_video=False, run_name="", gamma=0.99, render_mode="human",
#                continuing=True, num_blocks=4)()

# base_env = BikkleGymEnvironment()
# env = FlatBikkleGymEnvironmentWrapper(base_env)
# env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
# env = gym.wrappers.RecordEpisodeStatistics(env)
# env = gym.wrappers.ClipAction(env)
# env = gym.wrappers.NormalizeObservation(env)
# env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space)
# env = gym.wrappers.NormalizeReward(env, gamma=0.99)
# env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

base_env = env
while True:
    base_env = base_env.env
    if isinstance(base_env, BikkleGymEnvironment):
        break
# base_env.continuing = True

envs = gym.vector.SyncVectorEnv([lambda: env], autoreset_mode=AutoresetMode.SAME_STEP)
obs_space = env.observation_space
action_space = env.action_space

# Load the policy model
agent, *(_) = restore_models(observation_space=obs_space,
                          action_space=action_space, device=device, checkpoint_to_load=model_path)
agent.eval()
policy = agent.get_action

# Run the environment
obs, _ = envs.reset()
running = True

count = 0
while running:
    # Get the action from the policy
    with torch.no_grad():
        action, _, _ = policy(TensorDict(obs, device=device, batch_size=(1,)), greedy=True)
    action = action.cpu().numpy()

    # Step the environment
    obs, reward, terminated, truncated, info = envs.step(action)

    if "reward" in info and info["reward"] != 0:
        print(f"Reward: {info['reward']}")

    # Render the environment
    base_env.render()

    count += 1

    # Check for termination
    if terminated or truncated or count % 200 == 0:
        print("Episode finished!")
        obs, _ = envs.reset()
        count = 0

# Clean up
env.close()
pygame.quit()
