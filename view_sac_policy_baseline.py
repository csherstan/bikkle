import math

import torch
import numpy as np
import pygame
from tensordict.nn import TensorDictModule

from env import BikkleGymEnvironment, generate_flat_bikkle_env, FlatBikkleGymEnvironmentWrapper
from leanrl_sac_baseline_donotedit import Actor
from model import BikklePolicy, preprocess_bikkle_observation_with_mask, SimplePolicy
import gymnasium as gym

from sac import make_env

# Load the saved policy model
model_path = "/home/sherstancraig/work/maincode/data/sac-BikkleGymEnvironment-v0__sac__1__True__False__1746878776/actor_model_190000.pth"  # Replace with your saved model path
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Initialize the environment
base_env = BikkleGymEnvironment(continuing=True, render_mode="human")
env = FlatBikkleGymEnvironmentWrapper(base_env)
envs = gym.vector.SyncVectorEnv([lambda : env])
obs_space = env.observation_space
action_space = env.action_space
n_act = math.prod(envs.single_action_space.shape)
n_obs = math.prod(envs.single_observation_space.shape)

# Load the policy model
# policy = BikklePolicy(observation_space=obs_space, action_space=action_space).to(device)
policy = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)
policy.load_state_dict(torch.load(model_path, map_location=device))
policy = TensorDictModule(policy.get_action, in_keys=["observation"], out_keys=["action"])
policy.eval()

# Run the environment
obs, _ = envs.reset()
running = True

count = 0
while running:
    # Get the action from the policy
    with torch.no_grad():
        action = policy(obs)
    action = action.cpu().numpy()

    # Step the environment
    obs, reward, terminated, truncated, info = envs.step(action)

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