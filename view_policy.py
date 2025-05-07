import torch
import numpy as np
import pygame
from env import BikkleGymEnvironment
from model import BikklePolicy, preprocess_bikkle_observation_with_mask, SimplePolicy
import gymnasium as gym

from sac import make_env

# Load the saved policy model
model_path = "/home/sherstancraig/work/maincode/data/sac-BikkleGymEnvironment-v0__sac__1__True__False__1746621212/actor_model_19000.pth"  # Replace with your saved model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment
env = BikkleGymEnvironment(num_blocks=2)
envs = gym.vector.SyncVectorEnv([lambda : env])
obs_space = env.observation_space
action_space = env.action_space

# Load the policy model
# policy = BikklePolicy(observation_space=obs_space, action_space=action_space).to(device)
policy = SimplePolicy(obs_space, num_pink=1, num_cyan=1, action_space=action_space).to(device)
policy.load_state_dict(torch.load(model_path, map_location=device))
policy.eval()

# Run the environment
obs, _ = envs.reset()
running = True

count = 0
while running:
    # Preprocess the observation
    processed_obs = preprocess_bikkle_observation_with_mask(obs, observation_space=obs_space, max_blocks=2, device=device)

    # Get the action from the policy
    with torch.no_grad():
        action, _, _ = policy.get_action(**processed_obs, greedy=True)
    action = action.cpu().numpy()

    # Step the environment
    obs, reward, terminated, truncated, info = envs.step(action)

    # Render the environment
    env.render(mode="human")

    count += 1

    # Check for termination
    if terminated or truncated or count % 200 == 0:
        print("Episode finished!")
        obs, _ = envs.reset()
        count = 0


# Clean up
env.close()
pygame.quit()