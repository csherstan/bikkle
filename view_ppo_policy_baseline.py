import math

import gymnasium as gym
import pygame
import torch
from gymnasium.vector import AutoresetMode

from env import BikkleGymEnvironment
from leanrl_ppo_baseline import Agent, make_env

# Load the saved policy model
model_path = "/home/sherstancraig/work/maincode/data/leanrl_ppo_baseline-FlatBikkleGymEnvironment-v0__leanrl_ppo_baseline__1__True__False__1746878353/actor_model_1331200.pth"  # Replace with your saved model path
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Initialize the environment
env = make_env("FlatBikkleGymEnvironment-v0", 0, capture_video=False, run_name="", gamma=0.99, render_mode="human", continuing=True)()
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

envs = gym.vector.SyncVectorEnv([lambda : env], autoreset_mode=AutoresetMode.SAME_STEP)
obs_space = env.observation_space
action_space = env.action_space
n_act = math.prod(envs.single_action_space.shape)
n_obs = math.prod(envs.single_observation_space.shape)

# Load the policy model
# policy = BikklePolicy(observation_space=obs_space, action_space=action_space).to(device)
agent = Agent(n_obs=n_obs, n_act=n_act, device=device)
agent.load_state_dict(torch.load(model_path, map_location=device))
policy = agent.get_action_and_value

# Run the environment
obs, _ = envs.reset()
running = True

count = 0
while running:
    # Get the action from the policy
    with torch.no_grad():
        action, _, _, _ = policy(torch.as_tensor(obs, dtype=torch.float), greedy=True)
    action = action.cpu().numpy()

    # Step the environment
    obs, reward, terminated, truncated, info = envs.step(action)

    if reward != 0:
        print(f"Reward: {reward}")

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