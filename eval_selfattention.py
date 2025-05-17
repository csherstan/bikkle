from multiprocessing import Pool
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode
from tensordict import TensorDict
from tqdm import tqdm
import pandas as pd

from leanrl_ppo_selfattention import make_env, restore_models
import matplotlib.pyplot as plt

# Load the saved policy model
# model_path = Path(
#     "/home/sherstancraig/work/maincode/data/BikkleFakeEyeTracking-v0/leanrl_ppo_selfattention/1747399171_piacxpqy/checkpoint_1996800.pth")
# model_path = "/home/sherstancraig/work/maincode/data/BikkleFakeEyeTracking-v0/leanrl_ppo_selfattention/1747399171_piacxpqy"
model_path = "/home/sherstancraig/work/maincode/data/BikkleFakeEyeTracking-v0/leanrl_ppo_selfattention/1747370163_80dcr9mx"
# model_path = "/home/sherstancraig/work/maincode/data/BikkleFakeEyeTracking-v0/leanrl_ppo_selfattention/test"
device = "cpu"

def run_one_eval(env, policy):
    envs = gym.vector.SyncVectorEnv([lambda: env], autoreset_mode=AutoresetMode.SAME_STEP)

    # Run through 500 episodes and collect returns
    num_episodes = 20
    returns = []

    for episode in range(num_episodes):
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

def eval_model(policy):
    results = {}
    fake_eye_tracking_env = make_env("BikkleFakeEyeTracking-v0", 0,
                                     capture_video=False,
                                     run_name="",
                                     gamma=0.99,
                                     render_mode=None,
                                     continuing=True,
                                     num_blocks=2,
                                     )()

    results["fake"] = run_one_eval(fake_eye_tracking_env, policy)
    fake_eye_tracking_env.close()

    fake_eye_tracking_env = make_env("BikkleFakeEyeTracking-v0", 0,
                                     capture_video=False,
                                     run_name="",
                                     gamma=0.99,
                                     render_mode=None,
                                     continuing=True,
                                     num_blocks=2,
                                     uniform=True,
                                     )()
    results["fake-uniform"] = run_one_eval(fake_eye_tracking_env, policy)
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
    results["no_eye_tracking"] = run_one_eval(self_attention_env, policy)
    self_attention_env.close()

    return results

def process_file(file_name):
    print(f"Processing: {file_name}")
    model_clock = int(file_name.stem.split("_")[1])  # Extract model_clock as an integer
    # Load the policy model
    agent, *(_) = restore_models(observation_space=obs_space,
                                 action_space=act_space, device="cpu", checkpoint_to_load=file_name)
    agent.eval()
    policy = torch.compile(agent.get_action)
    return model_clock, eval_model(policy)

fake_eye_tracking_env = make_env("BikkleFakeEyeTracking-v0", 0,
                                 capture_video=False,
                                 run_name="",
                                 gamma=0.99,
                                 render_mode=None,
                                 continuing=True,
                                 num_blocks=2,
                                 )()
obs_space = fake_eye_tracking_env.observation_space
act_space = fake_eye_tracking_env.action_space
fake_eye_tracking_env.close()
model_path = Path(model_path)
file_names = []
if model_path.is_file():
    file_names.append(model_path)
else:
    file_names = [file for file in model_path.iterdir() if file.is_file()]

results_list = []
for file_name in tqdm(file_names):
    results_list.append(process_file(file_name))

# Convert the results into a dictionary
results = dict(results_list)


# Convert results dictionary to a DataFrame
results_df = pd.DataFrame.from_dict(results, orient="index").sort_index()

# Plot each column in results_df as a series
results_df.plot(marker='o')

# Set the x-axis label and title
plt.xlabel("Model Clock")
plt.ylabel("Values")
plt.title("Results by Model Clock")

# Show the legend and grid
plt.legend(title="Series")

# Display the plot
plt.show()

# Save the DataFrame to a CSV file
results_df.to_csv("results.csv", index_label="Model Clock")