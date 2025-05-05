import torch
import time
from env import BikkleGymEnvironment
from model import BikklePolicy

# Initialize the environment
env = BikkleGymEnvironment()
observation_space = env.observation_space
action_space = env.action_space

device= "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the Policy model
policy = BikklePolicy(observation_space, action_space).to("cuda" if torch.cuda.is_available() else "cpu")

# Generate a batch of observations
batch_size = 1
token_count = 1000
observations = []
for _ in range(batch_size):
    obs, _ = env.reset()
    observations.append(obs)

# Convert observations to tensors
# def preprocess_observations(observations, device):
#     processed = {}
#     for key in observations[0].keys():
#         if key == "screen_image":  # Skip screen images for this test
#             continue
#         values = [torch.tensor(obs[key], dtype=torch.float32) for obs in observations]
#         processed[key] = torch.stack(values).to(device)
#     return processed
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# batched_observations = preprocess_observations(observations, device)

batched_observations = torch.zeros((batch_size, token_count, 64), dtype=torch.float32, device=device)
indices = torch.zeros((batch_size, token_count), dtype=torch.int32, device=device)

# Test throughput
policy.eval()
with torch.no_grad():
    start_time = time.time()
    for _ in range(100):  # Run multiple iterations for better measurement
        mean, log_std = policy(batched_observations, indices)
    end_time = time.time()

# Calculate throughput
elapsed_time = end_time - start_time
throughput = (batch_size * 100) / elapsed_time  # Observations per second

print(f"Throughput: {throughput:.2f} passes per second")