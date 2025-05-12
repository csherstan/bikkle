import numpy as np
from env import generate_eye_tracking_env
import pygame

# Initialize the environment
env = generate_eye_tracking_env()
obs, _ = env.reset()

# Define movement mapping for arrow keys
key_to_action = {
    pygame.K_UP: np.array([0, -1], dtype=np.float32),
    pygame.K_DOWN: np.array([0, 1], dtype=np.float32),
    pygame.K_LEFT: np.array([-1, 0], dtype=np.float32),
    pygame.K_RIGHT: np.array([1, 0], dtype=np.float32),
}

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get pressed keys
    keys = pygame.key.get_pressed()
    action = np.array([0, 0], dtype=np.float32)

    # Accumulate actions based on pressed keys
    for key, movement in key_to_action.items():
        if keys[key]:
            action += movement

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Print reward if non-zero
    if reward != 0:
        print(f"Reward: {reward}")

    # Check if the game should terminate
    if terminated or truncated:
        print("Game over!")
        obs, _ = env.reset()

# Clean up
env.close()