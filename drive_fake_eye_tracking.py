import os

import numpy as np
from env import generate_eye_tracking_env, generate_fake_eye_tracking_env, BikkleGymEnvironment
import pygame

# Initialize the environment
# env = generate_eye_tracking_env()
env = generate_fake_eye_tracking_env(uniform=True)
obs, _ = env.reset()

# Define movement mapping for arrow keys
key_to_action = {
    pygame.K_UP: np.array([0, -1], dtype=np.float32),
    pygame.K_DOWN: np.array([0, 1], dtype=np.float32),
    pygame.K_LEFT: np.array([-1, 0], dtype=np.float32),
    pygame.K_RIGHT: np.array([1, 0], dtype=np.float32),
}

base_env = env
while not isinstance(base_env, BikkleGymEnvironment):
    base_env = base_env.env

running = True
clock = pygame.time.Clock()

# grab _human_image from the base environment, render it to the screen
pygame.display.init()
surface = pygame.display.set_mode((600, 600))

def denorm(data):
    return (data/2) + 0.5

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

    pygame.surfarray.blit_array(surface, base_env._human_image)

    # draw in eye_tracking token from obs as a small yellow circle
    eye_tracking = denorm(obs["tokens"]["eye_tracking"])
    eye_x = int(eye_tracking[0] * base_env._human_image.shape[1])
    eye_y = int(eye_tracking[1] * base_env._human_image.shape[0])
    pygame.draw.circle(pygame.display.get_surface(), (255, 255, 0), (eye_x, eye_y), 5)

    # show the image
    pygame.display.flip()


    # Print reward if non-zero
    if reward != 0:
        print(f"Reward: {reward}")

    # Check if the game should terminate
    if terminated or truncated:
        print("Game over!")
        obs, _ = env.reset()
    clock.tick(30)

# Clean up
env.close()