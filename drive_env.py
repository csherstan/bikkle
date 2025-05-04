import pygame
import numpy as np
from env import BikkleGymEnvironment

# Initialize the environment
env = BikkleGymEnvironment()
env.reset()

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((env.screen_size, env.screen_size))
pygame.display.set_caption("Control Agent with Arrow Keys")
clock = pygame.time.Clock()

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
    observation, reward, terminated, truncated, info = env.step(action)

    # Render the environment
    env.render(mode="human")

    # Print reward if non-zero
    if reward != 0:
        print(f"Reward: {reward}")

    # Check if the game should terminate
    if terminated or truncated:
        print("Game over!")
        running = False

    # Limit the frame rate
    clock.tick(30)

# Clean up
env.close()
pygame.quit()