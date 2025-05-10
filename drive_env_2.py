import pygame
import numpy as np
from env import generate_flat_bikkle_env, BikkleGymEnvironment, FlatBikkleGymEnvironmentWrapper

# Initialize the environment
env = BikkleGymEnvironment()
flat_env = FlatBikkleGymEnvironmentWrapper(env)
flat_env.reset()

# Initialize pygame
pygame.init()
screen_size = 600
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Control Agent with Arrow Keys")
clock = pygame.time.Clock()

# Define movement mapping for arrow keys
key_to_action = {
    pygame.K_UP: np.array([0, -1], dtype=np.float32),
    pygame.K_DOWN: np.array([0, 1], dtype=np.float32),
    pygame.K_LEFT: np.array([-1, 0], dtype=np.float32),
    pygame.K_RIGHT: np.array([1, 0], dtype=np.float32),
}

def render_observation(obs, screen):
    """Render the observation on the pygame screen."""
    screen.fill((0, 0, 0))  # Clear screen with black

    # Extract agent position, cyan blocks, and pink blocks from the observation
    agent_position = obs[:2] * screen_size
    cyan_blocks = (obs[2:2 + env.num_blocks // 2 * 2].reshape(-1, 2) + obs[:2]) * screen_size
    pink_blocks = (obs[2 + env.num_blocks // 2 * 2:].reshape(-1, 2) + obs[:2]) * screen_size

    # Draw cyan blocks
    for block in cyan_blocks:
        pygame.draw.circle(screen, (0, 255, 255), block.astype(int), int(env.block_size * screen_size))

    # Draw pink blocks
    for block in pink_blocks:
        pygame.draw.circle(screen, (255, 0, 255), block.astype(int), int(env.block_size * screen_size))

    # Draw agent
    pygame.draw.circle(screen, (255, 128, 0), agent_position.astype(int), int(env.block_size * screen_size))

    pygame.display.flip()

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
    obs, reward, terminated, truncated, info = flat_env.step(action)

    # Render the observation
    render_observation(obs, screen)

    # Print reward if non-zero
    if reward != 0:
        print(f"Reward: {reward}")

    # Check if the game should terminate
    if terminated or truncated:
        print("Game over!")
        obs, _ = flat_env.reset()

    # Limit the frame rate
    clock.tick(30)

# Clean up
env.close()
pygame.quit()