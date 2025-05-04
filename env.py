import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame

from gymnasium.spaces import Sequence, Box, Dict


class BikkleGymEnvironment(gym.Env):
    def __init__(self, default_pink_reward: float = 1.0, high_pink_reward: float = 10.0, cyan_penalty: float = -1.0,
                 screen_size: int = 300, num_blocks: int = 10, round_timeout: int = 100, max_action_size: float = 1.0,
                 block_size: float = 5.0) -> None:
        super().__init__()

        self.cyan_blocks = []
        self.pink_blocks = []

        # Environment parameters
        self.screen_size = screen_size
        self.num_blocks = num_blocks
        self.round_timeout = round_timeout
        self.max_action_size = max_action_size
        self.block_size = block_size

        self.steps = 0  # number of steps taken in the current round

        # Define action space: agent can move in x and y directions
        self.action_space = spaces.Box(low=-self.max_action_size, high=self.max_action_size, shape=(2,), dtype=np.float32)

        # Define observation space
        self.observation_space = Dict({
            "agent_position": Box(low=0, high=screen_size, shape=(2,), dtype=np.float32),
            "cyan": Sequence(Box(low=0, high=screen_size, shape=(2,), dtype=np.float32)),  # Variable-length cyan blocks
            "pink": Sequence(Box(low=0, high=screen_size, shape=(2,), dtype=np.float32)),  # Variable-length pink blocks
            "screen_image": Box(low=0, high=255, shape=(screen_size, screen_size, 3), dtype=np.uint8),  # RGB image
            "steps": Box(low=0, high=round_timeout, shape=(), dtype=np.int32)  # Steps taken in the current round
        })

        # Rewards
        self.default_pink_reward = default_pink_reward
        self.high_pink_reward = high_pink_reward
        self.cyan_penalty = cyan_penalty

        self.window = None
        self.clock = None

        # Initialize environment state
        self.reset()

    def reset(self, seed: int = None, options: dict = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.cyan_blocks, self.pink_blocks = self._generate_blocks()
        while True:
            self.agent_position = np.float32(np.random.uniform(low=0, high=self.screen_size, size=2))
            if all(np.linalg.norm(self.agent_position - np.array(block)) > (self.block_size * 2) for block in
                   (self.cyan_blocks + self.pink_blocks)):
                break
        self.high_reward_block = random.choice(range(len(self.pink_blocks)))
        self.steps = 0

        self._update_screen_image()

        # Return initial observation
        return self._get_observation(), {"high_reward_block": self.high_reward_block}

    def _handle_timeout(self) -> None:
        """Handle timeout by changing the high reward block and resetting the timer."""
        self.steps = 0
        self.high_reward_block = random.choice(range(len(self.pink_blocks)))

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        # Update agent position
        self.agent_position += np.clip(action, -1.0, 1.0)
        self.agent_position = np.clip(self.agent_position, 0, self.screen_size)

        # Check for collisions and calculate reward
        reward = 0.
        for i, block in enumerate(self.cyan_blocks):
            if self._is_touching(self.agent_position, block):
                reward = self.cyan_penalty
                self.cyan_blocks.pop(i)  # Remove the touched block
                self.cyan_blocks.append(self._generate_new_block())  # Add a new block
                break

        for i, block in enumerate(self.pink_blocks):
            if self._is_touching(self.agent_position, block):
                reward = self.high_pink_reward if i == self.high_reward_block else self.default_pink_reward
                self.pink_blocks.pop(i)  # Remove the touched block
                self.pink_blocks.append(self._generate_new_block())  # Add a new block
                if i == self.high_reward_block:
                    self.high_reward_block = random.choice(range(len(self.pink_blocks)))
                break

        # Increment step count
        self.steps += 1
        if self.steps >= self.round_timeout:
            self._handle_timeout()

        self._update_screen_image()

        # Return step information
        return self._get_observation(), reward, False, False, {"high_reward_block": self.high_reward_block}

    def _generate_blocks(self) -> tuple[list[list[float]], list[list[float]]]:
        cyan_blocks = []
        pink_blocks = []
        while len(cyan_blocks) + len(pink_blocks) < self.num_blocks:
            new_block = self._generate_new_block()
            if len(cyan_blocks) < self.num_blocks // 2:
                cyan_blocks.append(new_block)
            else:
                pink_blocks.append(new_block)
        return cyan_blocks, pink_blocks

    def _generate_new_block(self) -> list[float]:
        while True:
            x, y = np.float32(np.random.uniform(0, self.screen_size, size=2))
            new_block = [x, y]
            # Ensure the new block does not overlap with existing blocks
            if all(np.linalg.norm(np.array(new_block) - np.array(block)) > (self.block_size * 2) for block in
                   (self.cyan_blocks + self.pink_blocks)):
                return new_block

    def _get_observation(self) -> dict:
        return {
            "agent_position": self.agent_position,
            "cyan_blocks": self.cyan_blocks,
            "pink_blocks": self.pink_blocks,
            "screen_image": self.render(mode="rgb_array"),
            "steps": self.steps
        }

    def _is_touching(self, position: np.ndarray, block_position: list[float]) -> bool:
        return np.linalg.norm(position - block_position) <= self.block_size  # Collision threshold


    def _update_screen_image(self) -> None:
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Bikkle Gym Environment")
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))  # Clear screen with black

        # Draw cyan blocks
        for block in self.cyan_blocks:
            pygame.draw.circle(self.window, (0, 255, 255), (int(block[0]), int(block[1])), int(self.block_size))

        # Draw pink blocks
        for i, block in enumerate(self.pink_blocks):
            color = (255, 20, 147) if i == self.high_reward_block else (255, 182, 193)
            pygame.draw.circle(self.window, color, (int(block[0]), int(block[1])), int(self.block_size))

        # Draw agent
        pygame.draw.circle(self.window, (255, 255, 255), (int(self.agent_position[0]), int(self.agent_position[1])),
                           int(self.block_size))

        self._agent_image = pygame.surfarray.array3d(pygame.display.get_surface())

        high_reward_block = self.pink_blocks[self.high_reward_block]

        pygame.draw.circle(self.window, (0, 255, 0), (int(high_reward_block[0]), int(high_reward_block[1])), int(self.block_size))


    def render(self, mode: str = "human") -> np.ndarray | None:

        if mode == "human":
            pygame.display.flip()
            self.clock.tick(30)  # Limit to 30 FPS
            return pygame.surfarray.array3d(pygame.display.get_surface())

        return self._agent_image


# Register the environment
gym.envs.registration.register(
    id="BikkleGymEnvironment-v0",
    entry_point="env.BikkleGymEnvironment",
)