from collections import deque

import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
import numpy as np
import random
import pygame

from gymnasium.spaces import Sequence, Box, Dict
from gymnasium.wrappers import TimeLimit

import pygame
from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3


class BikkleGymEnvironment(gym.Env):

    render_modes = ["human", "rgb_array"]
    def __init__(self, default_pink_reward: float = 1.0, high_pink_reward: float = 5.0, cyan_penalty: float = -10.0,
                 screen_size: int = 600, num_blocks: int = 4, round_timeout: int = 100, max_action_size: float = 0.02,
                 block_size: float = 0.04, shaping_reward=0.0, continuing=False, render_mode: str = "rgb_array") -> None:
        super().__init__()

        assert block_size < 1.0
        assert max_action_size < 1.0

        self.cyan_blocks = np.empty((num_blocks//2, 2), dtype=np.float32)
        self.pink_blocks = np.empty((num_blocks//2, 2), dtype=np.float32)

        # Environment parameters
        self.screen_size = screen_size
        self.num_blocks = num_blocks
        self.round_timeout = round_timeout
        self.max_action_size = max_action_size
        self.max_action_pixels = max_action_size * screen_size
        self.block_size = block_size
        self.block_size_pixels = block_size * screen_size
        self.shaping_reward = shaping_reward
        self.continuing = continuing

        self.round_steps_count = 0  # number of steps taken in the current round

        self.recent_rewards = deque(maxlen=1000)
        self.recent_action_norms = deque(maxlen=1000)  # Sliding window for recent action norms

        # Define action space: agent can move in x and y directions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Define observation space
        self.observation_space = Dict({
            "agent_position": Box(low=0, high=1.0, shape=(2,), dtype=np.float32),
            "cyan": Sequence(Box(low=-1, high=1.0, shape=(2,), dtype=np.float32), stack=True),  # Variable-length cyan blocks
            "pink": Sequence(Box(low=-1, high=1.0, shape=(2,), dtype=np.float32), stack=True),  # Variable-length pink blocks
            # "screen_image": Box(low=0, high=255, shape=(screen_size, screen_size, 3), dtype=np.uint8),  # RGB image
            "steps": Box(low=0, high=1., shape=(1,), dtype=np.float32)  # Steps taken in the current round
        })

        # Rewards
        self.default_pink_reward = default_pink_reward
        self.high_pink_reward = high_pink_reward
        self.cyan_penalty = cyan_penalty

        self.window = None
        self.clock = None

        self.render_mode = render_mode
        pygame.init()
        self.surface = pygame.Surface((self.screen_size, self.screen_size))

        if self.render_mode == "human":
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Bikkle Gym Environment")
            self.clock = pygame.time.Clock()

        # Initialize environment state
        self.reset()

    def reset(self, seed: int = None, options: dict = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.cyan_blocks, self.pink_blocks = self._generate_blocks()
        while True:
            self.agent_position = np.random.uniform(low=0, high=1, size=2)  # Normalized position
            if not np.any(self._is_touching(self.agent_position, np.vstack((self.cyan_blocks, self.pink_blocks)))):
                break

        self.high_reward_block = random.choice(range(len(self.pink_blocks)))
        self.round_steps_count = 0
        self.total_reward = 0.0
        self.steps_since_reset = 0

        self._update_screen_image()

        # Return initial observation
        return self._get_observation(), {"high_reward_block": self.high_reward_block}

    def _handle_timeout(self) -> None:
        """Handle timeout by changing the high reward block and resetting the timer."""
        self.round_steps_count = 0
        self.high_reward_block = random.choice(range(len(self.pink_blocks)))

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        # Calculate initial distances to pink and cyan blocks
        initial_pink_distances = np.linalg.norm(self.pink_blocks - self.agent_position, axis=1)
        initial_cyan_distances = np.linalg.norm(self.cyan_blocks - self.agent_position, axis=1)

        # Update agent position
        self.agent_position += np.clip(action, -1.0, 1.0) * self.max_action_size
        action_norm = np.linalg.norm(action)  # Calculate the norm of the action
        self.recent_action_norms.append(action_norm)  # Track the action norm
        self.agent_position = np.clip(self.agent_position, 0, 1.0)

        pink_touched = 0
        cyan_touched = 0
        reward = 0.0

        # Calculate new distances to pink and cyan blocks
        new_pink_distances = np.linalg.norm(self.pink_blocks - self.agent_position, axis=1)
        new_cyan_distances = np.linalg.norm(self.cyan_blocks - self.agent_position, axis=1)

        # Calculate the delta in distances (inverted logic)
        delta_pink_distances = new_pink_distances - initial_pink_distances
        delta_cyan_distances = new_cyan_distances - initial_cyan_distances

        # Add shaping rewards/penalties based on the inverted delta
        reward -= self.shaping_reward * np.sum(delta_pink_distances)  # Penalty for reducing distance to pink blocks
        reward += self.shaping_reward * np.sum(delta_cyan_distances)  # Bonus for reducing distance to cyan blocks

        # Check for collisions with cyan blocks
        cyan_hits = self._is_touching(self.agent_position, self.cyan_blocks)
        if np.any(cyan_hits):
            cyan_touched = np.sum(cyan_hits)
            reward += self.cyan_penalty*cyan_touched
            self.cyan_blocks = np.vstack([
                self.cyan_blocks[~cyan_hits],
                np.array([self._generate_new_block(np.vstack((self.cyan_blocks, self.pink_blocks))) for _ in range(cyan_touched)])
            ])

        # Check for collisions with pink blocks
        pink_hits = self._is_touching(self.agent_position, self.pink_blocks)
        if np.any(pink_hits):
            pink_touched = np.sum(pink_hits)
            for i in np.where(pink_hits)[0]:
                self.pink_blocks[i] = self._generate_new_block(np.vstack((self.cyan_blocks, self.pink_blocks)))
                if i == self.high_reward_block:
                    reward += max(self.default_pink_reward, self.high_pink_reward * (self.round_timeout - self.round_steps_count) / self.round_timeout)
                    self.high_reward_block = random.choice(range(len(self.pink_blocks)))
                else:
                    reward += self.default_pink_reward

        # Increment step count and update total reward
        self.round_steps_count += 1
        self.steps_since_reset += 1
        self.recent_rewards.append(reward)

        if self.round_steps_count >= self.round_timeout:
            self._handle_timeout()

        self._update_screen_image()

        # Return step information
        obs = self._get_observation()
        return obs, reward, not self.continuing and (pink_touched > 0 or cyan_touched > 0), False, {
            "high_reward_block": self.high_reward_block,
            "average_reward": sum(self.recent_rewards) / max(len(self.recent_rewards), 1),
            "average_action_norm": sum(self.recent_action_norms) / max(len(self.recent_action_norms), 1),
            "pink_touched": pink_touched,
            "cyan_touched": cyan_touched,
        }

    def _generate_blocks(self) -> tuple[np.ndarray, np.ndarray]:
        all_blocks = []
        while len(all_blocks) < self.num_blocks:
            new_block = self._generate_new_block(np.array(all_blocks, dtype=np.float32))
            all_blocks.append(new_block)
        all_blocks = np.array(all_blocks, dtype=np.float32)
        return all_blocks[:self.num_blocks // 2], all_blocks[self.num_blocks // 2:]

    def _generate_new_block(self, existing: np.ndarray) -> np.ndarray:
        while True:
            new_block = np.random.uniform(0, 1, size=(1, 2)).astype(np.float32)
            if len(existing) == 0 or not np.any(np.linalg.norm(existing - new_block, axis=1) <= self.block_size * 2):
                return new_block[0]

    def _get_observation(self) -> dict:
        agent_position = np.array(self.agent_position, dtype=np.float32)
        return {
            "agent_position": agent_position,
            "cyan": np.array(self.cyan_blocks, dtype=np.float32) - agent_position,
            "pink": np.array(self.pink_blocks, dtype=np.float32) - agent_position,
            # "screen_image": self.render(mode="rgb_array"),
            "steps": np.array([self.round_steps_count/self.round_timeout], dtype=np.float32)
        }

    def _is_touching(self, position: np.ndarray, blocks: np.ndarray) -> np.ndarray:
        return np.linalg.norm(blocks - position, axis=1) <= self.block_size * 2

    def _update_screen_image(self) -> None:
        pygame.init()
        surface = pygame.Surface((self.screen_size, self.screen_size))

        surface.fill((0, 0, 0))  # Clear surface with black

        # Draw cyan blocks
        # Draw cyan blocks
        for block in self.cyan_blocks:
            pygame.draw.circle(surface, (0, 255, 255),
                               (int(block[0] * self.screen_size), int(block[1] * self.screen_size)),
                               int(self.block_size * self.screen_size))

        # Draw pink blocks
        for i, block in enumerate(self.pink_blocks):
            color = (255, 0, 255)  # Hot pink
            pygame.draw.circle(surface, color,
                               (int(block[0] * self.screen_size), int(block[1] * self.screen_size)),
                               int(self.block_size * self.screen_size))

        # Draw agent
        pygame.draw.circle(surface, (255, 128, 0),
                           (int(self.agent_position[0] * self.screen_size),
                            int(self.agent_position[1] * self.screen_size)),
                           int(self.block_size * self.screen_size))

        self._agent_image = pygame.surfarray.array3d(surface)

        high_reward_block = self.pink_blocks[self.high_reward_block]
        pygame.draw.circle(surface, (0, 255, 0),
                           (int(high_reward_block[0] * self.screen_size), int(high_reward_block[1] * self.screen_size)),
                           int(self.block_size * self.screen_size))

        self._human_image = pygame.surfarray.array3d(surface)

        # # Render text for average reward and countdown
        # font = pygame.font.SysFont(None, 24)
        # avg_reward_text = font.render(f"Avg Reward: {sum(self.recent_rewards) / max(len(self.recent_rewards), 1):.2f}", True,
        #                               (255, 255, 255))
        # countdown_text = font.render(f"Countdown: {self.round_timeout - self.round_steps_count}", True, (255, 255, 255))
        #
        # surface.blit(avg_reward_text, (10, 10))  # Display average reward at the top-left corner
        # surface.blit(countdown_text, (10, 40))  # Display countdown below the average reward

    def render(self) -> np.ndarray | None:

        # if mode == "human":
        #     pygame.display.flip()
        #     self.clock.tick(30)  # Limit to 30 FPS
        #     return pygame.surfarray.array3d(pygame.display.get_surface())
        #
        # return self._agent_image

        if self.render_mode == "human":
            pygame.surfarray.blit_array(self.window, self._human_image)
            self.clock.tick(10)
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return self._human_image


# Register the environment
gym.envs.registration.register(
    id="BikkleGymEnvironment-v0",
    entry_point="env:BikkleGymEnvironment",
)

class FlatBikkleGymEnvironmentWrapper(ObservationWrapper):
    def __init__(self, env: BikkleGymEnvironment):
        super().__init__(env)
        obs_space = env.observation_space
        total_dim = (
            obs_space["agent_position"].shape[0]
            + obs_space["cyan"].feature_space.shape[0]*env.num_blocks//2
            + obs_space["pink"].feature_space.shape[0]*env.num_blocks//2
            # + obs_space["steps"].shape[0]
        )
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(total_dim,), dtype=np.float32
        )

    def observation(self, observation):
        agent_position = observation["agent_position"].flatten()
        # these are all relative to the agent position

        cyan = observation["cyan"]
        cyan = cyan[np.argsort(np.linalg.norm(cyan, axis=1))]
        cyan = cyan.flatten()
        pink = observation["pink"]
        pink = pink[np.argsort(np.linalg.norm(pink, axis=1))]
        pink = pink.flatten()
        # steps = observation["steps"].flatten()
        return np.concatenate([agent_position, cyan, pink])


def generate_flat_bikkle_env(*args, **kwargs) ->  gym.Env:
    env = BikkleGymEnvironment(*args, **kwargs)
    return TimeLimit(FlatBikkleGymEnvironmentWrapper(env), max_episode_steps=100)

gym.envs.registration.register(
    id="FlatBikkleGymEnvironment-v0",
    entry_point=generate_flat_bikkle_env,
)

gym.envs.registration.register(
    id="FlatBikkleGymEnvironment-v0-shaping",
    entry_point=lambda *args, **kwargs: generate_flat_bikkle_env(shaping_reward=0.01),
)

# replace lines 244 to 244
class AddNegativeObservationWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        original_obs_space = env.observation_space
        assert isinstance(original_obs_space, spaces.Box)
        self.observation_space = original_obs_space

    def observation(self, observation):
        return np.concatenate([observation, np.full((2,), -2, dtype=np.float32)])


class EyeTrackingObservationWrapper(ObservationWrapper):

    def __init__(self, env: gym.Env, gestures):
        super().__init__(env)
        original_obs_space = env.observation_space
        assert isinstance(original_obs_space, spaces.Box)
        self.observation_space = original_obs_space
        self.gestures = gestures

        screen_info = pygame.display.Info()
        self.screen_width = screen_info.current_w
        self.screen_height = screen_info.current_h

        self.cap = VideoCapture(2)

    def observation(self, observation):
        frame = self.cap.read()
        # already calibrated before running.
        event, calibration = self.gestures.step(frame, False, self.screen_width, self.screen_height, context="my_context")
        # TODO: we need to make this relative to window. I think it might already do this, but we need to double-check.
        return np.concatenate([observation, event])
