from collections import deque
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
import numpy as np
import random
import pygame

from gymnasium.spaces import Sequence, Box, Dict
from gymnasium.wrappers import TimeLimit

from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3

from model import data_type_idx


class BikkleGymEnvironment(gym.Env):
    render_modes = ["human", "rgb_array"]

    def __init__(self, default_pink_reward: float = 1.0, high_pink_reward: float = 5.0, cyan_penalty: float = -10.0,
                 screen_size: int = 600, num_blocks: int = 4, round_timeout: int = 100, max_action_size: float = 0.02,
                 block_size: float = 0.04, shaping_reward=0.0, continuing=False, render_mode: str = "rgb_array",
                 *args, **kwargs) -> None:
        """
        This is 2D arena where the agent where the agent is rewarded for contacting pink blocks and penalized for
        touching cyan blocks. One of the pink blocks is denoted as the high-reward block, which gives a higher reward.
        This reward drops off linearly over time, based on round_timeout, until hitting a lower threshold of
        default_pink_reward. The agent cannot see which block is the high-reward block.

        At the end of round, the high-reward block is changed to a random pink block, and the round timer is reset.

        If not operating in continuing mode, the environment will terminate when the agent touches a block.

        Observation space:
            agent_position - position of the agent in the arena, normalized to [0, 1]
            cyan - an array of cyan block positions that are relative to the agent-position. normalized to [-1, 1]
            pink - an array of pink block positions that are relative to the agent-position. normalized to [-1, 1]
            steps - the round counter, starting from 0 to 1

        Action space:
            2d continuous valued change in position. Limited by `max_action_size`

        Reward:
            - Touching a pink block gives a reward of `default_pink_reward`
            - Touching the high-reward pink block gives a reward of
            max(default_pink_reward, high_pink_reward * (round_timeout - round_steps_count) / round_timeout)
            - Touching a cyan block gives a penalty of `cyan_penalty`
            - (optional, disabled by default) The agent is penalized for moving closer to cyan blocks and rewarded for
            moving closer to pink blocks

        Args:
            default_pink_reward (float): Reward for touching a pink block (default: 1.0).
            high_pink_reward (float): Higher reward for touching the high-reward pink block (default: 5.0).
            cyan_penalty (float): Penalty for touching a cyan block (default: -10.0).
            screen_size (int): Size of the environment screen in pixels (default: 600).
            num_blocks (int): Total number of blocks (cyan + pink) in the environment (default: 4).
            round_timeout (int): Maximum steps allowed per round (default: 100).
            max_action_size (float): Maximum movement size per action, normalized to [0, 1] (default: 0.02).
            block_size (float): Size of each block, normalized to [0, 1] (default: 0.04).
            shaping_reward (float): Reward shaping factor for distance-based rewards/penalties (default: 0.0).
            continuing (bool): Whether the environment continues after termination conditions (default: False).
            render_mode (str): Mode for rendering the environment, either "human" or "rgb_array" (default: "rgb_array").
        """
        super().__init__()

        assert block_size < 1.0
        assert max_action_size < 1.0

        self.cyan_blocks = np.empty((num_blocks // 2, 2), dtype=np.float32)
        self.pink_blocks = np.empty((num_blocks // 2, 2), dtype=np.float32)

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
            "cyan": Sequence(Box(low=-1, high=1.0, shape=(2,), dtype=np.float32), stack=True),
            # Variable-length cyan blocks
            "pink": Sequence(Box(low=-1, high=1.0, shape=(2,), dtype=np.float32), stack=True),
            # Variable-length pink blocks
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
        return self._get_observation(), {"high_reward_block": self.high_reward_block, "reward": 0.0}

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
            reward += self.cyan_penalty * cyan_touched
            self.cyan_blocks = np.vstack([
                self.cyan_blocks[~cyan_hits],
                np.array([self._generate_new_block(np.vstack((self.cyan_blocks, self.pink_blocks))) for _ in
                          range(cyan_touched)])
            ])

        # Check for collisions with pink blocks
        pink_hits = self._is_touching(self.agent_position, self.pink_blocks)
        if np.any(pink_hits):
            pink_touched = np.sum(pink_hits)
            for i in np.where(pink_hits)[0]:
                self.pink_blocks[i] = self._generate_new_block(np.vstack((self.cyan_blocks, self.pink_blocks)))
                if i == self.high_reward_block:
                    reward += max(self.default_pink_reward, self.high_pink_reward * (
                        self.round_timeout - self.round_steps_count) / self.round_timeout)
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
            "reward": reward,
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
            "steps": np.array([self.round_steps_count / self.round_timeout], dtype=np.float32)
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

    def render(self) -> np.ndarray | None:

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
        """
        Flattens the observation space into a single vector.
        :param env:
        """
        super().__init__(env)
        obs_space = env.observation_space
        total_dim = (
            obs_space["agent_position"].shape[0]
            + obs_space["cyan"].feature_space.shape[0] * env.num_blocks // 2
            + obs_space["pink"].feature_space.shape[0] * env.num_blocks // 2
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


def generate_flat_bikkle_env(*args, **kwargs) -> gym.Env:
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


class BikkleSelfAttentionWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, *args, **kwargs):
        """
        This wrapper is used to convert the observation space of the BikkleGymEnvironment into a format suitable for
        use by the self-attention model.
        The observation space is divided into three parts: tokens, indices, and masks.
        - tokens: Contains the actual observation data.
        - indices: Contains the data type of the token according to data_type_idx. This is probably unnecessary and
        could be computed in the model directly.
        - masks: Contains the masks indicating which tokens are present. Counterintuitively, a True value indicates the
        token should be ignored (masked out). A False value indicates the token is present and should be used.

        Entries for eye_tracking and steps are added, but they are masked out in this wrapper.

        :param env:
        :param args:
        :param kwargs:
        """
        super().__init__(env)
        assert isinstance(env, BikkleGymEnvironment)
        self.max_blocks = max_blocks = env.num_blocks
        self.observation_space = Dict({
            "tokens": Dict({
                "agent_position": Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "block": Box(low=-1, high=1, shape=(max_blocks, 3), dtype=np.float32),
                "eye_tracking": Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "steps": Box(low=-1, high=1., shape=(1,), dtype=np.float32),
            }),
            "indices": Dict({
                "agent_position": Box(low=data_type_idx["agent_position"], high=data_type_idx["agent_position"],
                                      shape=(1,), dtype=np.int64),
                "block": Box(low=data_type_idx["block"], high=data_type_idx["block"], shape=(max_blocks,),
                             dtype=np.int64),
                "eye_tracking": Box(low=data_type_idx["eye_tracking"], high=data_type_idx["eye_tracking"], shape=(1,),
                                    dtype=np.int64),
                "steps": Box(low=data_type_idx["steps"], high=data_type_idx["steps"], shape=(1,), dtype=np.int64),
            }),
            "mask": Dict({
                "agent_position": Box(low=0, high=1, shape=(1,), dtype=np.bool_),
                "block": Box(low=0, high=1, shape=(max_blocks,), dtype=np.bool_),
                "eye_tracking": Box(low=0, high=1, shape=(1,), dtype=np.bool_),
                "steps": Box(low=0, high=1, shape=(1,), dtype=np.bool_),
            }),
        })

    def observation(self, observation: dict) -> dict:
        # Extract agent position
        agent_position = observation["agent_position"]

        # Combine cyan and pink blocks
        cyan_blocks = observation["cyan"]
        pink_blocks = observation["pink"]
        num_cyan = cyan_blocks.shape[0]
        num_pink = pink_blocks.shape[0]

        # Add a column to indicate block type (cyan=1, pink=0)
        cyan_with_type = np.hstack((cyan_blocks, np.ones((num_cyan, 1), dtype=np.float32)))
        pink_with_type = np.hstack((pink_blocks, np.zeros((num_pink, 1), dtype=np.float32)))

        # Stack cyan and pink blocks
        blocks = np.vstack((cyan_with_type, pink_with_type))

        # Pad blocks to max_blocks
        padded_blocks = np.zeros((self.max_blocks, 3), dtype=np.float32)
        padded_blocks[:blocks.shape[0]] = blocks

        # Create mask for blocks
        block_mask = np.ones((self.max_blocks,), dtype=np.bool_)
        block_mask[:blocks.shape[0]] = 0  # Mark present blocks as 0

        # Eye tracking (set to zeros)
        eye_tracking = np.zeros((2,), dtype=np.float32)
        eye_tracking_mask = np.ones((1,), dtype=np.bool_)  # Mark as absent

        steps = (np.float32(observation["steps"]) - 0.5) * 2.0
        steps_mask = np.ones((1,), dtype=np.bool_)  # Mark as absent

        # Indices
        agent_position_index = np.array([data_type_idx["agent_position"]], dtype=np.int64)
        block_indices = np.full((self.max_blocks,), data_type_idx["block"], dtype=np.int64)
        eye_tracking_index = np.array([data_type_idx["eye_tracking"]], dtype=np.int64)
        steps_index = np.array([data_type_idx["steps"]], dtype=np.int64)

        # Combine into the final observation
        return {
            "tokens": {
                "agent_position": agent_position,
                "block": padded_blocks,
                "eye_tracking": eye_tracking,
                "steps": steps,
            },
            "indices": {
                "agent_position": agent_position_index,
                "block": block_indices,
                "eye_tracking": eye_tracking_index,
                "steps": steps_index,
            },
            "mask": {
                "agent_position": np.array([0], dtype=np.bool_),  # Always present
                "block": block_mask,
                "eye_tracking": eye_tracking_mask,
                "steps": steps_mask,
            },
        }


def generate_self_attention_wrapper(*args, **kwargs) -> gym.Env:
    env = BikkleGymEnvironment(*args, **kwargs)
    return TimeLimit(BikkleSelfAttentionWrapper(env), max_episode_steps=100)


gym.envs.registration.register(
    id="BikkleSelfAttention-v0",
    entry_point=lambda *args, **kwargs: generate_self_attention_wrapper(*args, **kwargs),
)


class EyeTrackingObservationWrapper(ObservationWrapper):

    def __init__(self, env: gym.Env, fps: int = 10, camera: int = 2, *args, **kwargs):
        """
        Adds eye tracking data to the observation space. The eye tracking data is obtained from the EyeGestures library.
        This wrapper is intended to be run with the user. It displays the arena to screen and collects eye tracking data
        from the user.

        Draws a fullscreen window with the arena in the center.

        Eye tracking location is indicated as a red crosshair. If eye tracking is not available, the background of
        the surround screen turns to white and the eye tracking tokens and steps tokens are masked out.

        :param env:
        :param fps:
        :param camera: The camera id to use.
        :param args:
        :param kwargs:
        """
        assert isinstance(env, BikkleSelfAttentionWrapper)
        super().__init__(env)
        original_obs_space = env.observation_space

        self.gestures = EyeGestures_v3(500)

        # saving and loading in the EyeGestures library is not working yet.
        self.model_path = Path("./gestures.pkl")
        # if self.model_path.exists():
        #     with open(self.model_path, "rb") as f:
        #         self.gestures.loadModel(f.read())

        self.cap = VideoCapture(camera)
        self.fps = fps

        self.observation_space = original_obs_space

        self.screen_info = pygame.display.Info()
        self.screen_width = self.screen_info.current_w
        self.screen_height = self.screen_info.current_h
        self.image_size = min(self.screen_width, self.screen_height)
        self.image_left_edge = (self.screen_width - self.image_size) / 2
        self.image_right_edge = self.image_left_edge + self.image_size
        self.image_top_edge = (self.screen_height - self.image_size) / 2
        self.image_bottom_edge = self.image_top_edge + self.image_size

        self.window = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.FULLSCREEN)

        self.gestures.setFixation(1.0)

        self.last_eye_tracking = np.array([0, 0], dtype=np.float32)
        self.last_event_point = None

        x = np.linspace(self.image_left_edge / self.screen_width, self.image_right_edge / self.screen_width, num=10)
        y = np.linspace(self.image_top_edge / self.screen_height, self.image_bottom_edge / self.screen_height, num=10)

        xx, yy = np.meshgrid(x, y)
        calibration_map = np.column_stack([xx.ravel(), yy.ravel()])
        np.random.shuffle(calibration_map)
        self.gestures.uploadCalibrationMap(calibration_map)
        self.calibrate(50)
        self.clock = pygame.time.Clock()

        self.base_env = self.env
        while not isinstance(self.base_env, BikkleGymEnvironment):
            self.base_env = self.base_env.env

    def calibrate(self, count: int) -> None:
        """
        Calibrates the eye tracking data by displaying a series of points on the screen that the user should look at.
        :param count: number of calibration points to collect
        :return:
        """
        prev = np.array([0, 0])
        iterator = 0
        BLUE = (100, 0, 255)  # surrounding tolerance area of the eye tracking point
        RED = (255, 0, 100)  # center of the eye tracking point
        GREEN = (0, 255, 0)  # eye tracking point
        clock = pygame.time.Clock()
        while iterator < count:
            self.window.fill((0, 0, 0))  # Clear the window with black
            ret, frame = self.cap.read()
            try:
                event, cevent = self.gestures.step(frame,
                                                   True,
                                                   self.screen_width,
                                                   self.screen_height, )
                pygame.draw.circle(self.window, BLUE, cevent.point, cevent.acceptance_radius)
                pygame.draw.circle(self.window, RED, cevent.point, 50)
                if not np.array_equal(cevent.point, prev):
                    iterator += 1
                    prev = cevent.point
                pygame.draw.circle(self.window, GREEN, event.point, 50)
                pygame.display.flip()
            except:
                # will occur if eye tracking is not available, for example, if the face is hidden.
                pass

            clock.tick(60)
        self.window.fill((0, 0, 0))  # Clear the window with black
        self.last_event_point = None
        self.last_eye_tracking = np.array([0, 0], dtype=np.float32)

        # with open(self.model_path, "wb") as file:
        #     file.write(self.gestures.saveModel())

    def observation(self, observation):
        # already calibrated before running.
        ret, frame = self.cap.read()

        mask_val = 1  # mask out the eye tracking and steps data by default
        eye_tracking = self.last_eye_tracking
        try:
            event, _ = self.gestures.step(frame, False, self.screen_width, self.screen_height)
            self.last_event_point = event.point

            # Check if the point lies inside the image window, if not we mask it out
            if (self.image_left_edge <= event.point[0] <= self.image_right_edge and
                self.image_top_edge <= event.point[1] <= self.image_bottom_edge):
                mask_val = 0

            # image. The image is centered in the screen, so its center and the screen center are the same.
            # The fake eye tracking data it was trained with has -1 to 1 range, with 0 being the image center.
            # The point data is in screen coordinates, meaning 0 to screen_width and 0 to screen_height, in floating
            # point.
            # 1. crop the point to the image
            # outputs are in the range 0 to image_size
            cropped_x = max(0, min(event.point[0] - (self.screen_width - self.image_size) // 2, self.image_size))
            cropped_y = max(0, min(event.point[1] - (self.screen_height - self.image_size) // 2, self.image_size))

            # 2. normalize the point to the image size
            normalized_x = cropped_x / self.image_size
            normalized_y = cropped_y / self.image_size

            # 3. scale the point to the range -1 to 1
            eye_tracking = np.array([normalized_x, normalized_y], dtype=np.float32) * 2.0 - 1.0
            self.last_eye_tracking = eye_tracking

        except:
            pass

        observation["tokens"]["eye_tracking"] = eye_tracking
        observation["mask"]["eye_tracking"] = np.array([mask_val], dtype=np.bool_)
        observation["mask"]["steps"] = np.array([mask_val], dtype=np.bool_)
        return observation

    def reset(self, seed: int = None, options: dict = None) -> tuple[dict, dict]:
        self.calibrate(5)  # Call the calibrate routine before resetting
        return super().reset(seed=seed)

    def step(self, action):
        obs, reward, truncated, terminated, info = super().step(action)

        # TODO: figure out how to keep the refresh rate high for the eye tracking but limit the step rate to 10 fps.
        # obvious steps haven't worked as expected.
        for i in range(1):
            obs = self.observation(obs)
            self.window.fill((255, 255, 255) if obs["mask"]["eye_tracking"][0] else (
                0, 0, 0))  # Set background to white if mask is True
            window_width, window_height = self.window.get_size()

            scaled_image = pygame.transform.scale(
                pygame.surfarray.make_surface(self.base_env._human_image),
                (self.image_size, self.image_size)
            )

            image_width, image_height = scaled_image.get_size()
            x_offset = (window_width - image_width) // 2
            y_offset = (window_height - image_height) // 2

            self.window.blit(scaled_image, (x_offset, y_offset))
            pygame.draw.rect(self.window, (255, 255, 255), (x_offset, y_offset, image_width, image_height),
                             2)  # White frame

            if self.last_event_point is not None:
                # draw a red crosshair at the last event point, this is the eye tracking point as projected across the
                # entire screen
                pygame.draw.line(self.window, (255, 0, 0), (self.last_event_point[0] - 10, self.last_event_point[1]),
                                 (self.last_event_point[0] + 10, self.last_event_point[1]), 2)
                pygame.draw.line(self.window, (255, 0, 0), (self.last_event_point[0], self.last_event_point[1] - 10),
                                 (self.last_event_point[0], self.last_event_point[1] + 10), 2)

                # Draw a small red circle based on self.last_eye_tracking. This is the eye tracking point as projected onto
                # the image. Recall that self.last_eye_tracking is in the range -1 to 1, with 0 being the image center.
                eye_x = int((self.last_eye_tracking[0] + 1) / 2 * self.image_size)
                eye_y = int((self.last_eye_tracking[1] + 1) / 2 * self.image_size)
                pygame.draw.circle(self.window, (255, 0, 0), (eye_x + x_offset, eye_y + y_offset), 5)

            pygame.display.flip()
            self.clock.tick(10)

        return obs, reward, truncated, terminated, info


def generate_eye_tracking_env(*args, **kwargs) -> gym.Env:
    env = BikkleGymEnvironment(*args, **kwargs)
    env = BikkleSelfAttentionWrapper(env)
    env = EyeTrackingObservationWrapper(env, *args, **kwargs)
    return TimeLimit(env, max_episode_steps=200)


gym.envs.registration.register(
    id="BikkleEyeTracking-v0",
    entry_point=lambda *args, **kwargs: generate_eye_tracking_env(*args, **kwargs),
)


class FakeEyeTrackingObservationWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env, uniform: bool = False, *args, **kwargs):
        """
        A wrapper that adds fake eye-tracking data to the observation. This is done by adding random noise:
        - If uniform is True, the data is generated uniformly across the whole arena
        - If uniform is False, the data is generated as a Gaussian centered on the high reward block.

        Both eye_tracking and steps tokens are unmasked in the observation.

        Args:
            env (gym.Env): The environment to wrap.
            uniform: If True, the face eye-tracking data is generated uniformly otherwise it is Gaussian centered on
            the high reward block.
        """
        super().__init__(env)

        assert isinstance(env, BikkleSelfAttentionWrapper)

        self.base_env = env
        self.uniform = uniform
        while not isinstance(self.base_env, BikkleGymEnvironment):
            self.base_env = env.env

        assert isinstance(self.base_env, BikkleGymEnvironment)

    def observation(self, observation):
        """
        Adds fake eye-tracking data to the observation.

        Args:
            observation (dict): The original observation.

        Returns:
            dict: The modified observation with fake eye-tracking data.
        """

        if self.uniform:
            # generate a fake high reward block across the whole arena
            high_reward_block = np.random.uniform(0, 1, size=(2,))
        else:
            high_reward_block = self.base_env.pink_blocks[self.base_env.high_reward_block]
        fake_eye_tracking = np.clip(high_reward_block + np.random.normal(loc=0.0, scale=0.05, size=2), 0, 1)

        # normalize to [-1, 1]
        observation["tokens"]["eye_tracking"] = np.float32((fake_eye_tracking - 0.5) * 2.0)

        # Normalize the fake eye-tracking data to the range [-1, 1]
        if np.random.uniform() > 0.75:
            observation["mask"]["eye_tracking"] = np.array([0], dtype=np.bool_)  # Mark as present
            observation["mask"]["steps"] = np.array([0], dtype=np.bool_)  # Mark as present

        return observation


def generate_fake_eye_tracking_env(*args, **kwargs) -> gym.Env:
    env = BikkleGymEnvironment(*args, **kwargs)
    env = BikkleSelfAttentionWrapper(env)
    env = FakeEyeTrackingObservationWrapper(env, *args, **kwargs)
    return TimeLimit(env, max_episode_steps=200)


gym.envs.registration.register(
    id="BikkleFakeEyeTracking-v0",
    entry_point=lambda *args, **kwargs: generate_fake_eye_tracking_env(*args, **kwargs),
)

gym.envs.registration.register(
    id="BikkleFakeEyeTracking-v0-uniform",
    entry_point=lambda *args, **kwargs: generate_fake_eye_tracking_env(*args, **kwargs, uniform=True),
)


class BikkleSemanticImageObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, image_size: int = 64):
        """
        Wraps the BikkleGymEnvironment to convert observations into a 4-channel image.
        Not actively used.

        Args:
            env (gym.Env): The environment to wrap.
            image_size (int): The width and height of the square image.
        """
        super().__init__(env)
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(4, image_size, image_size), dtype=np.float32
        )

    def observation(self, observation: dict) -> np.ndarray:
        """
        Converts the observation into a 4-channel image.

        Args:
            observation (dict): The original observation from the environment.

        Returns:
            np.ndarray: A 4-channel image representation of the observation.
        """
        # Initialize the 4-channel image
        image = np.zeros((4, self.image_size, self.image_size), dtype=np.float32)

        # Extract agent position and normalize to image coordinates
        agent_x, agent_y = observation["agent_position"]
        agent_x = int(agent_x * (self.image_size - 1))
        agent_y = int(agent_y * (self.image_size - 1))

        # Set the agent position in channel 1
        image[0, agent_y, agent_x] = 1.0

        # Process cyan blocks
        for block in observation["cyan"]:
            block_x = int((agent_x + block[0]) * (self.image_size - 1))
            block_y = int((agent_y + block[1]) * (self.image_size - 1))
            if 0 <= block_x < self.image_size and 0 <= block_y < self.image_size:
                image[1, block_y, block_x] = 1.0

        # Process pink blocks
        for block in observation["pink"]:
            block_x = int((agent_x + block[0]) * (self.image_size - 1))
            block_y = int((agent_y + block[1]) * (self.image_size - 1))
            if 0 <= block_x < self.image_size and 0 <= block_y < self.image_size:
                image[2, block_y, block_x] = 1.0

        # Channel 4 remains all zeros
        return image


def generate_semantic_image_wrapper(*args, **kwargs) -> gym.Env:
    env = BikkleGymEnvironment(*args, **kwargs)
    return TimeLimit(BikkleSemanticImageObservationWrapper(env), max_episode_steps=100)


gym.envs.registration.register(
    id="BikkleSemanticImage-v0",
    entry_point=lambda *args, **kwargs: generate_semantic_image_wrapper(*args, **kwargs),
)
