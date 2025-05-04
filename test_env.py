import unittest
import numpy as np
from env import BikkleGymEnvironment


class TestBikkleGymEnvironment(unittest.TestCase):
    def setUp(self):
        """Set up the environment for testing."""
        self.env = BikkleGymEnvironment()

    def test_initialization(self):
        """Test if the environment initializes correctly."""
        self.assertEqual(self.env.screen_size, 600)
        self.assertEqual(self.env.num_blocks, 10)
        self.assertEqual(self.env.round_timeout, 100)

    def test_reset(self):
        """Test the reset functionality."""
        observation, info = self.env.reset()
        self.assertIn("agent_position", observation)
        self.assertIn("cyan_blocks", observation)
        self.assertIn("pink_blocks", observation)
        self.assertIn("screen_image", observation)
        self.assertIn("high_reward_block", info)
        self.assertEqual(len(observation["cyan_blocks"]), self.env.num_blocks // 2)
        self.assertEqual(len(observation["pink_blocks"]), self.env.num_blocks // 2)

    def test_step(self):
        """Test the step functionality."""
        self.env.reset()
        action = np.array([0.5, -0.5], dtype=np.float32)
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.assertIn("agent_position", observation)
        self.assertIn("cyan_blocks", observation)
        self.assertIn("pink_blocks", observation)
        self.assertIn("screen_image", observation)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("high_reward_block", info)

    def test_render(self):
        """Test the render functionality."""
        self.env.reset()
        screen_image = self.env.render(mode="rgb_array")
        self.assertIsInstance(screen_image, np.ndarray)
        self.assertEqual(screen_image.shape, (self.env.screen_size, self.env.screen_size, 3))

    def test_collision_with_cyan_block(self):
        """Test collision with a cyan block."""
        self.env.reset()
        self.env.cyan_blocks = [np.array([50, 50], dtype=np.float32)]  # Place a cyan block at a fixed position
        self.env.agent_position = np.array([50, 50], dtype=np.float32)  # Place the agent at the same position
        _, reward, _, _, _ = self.env.step(np.array([0, 0], dtype=np.float32))
        self.assertEqual(reward, self.env.cyan_penalty)

    def test_collision_with_pink_block(self):
        """Test collision with a pink block."""
        self.env.reset()
        self.env.pink_blocks = [np.array([50, 50], dtype=np.float32)]  # Place a pink block at a fixed position
        self.env.agent_position = np.array([50, 50], dtype=np.float32)  # Place the agent at the same position
        self.env.high_reward_block = 0  # Set the first pink block as the high reward block
        _, reward, _, _, _ = self.env.step(np.array([0, 0], dtype=np.float32))
        self.assertEqual(reward, self.env.high_pink_reward)

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()


if __name__ == "__main__":
    unittest.main()