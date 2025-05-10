import argparse
import dataclasses
import math
import os
from pathlib import Path

import gymnasium as gym
import torch
import tyro
from sentry_sdk.integrations import typer

from env import generate_flat_bikkle_env, EyeGestures_v3, EyeTrackingObservationWrapper
from eyeGestures.utils import VideoCapture
import pygame

from leanrl_ppo_baseline import Agent

@dataclasses.dataclass
class Args:
    saved_model_path: Path
    calibration_steps: int = 30

BLUE = (100, 0, 255)

def main(args):
    # Step 1: Generate a FlatBikkle environment
    env = generate_flat_bikkle_env()

    n_act = math.prod(env.action_space.shape)
    n_obs = math.prod(env.observation_space.shape)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # Initialize the agent
    agent = Agent(n_obs, n_act, device=device)

    # Load the model if a path is provided
    if args.model_path:
        if os.path.exists(args.model_path):
            agent.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Model loaded from {args.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {args.model_path}")

    # Ensure the agent is in evaluation mode after loading
    agent.eval()

    # Step 2: Show the screen to the user
    env.render(mode="human")

    # Step 3: Set up eyeGestures to capture eyetracking data relative to the window
    gestures = EyeGestures_v3()
    cap = VideoCapture(2)  # Assuming camera index 2, adjust as needed

    # Step 4: Run the calibration routine
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h

    for i in range(args.calibration_steps):
        frame = cap.read()
        event, calibration = gestures.step(frame, True, screen_width, screen_height, context="my_context")
        pygame.draw.circle(screen, BLUE, calibration.point, calibration.acceptance_radius)


    # Step 5: Create an EyeTrackingObservationWrapper
    wrapped_env = EyeTrackingObservationWrapper(env, gestures)

    # Example usage of the wrapped environment
    obs = wrapped_env.reset()
    print("Initial observation:", obs)

    # Close the environment properly
    wrapped_env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))