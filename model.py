import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import math

import gymnasium.spaces as spaces
from gymnasium.spaces import Sequence
import gymnasium as gym



def create_sinusoidal_embedding(num_positions: int, embedding_dim: int) -> nn.Parameter:
    position = torch.arange(num_positions).unsqueeze(1)  # Shape: (num_positions, 1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
    sinusoidal_embedding = torch.zeros(num_positions, embedding_dim)
    sinusoidal_embedding[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
    sinusoidal_embedding[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
    return nn.Parameter(sinusoidal_embedding, requires_grad=False)

def assert_is_nan(x: dict[str, torch.Tensor]) -> None:
    for key, value in x.items():
        assert not torch.isnan(value).any(), f"Nan detected in {key}"

def assert_is_inf(x: dict[str, torch.Tensor]) -> None:
    for key, value in x.items():
        assert not torch.isinf(value).any(), f"Inf detected in {key}"

class BaseBikkleModel(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, token_size: int = 64,
                 num_attention_heads: int = 4):
        super(BaseBikkleModel, self).__init__()

        num_obs_types = len(observation_space.keys())

        self.token_size = token_size

        # Embedding offsets for observations and actions
        self.embedding_offsets = nn.Embedding(num_obs_types, token_size)

        def make_one_embedding(space):
            if isinstance(space, Sequence):
                space = space.feature_space

            return nn.Sequential(
                nn.Linear(space.shape[0], token_size),
                nn.LayerNorm(token_size),
            )

        token_dict = {
            key: make_one_embedding(space)
            for key, space in observation_space.items()
        }

        # MLPs for each observation key to map to token size
        self.input_mlps = nn.ModuleDict(token_dict)

        # Self-attention block
        self.self_attention = nn.MultiheadAttention(embed_dim=token_size, num_heads=num_attention_heads,
                                                    batch_first=True)
        self.post_attention_layer_norm = nn.LayerNorm(token_size)

    def forward(self, observations: dict[str, torch.Tensor], embedding_indices: dict[str, torch.Tensor],
                mask: dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param tokens: A tensor of shape (batch_size, num_tokens, token_size) representing the input tokens
        :param embedding_indices: Corresponding indices for the tokens for the embedding type. Shape (batch_size, num_tokens)
        :return:
        """
        token_list = []
        indices_list = []
        mask_list = []

        # I expect that the inputs to this method are consistent in shape and keys, so I don't think this
        # would retrigger tracing on the compute graph if we put this inside a compile block.
        # Process each observation key
        for i, (key, value) in enumerate(observations.items()):
            token = self.input_mlps[key](value)
            token_list.append(token)
            indices_list.append(embedding_indices[key])
            mask_list.append(mask[key])

        # Stack tokens and pass through self-attention
        tokens_tensor = torch.cat(token_list, dim=1)  # Shape: (batch_size, num_tokens, token_size)
        indices_tensor = torch.cat(indices_list, dim=1)  # Shape: (batch_size, num_tokens, token_size)
        mask_tensor = torch.cat(mask_list, dim=1)

        tokens_tensor += self.embedding_offsets(indices_tensor)

        attn_output, attn_weights = self.self_attention(tokens_tensor, tokens_tensor, tokens_tensor, key_padding_mask=mask_tensor, need_weights=True)
        attn_output = self.post_attention_layer_norm(attn_output)

        aggregated_tokens = torch.bmm(attn_weights, attn_output)  # Shape: (batch_size, num_tokens, token_size)
        aggregated_tokens = aggregated_tokens.sum(dim=1)  # Shape: (batch_size, token_size)

        return aggregated_tokens


class BikkleValueFunction(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict,
                 token_size: int = 64, num_attention_heads: int = 4, mlp_hidden_size: int = 128) -> None:
        super(BikkleValueFunction, self).__init__()

        assert isinstance(observation_space, spaces.Dict)
        observation_space = copy.deepcopy(observation_space)

        self.base = BaseBikkleModel(observation_space, token_size=token_size,
                                    num_attention_heads=num_attention_heads)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(token_size, mlp_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(mlp_hidden_size),
            nn.Linear(mlp_hidden_size, 1)  # Single output head
        )

    def forward(self, observations: dict[str, dict[torch.Tensor]]) -> torch.Tensor:
        tokens = observations["tokens"]
        indices = observations["indices"]
        mask = observations["mask"]
        # assert_is_nan(tokens)
        # assert_is_inf(tokens)
        # assert_is_nan(indices)
        # assert_is_inf(indices)
        # assert_is_nan(mask)
        # assert_is_inf(mask)
        # assert not torch.isnan(action).any(), "Nan detected in value action"
        # assert not torch.isinf(action).any(), "Inf detected in value action"

        aggregated_tokens = self.base(tokens, indices, mask)

        output = self.mlp(aggregated_tokens)

        # assert not torch.isnan(output).any(), "Nan detected in value prediction"
        # assert not torch.isinf(output).any(), "Inf detected in value prediction"

        return output
class BikkleActionValueFunction(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Box,
                 token_size: int = 64, num_attention_heads: int = 4, mlp_hidden_size: int = 128) -> None:
        super(BikkleActionValueFunction, self).__init__()

        assert isinstance(action_space, spaces.Box)
        assert isinstance(observation_space, spaces.Dict)
        observation_space = copy.deepcopy(observation_space)
        observation_space["action"] = action_space

        self.base = BaseBikkleModel(observation_space, token_size=token_size,
                                    num_attention_heads=num_attention_heads)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(token_size, mlp_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(mlp_hidden_size),
            nn.Linear(mlp_hidden_size, 1)  # Single output head
        )

    def forward(self, observations: dict[str, dict[torch.Tensor]], action: torch.Tensor) -> torch.Tensor:
        tokens = observations["tokens"]
        indices = observations["indices"]
        mask = observations["mask"]
        # assert_is_nan(tokens)
        # assert_is_inf(tokens)
        # assert_is_nan(indices)
        # assert_is_inf(indices)
        # assert_is_nan(mask)
        # assert_is_inf(mask)
        # assert not torch.isnan(action).any(), "Nan detected in value action"
        # assert not torch.isinf(action).any(), "Inf detected in value action"

        tokens["action"] = torch.unsqueeze(action, dim=1)
        indices["action"] = torch.full((action.shape[0], 1), key_to_idx["action"], dtype=torch.long, device=action.device)
        mask["action"] = torch.zeros((action.shape[0], 1), dtype=torch.bool, device=action.device)
        aggregated_tokens = self.base(tokens, indices, mask)

        output = self.mlp(aggregated_tokens)

        # assert not torch.isnan(output).any(), "Nan detected in value prediction"
        # assert not torch.isinf(output).any(), "Inf detected in value prediction"

        return output


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class BikklePolicy(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Box,
                 token_size: int = 64, num_attention_heads: int = 4, mlp_hidden_size: int = 128) -> None:
        super(BikklePolicy, self).__init__()

        self.base = BaseBikkleModel(observation_space, token_size=token_size,
                                    num_attention_heads=num_attention_heads)

        # Final MLP for Gaussian heads
        self.mean_head = nn.Sequential(
            nn.Linear(token_size, mlp_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(mlp_hidden_size),
            nn.Linear(mlp_hidden_size, action_space.shape[0])  # Output mean for each action dimension
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(token_size, mlp_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(mlp_hidden_size),
            nn.Linear(mlp_hidden_size, action_space.shape[0])  # Output log standard deviation for each action dimension
        )

    def forward(self, tokens: dict[str, torch.Tensor], indices: dict[str, torch.Tensor],
                mask: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """

        :param tokens: A tensor of shape (batch_size, num_tokens, token_size) representing the input tokens
        :param indices: Corresponding indices for the tokens for the embedding type. Shape (batch_size, num_tokens)
        :return:
        """

        aggregated_tokens = self.base(tokens, indices, mask)

        # Compute mean and log standard deviation
        mean = self.mean_head(aggregated_tokens)
        log_std = self.log_std_head(aggregated_tokens)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-5, max=2)
        #
        # log_std = torch.tanh(log_std)
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, tokens: dict[str, torch.Tensor], indices: dict[str, torch.Tensor],
                   mask: dict[str, torch.Tensor], greedy: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(tokens, indices, mask)
        std = log_std.exp()
        assert not torch.isnan(mean).any(), "NaN in mean"
        assert not torch.isinf(mean).any(), "Inf in mean"
        assert not torch.isnan(std).any(), "NaN in std"
        assert not torch.isinf(std).any(), "Inf in std"
        normal = torch.distributions.Normal(mean, std)
        entropy = normal.entropy()
        x_t = mean if greedy else normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        action = torch.tanh(x_t)
        # action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        # log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, entropy.sum(axis=1, keepdim=True)


key_to_idx = {
    "agent_position": 0,
    "cyan": 1,
    "pink": 2,
    "steps": 3,
    "action": 4,
    "face": 5,
    "eye_tracking": 6,
    "screen_image": 3  # do not use
}


def preprocess_bikkle_observation_with_mask(
    observation: dict[str, np.ndarray | list | torch.Tensor],
    observation_space: gym.spaces.Dict,
    max_blocks: int = 100,
    device: str = "cpu"
) -> dict[str, dict[str, torch.Tensor]]:

    max_blocks = max_blocks // 2 # TODO: temp hack

    """
    TODO: I'm applying padding to the blocks, but I shouldn't really worry about that, I should apply padding over the
    whole batch token set instead. I don't have a great way to handle that right now though because I'm stacking up
    the block tokens across the batch, so they need to be the same length.

    :param observation:
    :param observation_space:
    :param max_blocks:
    :param device:
    :return:
    """
    tokens = {}
    indices = {}
    masks = {}

    for key, value in observation.items():
        if key == "screen_image":  # Skip screen images for this model
            continue

        if isinstance(observation_space[key], Sequence):
            # Handle variable-length sequences (e.g., cyan and pink blocks)
            feature_space = observation_space[key].feature_space
            batch_size = len(value)
            padded_tensor = torch.zeros((batch_size, max_blocks, feature_space.shape[0]), dtype=torch.float32,
                                        device=device)

            # TODO: right now we're assuming that each block type has the same size, the code is a bit mixed on this
            if isinstance(value, torch.Tensor):
                tokens[key] = value.to(device)
                indices[key] = torch.full((value.shape[0], value.shape[1]), key_to_idx[key], dtype=torch.long,
                                          device=device)
                masks[key] = torch.zeros((value.shape[0], value.shape[1]), dtype=torch.bool, device=device)
            else:
                sequence_tensor = [torch.tensor(seq, dtype=torch.float32, device=device) for seq in value]

                # Create a mask for each batch
                mask = torch.ones((batch_size, max_blocks), dtype=torch.bool, device=device)

                for i, seq in enumerate(sequence_tensor):
                    length = min(len(seq), max_blocks)
                    padded_tensor[i, :length] = seq[:length]
                    mask[i, :length] = 0  # Mark valid tokens
                tokens[key] = padded_tensor
                indices[key] = torch.full((batch_size, max_blocks), key_to_idx[key], dtype=torch.long, device=device)
                masks[key] = mask

            tokens[key] = tokens[key]*2 - 1.0  # Normalize to [-1, 1]
            assert not torch.isnan(tokens[key]).any(), f"NaN in tokens for key {key}"
            assert not torch.isinf(tokens[key]).any(), f"Inf in tokens for key {key}"
        else:
            # Handle fixed-size observations
            if value is None:
                raise ValueError(f"Observation key '{key}' has a None value.")
            if not isinstance(value, (torch.Tensor, list, np.ndarray)):
                raise TypeError(f"Unexpected type for observation key '{key}': {type(value)}")

            tensor = torch.tensor(value, dtype=torch.float32, device=device) if not isinstance(value,
                                                                                               torch.Tensor) else value.to(
                device)
            tensor = torch.unsqueeze(tensor, dim=1)  # add a token dimension
            tokens[key] = tensor*2 - 1.0  # Normalize to [-1, 1]
            assert not torch.isnan(tokens[key]).any(), f"NaN in tokens for key {key}"
            assert not torch.isinf(tokens[key]).any(), f"Inf in tokens for key {key}"
            indices[key] = torch.full((tensor.shape[0], 1), key_to_idx[key], dtype=torch.long, device=device)
            masks[key] = torch.zeros((tensor.shape[0], 1), dtype=torch.bool, device=device)

    return {"tokens": tokens, "indices": indices, "mask": masks}

class SimpleValueFunction(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Box,
                 num_cyan: int, num_pink: int, mlp_hidden_size: int = 128) -> None:
        super(SimpleValueFunction, self).__init__()

        assert isinstance(action_space, spaces.Box)
        assert isinstance(observation_space, spaces.Dict)

        # Calculate the input size based on the observation space and action space
        agent_position_size = observation_space["agent_position"].shape[0]
        cyan_size = num_cyan * observation_space["cyan"].feature_space.shape[0]
        pink_size = num_pink * observation_space["pink"].feature_space.shape[0]
        action_size = action_space.shape[0]

        input_size = agent_position_size + cyan_size + pink_size + action_size

        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1)  # Single output head
        )

    def forward(self, observations: dict[str, dict[torch.Tensor]], action: torch.Tensor) -> torch.Tensor:
        # Extract and concatenate inputs in the specified order
        agent_position = observations["tokens"]["agent_position"]
        agent_position = agent_position.view(agent_position.shape[0], -1)
        cyan = observations["tokens"]["cyan"].view(agent_position.shape[0], -1)  # Flatten cyan
        pink = observations["tokens"]["pink"].view(agent_position.shape[0], -1)  # Flatten pink
        action = action

        # Concatenate all inputs
        inputs = torch.cat([agent_position, cyan, pink, action], dim=1)

        # Pass through the MLP
        output = self.mlp(inputs)
        return output

class SimplePolicy(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, num_cyan: int, num_pink: int,
                 action_space: gym.spaces.Box, mlp_hidden_size: int = 128, log_std_min: float = -2.0, log_std_max: float=2.0) -> None:
        super(SimplePolicy, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        assert isinstance(action_space, spaces.Box)
        assert isinstance(observation_space, spaces.Dict)

        # Calculate the input size based on the observation space
        agent_position_size = observation_space["agent_position"].shape[0]
        cyan_size = num_cyan * observation_space["cyan"].feature_space.shape[0]
        pink_size = num_pink * observation_space["pink"].feature_space.shape[0]

        input_size = agent_position_size + cyan_size + pink_size

        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )

        # Gaussian output layers
        self.mean_head = nn.Linear(mlp_hidden_size, action_space.shape[0])
        self.log_std_head = nn.Linear(mlp_hidden_size, action_space.shape[0])

    def forward(self, observations: dict[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract and concatenate inputs in the specified order
        agent_position = observations["agent_position"]
        agent_position = agent_position.view(agent_position.shape[0], -1)
        cyan = observations["cyan"].view(agent_position.shape[0], -1)  # Flatten cyan
        pink = observations["pink"].view(agent_position.shape[0], -1)  # Flatten pink

        # Concatenate all inputs
        inputs = torch.cat([agent_position, cyan, pink], dim=1)

        # Pass through the MLP
        features = self.mlp(inputs)

        # Compute mean and log standard deviation
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return mean, log_std

    def get_action(self, tokens: dict[str, torch.Tensor], indices: dict[str, torch.Tensor],
                   mask: dict[str, torch.Tensor], greedy: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(tokens)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = mean if greedy else normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean