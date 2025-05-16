import copy

import numpy as np
import torch
import torch.nn as nn
import math

import gymnasium.spaces as spaces
from gymnasium.spaces import Sequence
import gymnasium as gym
from dataclasses import dataclass

from torch.nn import TransformerEncoderLayer, TransformerEncoder

data_type_idx = {
    "agent_position": 0,
    "block": 1,
    "eye_tracking": 2,
    "action": 3,
    "face": 4,
    "steps": 5,
}

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

def activation_fn(name: str) -> nn.Module:
    if name == "relu":
        activation = nn.ReLU()
    elif name == "gelu":
        activation = nn.GELU()
    else:
        raise ValueError(f"Unknown activation function: {name}")
    return activation


@dataclass
class BaseBikkleModelParams:
    """
    This is being used for both BaseBikkleModel and BaseBikkle2Model-> Not clean, but I didn't want to deal with
    clever config right now
    """
    token_size: int = 64
    num_attention_heads: int = 4
    num_layers: int = 1  # Number of TransformerEncoder layers

class BaseBikkleModel(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, params: BaseBikkleModelParams):
        super(BaseBikkleModel, self).__init__()

        num_obs_types = len(observation_space.keys())

        token_size = params.token_size
        num_attention_heads = params.num_attention_heads

        # Embedding offsets for observations and actions
        self.embedding_offsets = nn.Embedding(num_obs_types, token_size)

        # Pre-attention LayerNorm (pre-norm style)
        self.pre_attention_ln = nn.LayerNorm(token_size)

        def make_one_embedding(space):
            if isinstance(space, Sequence):
                space = space.feature_space

            return nn.Sequential(
                nn.Linear(space.shape[-1], token_size),
            )

        token_dict = {
            key: make_one_embedding(space)
            for key, space in observation_space["tokens"].items()
        }

        # MLPs for each observation key to map to token size
        self.tokenizer = nn.ModuleDict(token_dict)

        # Self-attention block
        self.self_attention = nn.MultiheadAttention(embed_dim=token_size, num_heads=num_attention_heads,
                                                    batch_first=True)

        # Post-residual LayerNorm
        self.post_attention_ln = nn.LayerNorm(token_size)

        self.post_attention_layer_norm = nn.LayerNorm(token_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier-style init for linears/embeddings/attention, and
           ones/zeros for LayerNorm."""
        if isinstance(module, nn.Linear):
            # TODO: this doesn't actually identify the 'ff1' layer
            nn.init.xavier_uniform_(module.weight,
                                    gain=math.sqrt(2) if 'ff1' in module._get_name().lower() else 1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

        elif isinstance(module, nn.MultiheadAttention):
            # in_proj contains QKV stacked
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.zeros_(module.in_proj_bias)
            # out projection
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.out_proj.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tokens: dict[str, torch.Tensor], type_indices: dict[str, torch.Tensor],
                mask: dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param tokens: A tensor of shape (batch_size, num_tokens, token_size) representing the input tokens
        :param type_indices: Corresponding indices for the tokens for the embedding type. Shape (batch_size, num_tokens)
        :return:
        """
        token_list = []
        indices_list = []
        mask_list = []

        # I expect that the inputs to this method are consistent in shape and keys, so I don't think this
        # would retrigger tracing on the compute graph if we put this inside a compile block.
        # Process each observation key
        # for i, (key, value) in enumerate(tokens.items()):
        #     token = self.tokenizer[key](value)
        #     token_list.append(token)
        #     indices_list.append(type_indices[key])
        #     mask_list.append(mask[key])

        token_list.append(self.tokenizer["agent_position"](tokens["agent_position"]).unsqueeze(1))
        indices_list.append(type_indices["agent_position"])
        mask_list.append(mask["agent_position"])

        token_list.append(self.tokenizer["block"](tokens["block"]))
        indices_list.append(type_indices["block"])
        mask_list.append(mask["block"])

        token_list.append(self.tokenizer["eye_tracking"](tokens["eye_tracking"]).unsqueeze(1))
        indices_list.append(type_indices["eye_tracking"])
        mask_list.append(mask["eye_tracking"])

        # Stack tokens and pass through self-attention
        tokens_tensor = torch.cat(token_list, dim=1)  # Shape: (batch_size, num_tokens, token_size)
        indices_tensor = torch.cat(indices_list, dim=1)  # Shape: (batch_size, num_tokens, token_size)
        mask_tensor = torch.cat(mask_list, dim=1)

        tokens_tensor += self.embedding_offsets(indices_tensor)

        tokens_tensor = self.pre_attention_ln(tokens_tensor)

        attn_output, attn_weights = self.self_attention(tokens_tensor, tokens_tensor, tokens_tensor, key_padding_mask=mask_tensor, need_weights=True)
        tokens_tensor += attn_output
        tokens_tensor = self.post_attention_layer_norm(tokens_tensor)

        aggregated_tokens = torch.bmm(attn_weights, tokens_tensor)  # Shape: (batch_size, num_tokens, token_size)
        aggregated_tokens = aggregated_tokens.sum(dim=1)  # Shape: (batch_size, token_size)

        return aggregated_tokens

class BaseBikkle2Model(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, params: BaseBikkleModelParams):
        super(BaseBikkle2Model, self).__init__()

        num_obs_types = len(data_type_idx)
        token_size = params.token_size
        num_attention_heads = params.num_attention_heads
        num_layers = params.num_layers

        # Embedding offsets for observations and actions
        self.embedding_offsets = nn.Embedding(num_obs_types, token_size)

        # Pre-attention LayerNorm (pre-norm style)
        self.pre_attention_ln = nn.LayerNorm(token_size)


        def make_one_embedding(space):
            if isinstance(space, Sequence):
                space = space.feature_space

            return nn.Linear(space.shape[-1], token_size)

        token_dict = {
            key: make_one_embedding(space)
            for key, space in observation_space["tokens"].items()
        }

        # MLPs for each observation key to map to token size
        self.tokenizer = nn.ModuleDict(token_dict)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=token_size,
            nhead=num_attention_heads,
            dim_feedforward=4 * token_size,  # Feedforward dimension
            activation="gelu",  # Use gelu activation
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Post-residual LayerNorm
        self.post_attention_layer_norm = nn.LayerNorm(token_size)

    def forward(self, tokens: dict[str, torch.Tensor], type_indices: dict[str, torch.Tensor],
                mask: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the BaseBikkleModel.

        :param tokens: A tensor of shape (batch_size, num_tokens, token_size) representing the input tokens.
        :param type_indices: Corresponding indices for the tokens for the embedding type. Shape (batch_size, num_tokens).
        :param mask: A tensor of shape (batch_size, num_tokens) representing the key padding mask.
        :return: Aggregated tokens of shape (batch_size, token_size).
        """
        token_list = []
        indices_list = []
        mask_list = []

        # Process each observation key
        for key in tokens.keys():
            token = self.tokenizer[key](tokens[key])
            if len(token.shape) == 2:
                token = token.unsqueeze(1)
            token_list.append(token)
            indices_list.append(type_indices[key])
            mask_list.append(mask[key])

        # Stack tokens and pass through TransformerEncoder
        tokens_tensor = torch.cat(token_list, dim=1)  # Shape: (batch_size, num_tokens, token_size)
        indices_tensor = torch.cat(indices_list, dim=1)  # Shape: (batch_size, num_tokens)
        mask_tensor = torch.cat(mask_list, dim=1)  # Shape: (batch_size, num_tokens)

        tokens_tensor = tokens_tensor + self.embedding_offsets(indices_tensor)
        # tokens_tensor = self.pre_attention_ln(tokens_tensor)

        # Pass through TransformerEncoder
        tokens_tensor = self.transformer_encoder(tokens_tensor, src_key_padding_mask=mask_tensor) # [batch, num_tokens, token_size]

        # Aggregate tokens
        aggregated_tokens = tokens_tensor.sum(dim=1)  # Shape: (batch_size, token_size)

        return aggregated_tokens


@dataclass
class BikkleValueFunctionParams:
    base_params: BaseBikkleModelParams = BaseBikkleModelParams()
    activation: str = "relu"
    mlp_hidden_size: int = 128

class BikkleValueFunction(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, params: BikkleValueFunctionParams) -> None:
        super(BikkleValueFunction, self).__init__()

        assert isinstance(observation_space, spaces.Dict)
        observation_space = copy.deepcopy(observation_space)

        token_size = params.base_params.token_size

        if params.base_params.num_layers > 1:
            self.base = BaseBikkle2Model(observation_space, params.base_params)
        else:
            self.base = BaseBikkleModel(observation_space, params.base_params)

        mlp_hidden_size = params.mlp_hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(token_size, mlp_hidden_size),
            activation_fn(params.activation),
            nn.Linear(mlp_hidden_size, 1)
        )

        assert isinstance(observation_space, spaces.Dict)


    def get_value(self, observations: dict[str, dict[torch.Tensor]]) -> torch.Tensor:
        tokens = observations["tokens"]
        indices = observations["indices"]
        mask = observations["mask"]

        aggregated_tokens = self.base(tokens, indices, mask)

        output = self.mlp(aggregated_tokens)

        return output

@dataclass
class BikkleActionValueFunctionParams:
    base_params: BaseBikkleModelParams = BaseBikkleModelParams()
    activation: str = "relu"
    mlp_hidden_size: int = 128

class BikkleActionValueFunction(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Box, params: BikkleActionValueFunctionParams) -> None:
        super(BikkleActionValueFunction, self).__init__()

        assert isinstance(action_space, spaces.Box)
        assert isinstance(observation_space, spaces.Dict)
        observation_space = copy.deepcopy(observation_space)
        observation_space["action"] = action_space

        self.base = BaseBikkleModel(observation_space, params.base_params)
        token_size = params.base_params.token_size
        mlp_hidden_size = params.mlp_hidden_size


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

        tokens["action"] = torch.unsqueeze(action, dim=1)
        indices["action"] = torch.full((action.shape[0], 1), data_type_idx["action"], dtype=torch.long, device=action.device)
        mask["action"] = torch.zeros((action.shape[0], 1), dtype=torch.bool, device=action.device)
        aggregated_tokens = self.base(tokens, indices, mask)

        output = self.mlp(aggregated_tokens)

        return output


LOG_STD_MAX = 2
LOG_STD_MIN = -5


@dataclass
class BikklePolicyParams:
    base_params: BaseBikkleModelParams = BaseBikkleModelParams()
    mlp_hidden_size: int = 128
    activation: str = "relu"
    dropout: float = 0.1

class BikklePolicy(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Box, params: BikklePolicyParams) -> None:
        super(BikklePolicy, self).__init__()

        token_size = params.base_params.token_size
        mlp_hidden_size = params.mlp_hidden_size

        if params.base_params.num_layers == 1:
            self.base = BaseBikkleModel(observation_space, params.base_params)
        else:
            self.base = BaseBikkle2Model(observation_space, params.base_params)

        # Final MLP for Gaussian heads
        self.mean_head = nn.Sequential(
            nn.Linear(token_size, mlp_hidden_size),
            activation_fn(params.activation),
            nn.Dropout(params.dropout),
            nn.Linear(mlp_hidden_size, action_space.shape[0])
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(token_size, mlp_hidden_size),
            activation_fn(params.activation),
            nn.Dropout(params.dropout),
            nn.Linear(mlp_hidden_size, action_space.shape[0])
        )

    def forward(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """

        :param tokens: A tensor of shape (batch_size, num_tokens, token_size) representing the input tokens
        :param indices: Corresponding indices for the tokens for the embedding type. Shape (batch_size, num_tokens)
        :return:
        """
        tokens = obs["tokens"]
        indices = obs["indices"]
        mask = obs["mask"]
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

    def get_action(self, obs, action = None, greedy: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = torch.exp(log_std)
        assert not torch.isnan(mean).any(), "NaN in mean"
        assert not torch.isinf(mean).any(), "Inf in mean"
        assert not torch.isnan(std).any(), "NaN in std"
        assert not torch.isinf(std).any(), "Inf in std"
        normal = torch.distributions.Normal(mean, std)

        # x_t = mean if greedy else normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # if greedy:
        #     x_t = mean
        # if action is None:
        #     # action = torch.tanh(x_t)
        #
        #
        # # action = y_t * self.action_scale + self.action_bias
        # log_prob = normal.log_prob(x_t)
        # # Enforcing Action Bound
        # # log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        # log_prob = log_prob.sum(1, keepdim=True)
        # # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # return action, log_prob, entropy.sum(axis=1, keepdim=True)

        if greedy:
            action = mean
        if action is None:
            action = mean + std * torch.randn_like(mean)
        return action, normal.log_prob(action).sum(1), normal.entropy().sum(1)


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
                indices[key] = torch.full((value.shape[0], value.shape[1]), data_type_idx[key], dtype=torch.long,
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
                indices[key] = torch.full((batch_size, max_blocks), data_type_idx[key], dtype=torch.long, device=device)
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
            indices[key] = torch.full((tensor.shape[0], 1), data_type_idx[key], dtype=torch.long, device=device)
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


@dataclass
class BikkleSemanticImagePolicyParams:
    cnn_channels: int = 32
    mlp_hidden_size: int = 128

class BikkleSemanticImagePolicy(nn.Module):
    def __init__(self, image_size: int, action_space: gym.spaces.Box, params: BikkleSemanticImagePolicyParams):
        """
        A CNN-based policy for processing 4-channel image observations.

        Args:
            image_size (int): The width and height of the square input image.
            action_space (gym.spaces.Box): The action space of the environment.
            cnn_channels (int): Number of output channels for the CNN layers.
            mlp_hidden_size (int): Number of hidden units in the fully connected layers.
        """
        super(BikkleSemanticImagePolicy, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, params.cnn_channels, kernel_size=3, stride=1, padding=1),  # Input: (4, image_size, image_size)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2
            nn.Conv2d(params.cnn_channels, params.cnn_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2
        )

        # Calculate the flattened size after the CNN layers
        cnn_output_size = (image_size // 4) * (image_size // 4) * (params.cnn_channels * 2)

        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_size, params.mlp_hidden_size),
            nn.ReLU(),
        )

        # Output layers for Gaussian distribution
        self.mean_head = nn.Linear(params.mlp_hidden_size, action_space.shape[0])
        self.log_std_head = nn.Linear(params.mlp_hidden_size, action_space.shape[0])

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute the mean and log standard deviation of the action distribution.

        Args:
            obs (torch.Tensor): Input image tensor of shape (batch_size, 4, image_size, image_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and log standard deviation of the action distribution.
        """
        features = self.cnn(obs)  # Shape: (batch_size, cnn_channels*2, image_size//4, image_size//4)
        features = features.view(features.size(0), -1)  # Flatten
        features = self.mlp(features)  # Shape: (batch_size, mlp_hidden_size)

        mean = self.mean_head(features)  # Shape: (batch_size, action_dim)
        log_std = self.log_std_head(features)  # Shape: (batch_size, action_dim)
        log_std = torch.clamp(log_std, min=-5, max=2)  # Clamp for numerical stability

        return mean, log_std

    def get_action(self, obs: torch.Tensor, action: torch.Tensor = None, greedy: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy.

        Args:
            obs (torch.Tensor): Input image tensor of shape (batch_size, 4, image_size, image_size).
            action (torch.Tensor, optional): Predefined action to evaluate. Defaults to None.
            greedy (bool, optional): If True, returns the mean action. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Sampled action, log probability, and entropy.
        """
        mean, log_std = self(obs)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)

        if greedy:
            action = mean
        if action is None:
            action = normal.rsample()  # Reparameterization trick

        log_prob = normal.log_prob(action).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)

        return action, log_prob, entropy

@dataclass
class BikkleSemanticImageValueParams:
    cnn_channels: int = 32
    mlp_hidden_size: int = 128

class BikkleSemanticImageValue(nn.Module):
    def __init__(self, image_size: int, params: BikkleSemanticImageValueParams):
        super(BikkleSemanticImageValue, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, params.cnn_channels, kernel_size=3, stride=1, padding=1),  # Input: (4, image_size, image_size)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2
            nn.Conv2d(params.cnn_channels, params.cnn_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2
        )

        # Calculate the flattened size after the CNN layers
        cnn_output_size = (image_size // 4) * (image_size // 4) * (params.cnn_channels * 2)

        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_size, params.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(params.mlp_hidden_size, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the value.

        Args:
            obs (torch.Tensor): Input image tensor of shape (batch_size, 4, image_size, image_size).

        Returns:
            torch.Tensor: Scalar value for each observation in the batch.
        """
        features = self.cnn(obs)  # Shape: (batch_size, cnn_channels*2, image_size//4, image_size//4)
        features = features.view(features.size(0), -1)  # Flatten
        value = self.mlp(features)  # Shape: (batch_size, 1)
        return value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Computes the value for the given observations.

        Args:
            obs (torch.Tensor): Input image tensor of shape (batch_size, 4, image_size, image_size).

        Returns:
            torch.Tensor: Scalar value for each observation in the batch.
        """
        return self.forward(obs)

