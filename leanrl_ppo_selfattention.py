# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
from pathlib import Path

from gymnasium.vector import AutoresetMode

from model import BikklePolicy, BikkleValueFunction, BikklePolicyParams, BikkleValueFunctionParams

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Any, Optional

import gymnasium as gym
import numpy as np
import tensordict
import torch
torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import from_module, TensorDict
from tensordict.nn import CudaGraphModule
import env

tensordict.nn.functional_modules._exclude_td_from_pytree().set()


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""

    model_save_interval: int = 100

    dropout: float = 0.1

    policy_params: BikklePolicyParams = BikklePolicyParams()
    value_params: BikkleValueFunctionParams = BikkleValueFunctionParams()

    checkpoint_to_load: Optional[Path] = None


def make_env(env_id, idx, capture_video, run_name, gamma, **kwargs):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **kwargs)
        # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def gae(next_obs, next_done, container):
    # bootstrap value if not done
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done, avg_returns=[]):
    ts = []
    for step in range(args.num_steps):
        # ALGO LOGIC: action logic
        action, logprob, _ = policy(obs=obs)
        value = get_value(obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, next_done, infos = step_func(action)

        if "final_info" in infos:
            info = infos["final_info"]
            avg_returns.extend(info["episode"]["r"])

        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                # cleanrl ppo examples associate the done with the previous obs (not the done resulting from action)
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )

        obs = next_obs = next_obs.to(device, non_blocking=True)
        done = next_done.to(device, non_blocking=True)

    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container


def update(obs, actions, logprobs, advantages, returns, vals):
    optimizer.zero_grad()
    _, newlogprob, entropy = policy_m.get_action(obs, actions)
    newvalue = value_m.get_value(obs)

    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss.backward()
    gn = nn.utils.clip_grad_norm_(list(policy_m.parameters()) + list(value_m.parameters()), args.max_grad_norm)
    optimizer.step()

    return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
)

def save_model_checkpoint(policy_model, value_model, global_step, outdir, policy_params, value_params):
    """
    Saves the model parameters and additional information to a checkpoint file.

    Args:
        policy_model (nn.Module): The policy model to save.
        value_model (nn.Module): The value model to save.
        global_step (int): The current global step.
        outdir (Path): The directory to save the checkpoint.
        policy_params (BikklePolicyParams): The policy parameters.
        value_params (BikkleValueFunctionParams): The value function parameters.
    """
    checkpoint = {
        "version": 1,
        "policy_state_dict": policy_model.state_dict(),
        "value_state_dict": value_model.state_dict(),
        "global_step": global_step,
        "policy_params": policy_params,
        "value_params": value_params,
    }
    torch.save(checkpoint, outdir / f"checkpoint_{global_step}.pth")

def restore_models(observation_space, action_space, args, device):
    """
    Restores the policy and value models from a checkpoint or creates them from args.

    Args:
        observation_space (gym.Space): The observation space of the environment.
        action_space (gym.Space): The action space of the environment.
        args (Args): The arguments containing model parameters and checkpoint path.
        device (torch.device): The device to load the models onto.

    Returns:
        tuple: A tuple containing the policy model and value model.
    """

    # Create the policy model
    policy_model = BikklePolicy(observation_space=observation_space, action_space=action_space, params=args.policy_params).to(device)

    # Create the value model
    value_model = BikkleValueFunction(observation_space=observation_space, params=args.value_params).to(device)

    # Load checkpoint if provided
    if args.checkpoint_to_load:
        checkpoint = torch.load(args.checkpoint_to_load, map_location=device)
        policy_model.load_state_dict(checkpoint["policy_state_dict"])
        value_model.load_state_dict(checkpoint["value_state_dict"])

    return policy_model, value_model

if __name__ == "__main__":
    args = tyro.cli(Args)

    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    time_str = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time_str}"

    wandb_run = wandb.init(
        project="ppo_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    outdir = Path("data") / args.env_id / args.exp_name / f"{time_str}_{wandb_run.id}"
    outdir.mkdir(parents=True, exist_ok=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    # n_act = math.prod(envs.single_action_space.shape)
    # n_obs = math.prod(envs.single_observation_space.shape)
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Register step as a special op not to graph break
    # @torch.library.custom_op("mylib::step", mutates_args=())
    def step_func(action: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        next_obs_np, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        return TensorDict(next_obs_np, device=device, batch_size=(args.num_envs, )), torch.as_tensor(reward), torch.as_tensor(next_done), info

    ####### Agent #######

    # policy_m = BikklePolicy(observation_space=obs_space, action_space=act_space, params=args.policy_params).to(device)
    assert isinstance(obs_space, gym.spaces.Dict)
    assert isinstance(act_space, gym.spaces.Box)

    policy_m, value_m = restore_models(observation_space=obs_space, action_space=act_space, args=args, device=device)

    policy_m.train()
    # Make a version of agent with detached params
    policy_inference_m = BikklePolicy(observation_space=obs_space, action_space=act_space, params=args.policy_params).to(device)
    policy_inference_p = from_module(policy_m).data
    policy_inference_p.to_module(policy_inference_m)
    policy_inference_m.train()

    value_inference_m = BikkleValueFunction(observation_space=obs_space, params=args.value_params).to(device)
    value_inference_p = from_module(value_m).data
    value_inference_p.to_module(value_inference_m)

    ####### Optimizer #######
    optimizer = optim.Adam(
        list(policy_m.parameters()) + list(value_m.parameters()),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = policy_inference_m.get_action
    get_value = value_inference_m.get_value

    # Compile policy
    if args.compile:
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)

    avg_returns = deque(maxlen=20)
    global_step = 0
    container_local = None
    next_obs = TensorDict(envs.reset()[0], device=device, batch_size=(args.num_envs,))
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    # max_ep_ret = -float("inf")
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    # desc = ""
    global_step_burnin = None
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container = rollout(next_obs, next_done, avg_returns=avg_returns)
        global_step += container.numel()

        container = gae(next_obs, next_done, container)
        container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        update_results = []  # Collect results from update calls

        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]

                out = update(container_local, tensordict_out=tensordict.TensorDict())
                update_results.append(out)  # Store the results

                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break

        # Compute the mean of the collected results
        mean_results = {key: torch.stack([res[key] for res in update_results]).mean().item() for key in update_results[0].keys()}

        # Log the mean results to wandb
        wandb.log({f"mean_{key}": value for key, value in mean_results.items()}, step=global_step)

        if global_step_burnin is not None and iteration % 10 == 0:
            speed = (global_step - global_step_burnin) / (time.time() - start_time)
            r = container["rewards"].mean()
            r_max = container["rewards"].max()
            avg_returns_t = torch.tensor(avg_returns).mean()

            with torch.no_grad():
                logs = {
                    "episode_return": np.array(avg_returns).mean(),
                    "logprobs": container["logprobs"].mean(),
                    "advantages": container["advantages"].mean(),
                    "returns": container["returns"].mean(),
                    "vals": container["vals"].mean(),
                    "gn": out["gn"].mean(),
                }

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"speed: {speed: 4.1f} sps, "
                f"reward avg: {r :4.2f}, "
                f"reward max: {r_max:4.2f}, "
                f"returns: {avg_returns_t: 4.2f},"
                f"lr: {lr: 4.2f}"
            )
            wandb.log(
                {"speed": speed, "episode_return": avg_returns_t, "r": r, "r_max": r_max, "lr": lr, **logs}, step=global_step
            )

        if global_step % args.model_save_interval == 0:
            # Save models
            save_model_checkpoint(
                policy_model=policy_m,
                value_model=value_m,
                global_step=global_step,
                outdir=outdir,
                policy_params=args.policy_params,
                value_params=args.value_params,
            )

    envs.close()