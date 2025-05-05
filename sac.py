"""
Original source: https://github.com/pytorch-labs/LeanRL/blob/main/leanrl/sac_continuous_action_torchcompile.py
"""

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
from pathlib import Path

import gymnasium.wrappers

from torch.utils._pytree import tree_map
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule

from torchrl.data import ReplayBuffer, ListStorage
from model import BikklePolicy, preprocess_bikkle_observation_with_mask, BikkleValueFunction
from env import *

torch.set_float32_matmul_precision('high')


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
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""

    max_blocks: int = 100

    model_save_interval: int = 1000


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=300)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class Metrics:

    def __init__(self):
        self._metrics = {}

    def add(self, data: dict) -> None:
        for k, v in data.items():
            if k not in self._metrics:
                self._metrics[k] = []
            self._metrics[k].append(v)

    def reset(self):
        self._metrics = {}

    def get_data(self):
        ret_data = {}
        for k, v in self._metrics.items():
            ret_data[k] = np.array(v).mean()

        return ret_data



if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}__{int(time.time())}"

    wandb_run = wandb.init(
        project="bikkle",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    outdir = Path("data") / Path(wandb_run.name)
    outdir.mkdir(parents=True, exist_ok=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # actor = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)
    # actor_detach = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)

    actor = BikklePolicy(observation_space=obs_space,
                         action_space=action_space).to(device)
    actor_detach = BikklePolicy(observation_space=obs_space,
                                action_space=action_space).to(device)

    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = TensorDictModule(actor_detach.get_action, in_keys=["tokens", "indices", "mask"], out_keys=["action"])


    def get_q_params():
        qf1 = BikkleValueFunction(observation_space=obs_space, action_space=action_space).to(device)
        qf2 = BikkleValueFunction(observation_space=obs_space, action_space=action_space).to(device)
        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target = qnet_params.data.clone()

        # discard params of net
        qnet = BikkleValueFunction(observation_space=obs_space, action_space=action_space).to("meta")
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target, qnet


    qnet_params, qnet_target, qnet = get_q_params()

    q_optimizer = optim.Adam(qnet.parameters(), lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr,
                                 capturable=args.cudagraphs and not args.compile)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    envs.single_observation_space.dtype = np.float32
    rb_device = "cpu"
    rb = ReplayBuffer(storage=ListStorage(args.buffer_size), prefetch=1, batch_size=args.batch_size)


    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                loss_val = F.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals


    def update_main(data):
        data = tree_map(lambda x: x.to(device), data)
        # optimize the model
        q_optimizer.zero_grad()
        with torch.no_grad():
            preprop_next = preprocess_bikkle_observation_with_mask(data["next_observations"],
                                                                   observation_space=obs_space,
                                                                   max_blocks=args.max_blocks, device=device)
            next_state_actions, next_state_log_pi, _ = actor.get_action(**preprop_next)
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(
                qnet_target, preprop_next, next_state_actions
            )
            min_qf_next_target = qf_next_target.min(dim=0).values - alpha * next_state_log_pi
            next_q_value = data["rewards"].flatten() + (
                    1.0 - data["dones"].flatten()) * args.gamma * min_qf_next_target.view(-1)

        qf_a_values = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params, preprocess_bikkle_observation_with_mask(data["observations"], observation_space=obs_space,
                                                                 max_blocks=args.max_blocks, device=device),
            data["actions"], next_q_value
        )
        qf_loss = qf_a_values.sum(0)

        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())


    def update_pol(data):
        data = tree_map(lambda x: x.to(device), data)
        actor_optimizer.zero_grad()
        preprop = preprocess_bikkle_observation_with_mask(observation=data["observations"],
                                                          observation_space=obs_space,
                                                          max_blocks=args.max_blocks,
                                                          device=device)
        pi, log_pi, _ = actor.get_action(**preprop)
        qf_pi = torch.vmap(batched_qf, (0, None, None))(qnet_params.data, preprop, pi)
        stddev = log_pi.exp().mean()
        min_qf_pi = qf_pi.min(0).values
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        metrics = {
            "actor_loss": actor_loss.detach(),
            "alpha": alpha.detach(),
            "stddev": stddev.detach(),
        }

        actor_loss.backward()
        actor_optimizer.step()

        if args.autotune:
            a_optimizer.zero_grad()
            with torch.no_grad():
                _, log_pi, _ = actor.get_action(**preprop)
                stddev = log_pi.exp().mean()  # Calculate the mean stddev
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
            metrics["alpha_loss"] = alpha_loss.detach()

            alpha_loss.backward()
            a_optimizer.step()
        return TensorDict(**metrics)


    def extend_and_sample(transition):
        rb.extend(transition)
        return rb.sample(args.batch_size)


    def prep_obs_for_replay_buffer(obs: dict, device):
        return tree_map(lambda x: torch.as_tensor(x, dtype=torch.float32, device=device), obs)


    is_extend_compiled = False
    if args.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[])
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[])
        # policy = CudaGraphModule(policy)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    # obs = torch.as_tensor(obs, device=device, dtype=torch.float)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""
    avg_reward = 0.0

    metrics = Metrics()

    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            processed_obs = preprocess_bikkle_observation_with_mask(obs,
                                                                    observation_space=obs_space,
                                                                    max_blocks=args.max_blocks,
                                                                    device=device)
            actions = policy(**processed_obs)
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # avg_reward = infos["average_reward"]
        # avg_action_norm = infos["average_action_norm"]
        metrics.add({"actions": actions, "rewards": rewards, "actions_norm": np.linalg.norm(actions)})

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                r = float(info["episode"]["r"])
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)
            desc = (
                f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"
            )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float)

        real_next_obs = next_obs
        # TODO: address this -> doesn't actually matter since our env never returns final_observation
        # real_next_obs = next_obs.clone()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = torch.as_tensor(infos["final_observation"][idx], device=device, dtype=torch.float)
        # obs = torch.as_tensor(obs, device=device, dtype=torch.float)
        transition = TensorDict(
            observations=prep_obs_for_replay_buffer(obs, device=rb_device),
            next_observations=prep_obs_for_replay_buffer(real_next_obs, device=rb_device),
            actions=torch.as_tensor(actions, device=rb_device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=rb_device, dtype=torch.float),
            terminations=torch.as_tensor(terminations, device=rb_device, dtype=torch.float),
            dones=torch.as_tensor(terminations, device=rb_device, dtype=torch.float),
            batch_size=1,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        data = extend_and_sample(transition)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            out_main = update_main(data)
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    out_main.update(update_pol(data))

                    if args.autotune:
                        alpha.copy_(log_alpha.detach().exp())

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target.lerp_(qnet_params.data, args.tau)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        # "episode_return": torch.tensor(avg_returns).mean(),
                        "actor_loss": out_main["actor_loss"].mean(),
                        "alpha_loss": out_main.get("alpha_loss", 0),
                        "qf_loss": out_main["qf_loss"].mean(),
                        "alpha": out_main["alpha"].mean(),
                        "stddev": out_main["stddev"].mean(),
                    }

                    logs.update(metrics.get_data())
                    metrics.reset()
                wandb.log(
                    {
                        "speed": speed,
                        **logs,
                    },
                    step=global_step,
                )

            if global_step % args.model_save_interval == 0:
                # Save models
                torch.save(actor.state_dict(), outdir / f"actor_model_{global_step}.pth")
                torch.save(qnet_params.state_dict(), outdir / f"qnet_model_{global_step}.pth")

    envs.close()
