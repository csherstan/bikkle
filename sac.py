"""
Original source: https://github.com/pytorch-labs/LeanRL/blob/main/leanrl/sac_continuous_action_torchcompile.py
"""

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["EXCLUDE_TD_FROM_PYTREE"] = "1"
from pathlib import Path

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
from model import BikklePolicy, preprocess_bikkle_observation_with_mask, BikkleValueFunction, SimplePolicy, \
    SimpleValueFunction
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
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005

    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-5
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-4
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

    max_blocks: int = 2

    model_save_interval: int = 1000
    num_envs: int = 1

    # --------------replay buffers
    """target smoothing coefficient (default: 0.005)"""
    buffer_size_default: int = int(1e5)
    """the replay memory buffer size for the default buffer"""
    buffer_size_reward: int = int(1e4)
    """the replay memory buffer size for the reward buffer"""
    buffer_size_user: int = int(0)
    """the replay memory buffer size for the user buffer"""
    samples_from_default: int = 256
    """number of samples to draw from the default buffer"""
    samples_from_reward: int = 64
    """number of samples to draw from the reward buffer"""
    samples_from_user: int = 0
    """number of samples to draw from the user buffer"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, num_blocks=2)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=150)
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
            if k == "actions":
                mean_action_vector = np.array(v).mean(axis=0)
                ret_data[k] = np.linalg.norm(mean_action_vector)
                ret_data[k] = np.array(v).mean()
            elif k in ["cyan_touched", "pink_touched"]:
                ret_data[k] = np.sum(v)

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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, args.capture_video, run_name) for i in range(args.num_envs)])
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # actor = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)
    # actor_detach = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)

    # actor = BikklePolicy(observation_space=obs_space,
    #                      action_space=action_space).to(device)
    # actor_detach = BikklePolicy(observation_space=obs_space,
    #                             action_space=action_space).to(device)

    num_pink = num_cyan = args.max_blocks // 2
    actor = SimplePolicy(observation_space=obs_space, action_space=action_space, num_cyan=num_cyan,
                         num_pink=num_pink).to(device)
    actor_detach = SimplePolicy(observation_space=obs_space, action_space=action_space, num_cyan=num_cyan,
                                num_pink=num_pink).to(device)

    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = TensorDictModule(actor_detach.get_action, in_keys=["tokens", "indices", "mask"], out_keys=["action"])


    def get_q_params():
        # qf1 = BikkleValueFunction(observation_space=obs_space, action_space=action_space).to(device)
        # qf2 = BikkleValueFunction(observation_space=obs_space, action_space=action_space).to(device)
        qf1 = SimpleValueFunction(observation_space=obs_space, action_space=action_space, num_cyan=num_cyan,
                                  num_pink=num_pink).to(device)
        qf2 = SimpleValueFunction(observation_space=obs_space, action_space=action_space, num_cyan=num_cyan,
                                  num_pink=num_pink).to(device)
        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target = qnet_params.data.clone()

        # discard params of net
        # qnet = BikkleValueFunction(observation_space=obs_space, action_space=action_space).to("meta")
        qnet = SimpleValueFunction(observation_space=obs_space, action_space=action_space, num_cyan=num_cyan,
                                   num_pink=num_pink).to("meta")
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
    rb_default = ReplayBuffer(storage=ListStorage(args.buffer_size_default), prefetch=1,
                              batch_size=args.samples_from_default)
    rb_reward = ReplayBuffer(storage=ListStorage(args.buffer_size_reward), prefetch=1,
                             batch_size=args.samples_from_reward)
    rb_user = ReplayBuffer(storage=ListStorage(args.buffer_size_user), prefetch=1, batch_size=args.samples_from_user)


    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            # assert not torch.isnan(vals).any(), "Nan detected in value prediction"
            # assert not torch.isinf(vals).any(), "Inf detected in value prediction"
            if next_q_value is not None:
                loss_val = F.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals


    def update_value_function(obs, next_obs, actions, rewards, dones):
        # optimize the model
        q_optimizer.zero_grad()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(**next_obs)
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(
                qnet_target, next_obs, next_state_actions
            )
            # assert not torch.isnan(qf_next_target).any(), f"NaN in qf_next_target"
            # assert not torch.isinf(qf_next_target).any(), f"inf in qf_next_target"
            min_qf_next_target = qf_next_target.min(dim=0).values - alpha * next_state_log_pi
            next_q_value = rewards.flatten() + (
                1.0 - dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

        qf_a_values = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params, obs, actions, next_q_value
        )

        # assert not torch.isnan(qf_a_values).any(), f"NaN in qf_a_values"
        # assert not torch.isinf(qf_a_values).any(), f"inf in qf_a_values"

        qf_loss = qf_a_values.sum(0)

        # assert not torch.isnan(qf_loss).any(), f"NaN in qf_loss"
        # assert not torch.isinf(qf_loss).any(), f"Inf in qf_loss"

        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())


    def update_policy(obs):
        actor_optimizer.zero_grad()
        pi, log_pi, _ = actor.get_action(**obs)
        qf_pi = torch.vmap(batched_qf, (0, None, None))(qnet_params.data, obs, pi)
        stddev = log_pi.exp().mean()
        min_qf_pi = qf_pi.min(0).values
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        # assert not torch.isnan(actor_loss).any(), f"NaN in actor_loss"
        # assert not torch.isinf(actor_loss).any(), f"Inf in actor_loss"

        actor_loss.backward()
        actor_optimizer.step()

        metrics = {
            "actor_loss": actor_loss.detach(),
            "alpha": alpha.detach(),
            "stddev": stddev.detach(),
        }

        if args.autotune:
            a_optimizer.zero_grad()
            with torch.no_grad():
                _, log_pi, _ = actor.get_action(**obs)
                stddev = log_pi.exp().mean()  # Calculate the mean stddev
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
            metrics["alpha_loss"] = alpha_loss.detach()

            alpha_loss.backward()
            a_optimizer.step()
        return TensorDict(**metrics)


    def extend_replay_buffers(transition: dict) -> None:

        for k, v in transition.items():
            if k == "default":
                rb_default.extend(v)
            elif k == "reward":
                rb_reward.extend(v)
            elif k == "user":
                rb_user.extend(v)


    def sample_from_replay_buffers(device) -> TensorDict:
        samples = []
        total_sampled = 0
        if len(rb_reward) > 0:
            samples_reward = rb_reward.sample(args.samples_from_reward)
            samples.append(samples_reward)
            total_sampled += len(samples_reward)

        if len(rb_user) > 0:
            samples_user = rb_user.sample(args.samples_from_user)
            samples.append(samples_user)
            total_sampled += len(samples_user)

        expected_batch_size = args.samples_from_default + args.samples_from_user + args.samples_from_reward

        samples.append(rb_default.sample(expected_batch_size - total_sampled))

        # Combine samples from all buffers
        combined_samples = TensorDict.cat(samples, dim=0)

        data = tree_map(lambda x: x.to(device), combined_samples)
        return data


    def prep_obs_for_replay_buffer(obs: dict, num_envs: int, device) -> list:
        """
        Unpacks observations from a SyncVectorEnv and prepares them for the replay buffer.

        Args:
            obs: A dictionary of observations from the environment.
            device: The device to which the observations should be moved.

        Returns:
            A list of dictionaries, each corresponding to a single environment's observation.
        """
        obs_list = []

        for i in range(num_envs):
            single_obs = {}
            for key, value in obs.items():
                assert num_envs == len(value), "Inconsistent number of environments in observations"
                single_obs[key] = torch.as_tensor(value[i], dtype=torch.float32, device=device)
            assert not any(torch.isinf(x).any() for x in single_obs.values()), "Inf detected in observations"
            assert not any(torch.isnan(x).any() for x in single_obs.values()), "NaN detected in observations"
            obs_list.append(single_obs)

        return obs_list


    def prep_transition_list(num_envs: int, obs, next_obs, actions, rewards, terminations, device):
        """
        Unpacks observations from a SyncVectorEnv and prepares them for the replay buffer.
        :param num_envs:
        :param obs:
        :param next_obs:
        :param actions:
        :param rewards:
        :param terminations:
        :param device:
        :return:
        """
        unpacked_obs = prep_obs_for_replay_buffer(obs, num_envs, device)
        unpacked_next_obs = prep_obs_for_replay_buffer(next_obs, num_envs, device)

        transitions = []
        for _obs, _next_obs, _actions, _rewards, _terminations in zip(unpacked_obs, unpacked_next_obs,
                                                                      actions, rewards, terminations):
            transitions.append(TensorDict(
                observations=_obs,
                next_observations=_next_obs,
                actions=torch.as_tensor(_actions, device=rb_device, dtype=torch.float32),
                rewards=torch.as_tensor(_rewards, device=rb_device, dtype=torch.float32),
                terminations=torch.as_tensor(_terminations, device=rb_device, dtype=torch.float32),
                dones=torch.as_tensor(_terminations, device=rb_device, dtype=torch.float32),
                batch_size=None,
            ))

        return transitions


    is_extend_compiled = False
    if args.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        update_value_function = torch.compile(update_value_function, mode=mode)
        update_policy = torch.compile(update_policy, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if args.cudagraphs:
        update_value_function = CudaGraphModule(update_value_function, in_keys=[], out_keys=[])
        update_policy = CudaGraphModule(update_policy, in_keys=[], out_keys=[])
        # policy = CudaGraphModule(policy)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""
    avg_reward = 0.0
    reward_history = [deque(maxlen=20) for _ in range(args.num_envs)]

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
        metrics.add({"actions": actions,
                     "rewards": rewards,
                     "actions_norm": np.linalg.norm(actions),
                     })
        if "pink_touched" in infos:
            metrics.add({"pink_touched": infos["pink_touched"]})

        if "cyan_touched" in infos:
            metrics.add({"cyan_touched": infos["cyan_touched"]})

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

        # TODO: address this -> doesn't actually matter since our env never returns final_observation
        # real_next_obs = next_obs
        # real_next_obs = next_obs.clone()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = torch.as_tensor(infos["final_observation"][idx], device=device, dtype=torch.float)
        # obs = torch.as_tensor(obs, device=device, dtype=torch.float)

        transitions = prep_transition_list(num_envs=args.num_envs, obs=obs, next_obs=next_obs, actions=actions,
                                           rewards=rewards, terminations=terminations, device=rb_device)

        for env_idx in range(args.num_envs):
            reward_history[env_idx].append(transitions[env_idx])

            if rewards[env_idx] != 0:
                extend_replay_buffers({"reward": list(reward_history[env_idx])})

            if terminations[env_idx] or truncations[env_idx]:
                reward_history[env_idx].clear()

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        extend_replay_buffers({"default": transitions})
        data = sample_from_replay_buffers(device)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:

            pp_obs = preprocess_bikkle_observation_with_mask(data["observations"],
                                                          observation_space=obs_space,
                                                          max_blocks=args.max_blocks,
                                                          device=device)
            pp_next_obs = preprocess_bikkle_observation_with_mask(data["next_observations"],
                                                               observation_space=obs_space,
                                                               max_blocks=args.max_blocks,
                                                               device=device)
            out_main = update_value_function(obs=pp_obs, next_obs=pp_next_obs, actions=data["actions"],
                                             rewards=data["rewards"], dones=data["dones"])

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    out_main.update(update_policy(pp_obs))

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
