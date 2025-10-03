from os import cpu_count
import random
from typing import Any
from tqdm import tqdm
import time
from rich import print
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Literal
from uuid import uuid4

import cloudpickle
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from numpy.random import default_rng
from supersuit.vector import MakeCPUAsyncConstructor
from tensordict import TensorDict
from torch import Tensor

from environment.agents.geniusweb import TRAINING_AGENTS, TESTING_AGENTS
from environment.agents.policy.PPO import GNN, HigaEtAl
from environment.negotiation import NegotiationEnvZoo
from environment.scenario import ScenarioLoader

MAP_DTYPE = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
    "bool": torch.bool,
}


class Policies(Enum):
    GNN = GNN
    HigaEtAl = HigaEtAl


def find_opponents(training: bool, exp: str) -> tuple[tuple[str, ...], dict[str, Any]]:
    if exp in ("scml_dynamic",):
        opponent_types = (
            ("ConcederAgent", "BoulwareAgent") if training else ("LinearAgent",)
        )
    elif exp in (
        "anac",
        "acquisition",
        "anac2024",
        "camera",
        "car",
        "energy",
        "grocery",
        "itex",
        "laptop",
        "scml_dynamic",
        "thompson",
    ):
        opponent_types = ("BoulwareAgent",) if training else ("BoulwareAgent",)
    elif exp in ("anac2024",):
        opponent_types = (
            ("BoulwareAgent", "ConcederAgent") if training else ("BoulwareAgent",)
        )
    else:
        opponent_types = (
            ("BoulwareAgent", "ConcederAgent") if training else ("BoulwareAgent",)
        )

    base_map = TRAINING_AGENTS if training else TESTING_AGENTS
    opponent_map = {
        k: v for k, v in base_map.items() if any(_ in k for _ in opponent_types)
    }
    if not opponent_map:
        print(base_map)
        print(opponent_types)
    print(
        f"Will use opponents {opponent_types} for {'training' if training else 'testing'}"
    )
    print(opponent_map)
    return opponent_types, opponent_map


@dataclass
class Args:
    debug: bool = False
    retrain: bool = True
    deadline: int = 100
    time_limit: int = 10000
    policy: Policies = Policies.GNN
    opponent: Literal["all", "random"] = "all"
    opponent_sets: tuple[Literal["ANL2022", "ANL2023", "CSE3210", "BASIC"], ...] = (
        "BASIC",
    )
    # scenario: str = "environment/scenarios/fixed_utility"
    exp: str = "scml_dynamic"
    random_agent_order: bool = True

    # GNN policy settings
    gat_v2: bool = False
    add_self_loops: bool = True
    hidden_size: int = 256
    heads: int = 4
    out_layers: int = 1
    gnn_layers: int = 4

    # Experiment settings
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    wandb: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RL_negotiation"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    # total_timesteps: int = 2_000_000
    total_timesteps: int = 0  # 2000000
    """total timesteps of the experiments"""
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    num_minibatches: int = 30
    """the number of mini-batches"""
    update_epochs: int = 30
    """the K epochs to update the policy"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    # num_envs: int = 30
    num_envs: int = int(0.8 * cpu_count())
    """the number of parallel game environments"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    norm_adv: bool = False  # NOTE:?
    """Toggles advantages normalization"""
    clip_coef: float = 0.3
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # extension parameters
    issue_size = 0
    """Maximum allowed number of values per issue"""
    n_issues = 0
    """Maximum allowed number of issues"""

    opponent_types: tuple[str, ...] | None = None
    opponent_map: dict[str, Any] | None = None

    def __post_init__(self):
        self.scenario = f"environment/scenarios/random_tmp_{self.exp}"
        if self.total_timesteps < 1:
            if self.exp in ("scml_dynamic", "anac2024"):
                self.total_timesteps = 400_000
            elif self.exp in ("anac",):
                self.total_timesteps = 200_000
            else:
                self.total_timesteps = 100_000

        if self.opponent_types is None:
            self.opponent_types, self.opponent_map = find_opponents(True, self.exp)

        # if self.exp == "scml_dynamic":
        #     self.issue_size = max(self.issue_size, 20)
        #     self.n_issues = max(self.n_issues, 3)


def concat_envs(env_config, num_vec_envs, num_cpus=0):
    def vec_env_args(env, num_envs):
        def env_fn(worker_id):
            env_copy = cloudpickle.loads(cloudpickle.dumps(env))
            env_copy.par_env.worker_id = worker_id
            return env_copy

        return (
            [partial(env_fn, i) for i in range(num_envs)],
            env.observation_space,
            env.action_space,
        )

    env = NegotiationEnvZoo(env_config)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    num_cpus = min(num_cpus, num_vec_envs)
    vec_env = MakeCPUAsyncConstructor(num_cpus)(*vec_env_args(vec_env, num_vec_envs))

    vec_env.single_observation_space = vec_env.observation_space
    vec_env.single_action_space = vec_env.action_space
    vec_env.is_vector_env = True
    return vec_env


def init_tensors(batch_size, envs, device) -> tuple[TensorDict, Tensor]:
    # ALGO Logic: Storage setup
    obs = {}
    for obs_key, obs_space in envs.single_observation_space.items():
        if isinstance(obs_space, MultiDiscrete):
            shape = (sum(obs_space.nvec),)
        elif isinstance(obs_space, Discrete):
            shape = ()  # TODO, fix to one hot?
        elif isinstance(obs_space, Box):
            shape = obs_space.shape
        else:
            raise NotImplementedError
        obs[obs_key] = torch.zeros(
            batch_size + shape, dtype=MAP_DTYPE[str(obs_space.dtype)]
        )
    obs: TensorDict = TensorDict(obs, batch_size=batch_size, device=device)
    actions: torch.Tensor = torch.zeros(batch_size + envs.single_action_space.shape).to(
        device
    )

    return obs, actions


def main():
    args = tyro.cli(Args)
    run_name_base = f"{args.policy.name}_{args.exp}"
    run_name = (
        f"{run_name_base}.{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}_{uuid4()}"
    )
    models = [_.name for _ in Path("models").glob(f"{run_name_base}.*")]
    if not args.retrain and any(_.startswith(run_name_base) for _ in models):
        print(f"Found existing model at {run_name}, will not retrain")
        return
    print(f"Will use {args.num_envs} environments")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.debug:
        args.num_envs = 2
        args.batch_size = 20
        args.minibatch_size = 20
        args.num_iterations = 2
        args.update_epochs = 1

    if args.wandb:
        import wandb

        logger = wandb.init(
            entity="brenting",
            project=args.wandb_project_name,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    scenario_rng = default_rng(args.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    # env setup
    used_agents = [
        a for a in args.opponent_map if a.startswith(tuple(args.opponent_sets))
    ]
    env_config = {
        "agents": [f"RL_{args.policy.name}", args.opponent],
        "used_agents": used_agents,
        "exp": args.exp,
        "scenario": args.scenario,
        "deadline": {"rounds": args.deadline, "ms": args.time_limit},
        "random_agent_order": args.random_agent_order,
        "issue_size": args.issue_size,
        "n_issues": args.n_issues,
        "testing": False,
    }

    loader = ScenarioLoader(
        Path(f"environment/scenarios/training/{args.exp}"), random=True
    )
    if args.scenario.startswith("environment/scenarios/random_tmp"):
        scenario = loader.random_scenario()
        scenario.to_directory(Path(args.scenario))
    envs = concat_envs(env_config, args.num_envs, num_cpus=args.num_envs)

    agent: GNN = args.policy.value(envs, args).to(device)  # type: ignore
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_size = (args.num_steps, args.num_envs)
    logprobs = torch.zeros(batch_size).to(device)
    rewards = torch.zeros(batch_size).to(device)
    dones = torch.zeros(batch_size).to(device)
    values = torch.zeros(batch_size).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    _strt = time.perf_counter()
    for iteration in tqdm(range(1, args.num_iterations + 1)):
        if (
            args.scenario.startswith("environment/scenarios/random_tmp")
            or iteration == 1
        ):
            if args.scenario.startswith("environment/scenarios/random_tmp"):
                scenario = loader.next_scenario()
                # scenario = Scenario.create_random([200, 1000], scenario_rng, 5, True)
                scenario.to_directory(Path(args.scenario))

            envs = concat_envs(env_config, args.num_envs, num_cpus=args.num_envs)
            agent.action_nvec = tuple(envs.single_action_space.nvec)
            obs, actions = init_tensors(batch_size, envs, device)

            next_obs, _ = envs.reset(seed=args.seed)
            next_obs = TensorDict(next_obs, batch_size=(args.num_envs,), device=device)
            next_done = torch.zeros(args.num_envs).to(device)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        utility_all_agents = defaultdict(lambda: 0.0)
        count_all_agents = defaultdict(lambda: 0)
        log_metrics = defaultdict(lambda: [0.0, 0])
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done_bool = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                TensorDict(next_obs, batch_size=(args.num_envs,), device=device),
                torch.Tensor(next_done_bool).to(device),
            )

            if next_done_bool.any():
                for info in infos:
                    if info:
                        for agent_id, utility in info["utility_all_agents"].items():
                            utility_all_agents[agent_id] += utility
                            count_all_agents[agent_id] += 1
                        log_metrics["rounds_played"][0] += info["rounds_played"]
                        log_metrics["rounds_played"][1] += 1
                        log_metrics["self_accepted"][0] += info["self_accepted"]
                        log_metrics["self_accepted"][1] += 1
                        log_metrics["found_agreement"][0] += info["found_agreement"]
                        log_metrics["found_agreement"][1] += 1
                log_metrics["episode_reward_mean"][0] += reward[next_done_bool].sum()
                log_metrics["episode_reward_mean"][1] += next_done_bool.sum()

        if args.wandb:
            for metric, (value, count) in log_metrics.items():
                logger.log({metric: value / count}, step=global_step)
                print(f"{metric}: {value / count}")
            for agent_id, utility in utility_all_agents.items():
                logger.log(
                    {f"utility/{agent_id}": utility / count_all_agents[agent_id]},
                    step=global_step,
                )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # NOTE: identical

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (
                        newvalue - b_returns[mb_inds]
                    ) ** 2  # NOTE: log clips, clip vloss directly
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if args.wandb:
            logger.log({"learning_rate": optimizer.param_groups[0]["lr"]}, global_step)
            logger.log(
                {"losses/unclipped_value": newvalue.detach().mean().cpu().numpy()},
                global_step,
            )
            logger.log(
                {"losses/gradient_norm": total_norm.detach().cpu().numpy()}, global_step
            )
            logger.log({"losses/total_loss": loss.item()}, global_step)
            logger.log({"losses/value_loss": v_loss.item()}, global_step)
            logger.log({"losses/policy_loss": pg_loss.item()}, global_step)
            logger.log({"losses/entropy": entropy_loss.item()}, global_step)
            logger.log({"losses/old_approx_kl": old_approx_kl.item()}, global_step)
            logger.log({"losses/approx_kl": approx_kl.item()}, global_step)
            logger.log({"losses/clipfrac": np.mean(clipfracs)}, global_step)
            logger.log({"losses/explained_variance": explained_var}, global_step)
            logger.log(
                {"SPS": int(global_step / (time.time() - start_time))}, global_step
            )
        # print("SPS:", int(global_step / (time.time() - start_time)))
        model_path = f"models/{run_name}"
        # print(f"Will save the model to {model_path}")
        torch.save(agent.state_dict(), model_path)

    print(f"Model saved to to {model_path}")
    envs.close()
    total_time = time.perf_counter() - _strt

    if args.wandb:
        artifact = wandb.Artifact("model", type="model")  # type: ignore
        artifact.add_file(model_path)  # type: ignore
        logger.log_artifact(artifact)  # type: ignore
        logger.finish()  # type: ignore
    print(f"Total time: {total_time:.3f} seconds")


if __name__ == "__main__":
    main()
