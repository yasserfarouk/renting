# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from collections import defaultdict
import random
import time
from dataclasses import dataclass
from typing import Literal

import cloudpickle
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from supersuit.vector import MakeCPUAsyncConstructor
from tensordict import TensorDict

from environment.agents.geniusweb import AGENTS
from environment.agents.policy.PPO import AttentionGraphToGraph3, PureGNN2, FixedToFixed3
from environment.negotiation import NegotiationEnvZoo
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

AGENT_MODULES = {
    "AttentionGraphToGraph3": AttentionGraphToGraph3,
    "FixedToFixed3": FixedToFixed3,
    "PureGNN2": PureGNN2
}
MAP_DTYPE = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
    "bool": torch.bool,
}

@dataclass
class Args:
    deadline: int
    module: str
    opponent: str
    debug: bool = False
    use_opponent_encoding: bool = False
    opponent_sets: tuple[Literal["ANL2022","ANL2023","CSE3210","BASIC"], ...] = ("ANL2022","ANL2023","CSE3210")
    scenario: str = "environment/scenarios/fixed_utility"
    random_agent_order: bool = False
    DPO: bool = False
    hidden_size: int = 32

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
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 30
    """the number of parallel game environments"""
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1
    """the discount factor gamma"""
    gae_lambda: float = 1
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 30
    """the number of mini-batches"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    norm_adv: bool = False #NOTE:?
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



def concat_envs(env_config, num_vec_envs, num_cpus=0):
    def vec_env_args(env, num_envs):
        def env_fn():
            env_copy = cloudpickle.loads(cloudpickle.dumps(env))
            return env_copy

        return [env_fn] * num_envs, env.observation_space, env.action_space
    env = NegotiationEnvZoo(env_config)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    num_cpus = min(num_cpus, num_vec_envs)
    vec_env = MakeCPUAsyncConstructor(num_cpus)(*vec_env_args(vec_env, num_vec_envs))

    vec_env.single_observation_space = vec_env.observation_space
    vec_env.single_action_space = vec_env.action_space
    vec_env.is_vector_env = True
    return vec_env

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.debug:
        args.num_envs = 2
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.module}_{int(time.time())}"
    if args.wandb:
        import wandb

        wandb.init(
            entity="brenting",
            project=args.wandb_project_name,
            config=vars(args),
            name=args.module,
            save_code=True,
        )
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    used_agents = [a for a in AGENTS if a.startswith(tuple(args.opponent_sets))]
    env_config = {
        "agents": [f"RL_{args.module}", args.opponent],
        "used_agents": used_agents,
        "scenario": args.scenario,
        "deadline": {"rounds": args.deadline, "ms": 10000},
        "random_agent_order": args.random_agent_order,
    }
    # envs = gym.vector.AsyncVectorEnv(
    #     [lambda: NegotiationEnvZoo(env_config, i) for i in range(args.num_envs)],
    # )

    envs = concat_envs(env_config, args.num_envs, num_cpus=args.num_envs)

    agent = AGENT_MODULES[args.module](envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    batch_size = (args.num_steps, args.num_envs)
    obs = {}
    for obs_key, obs_space in envs.single_observation_space.items():
        if isinstance(obs_space, MultiDiscrete):
            shape = (sum(obs_space.nvec),)
        elif isinstance(obs_space, Discrete):
            shape = () #TODO, fix to one hot?
        elif isinstance(obs_space, Box):
            shape = obs_space.shape
        else:
            raise NotImplementedError
        obs[obs_key] = torch.zeros(batch_size + shape, dtype=MAP_DTYPE[str(obs_space.dtype)])
    obs: TensorDict = TensorDict(obs, batch_size=batch_size, device=device)
    # obs2: TensorDict = TensorDict(
    #     {
    #         "head_node": torch.zeros(batch_size + envs.single_observation_space["head_node"].shape),
    #         "objective_nodes": torch.zeros(batch_size + envs.single_observation_space["objective_nodes"].shape),
    #         "value_nodes": torch.zeros(batch_size + envs.single_observation_space["value_nodes"].shape),
    #         # "edge_indices_val_obj": torch.zeros(batch_size + envs.single_observation_space["edge_indices_val_obj"].shape, dtype=torch.int64),
    #         # "edge_indices_obj_head": torch.zeros(batch_size + envs.single_observation_space["edge_indices_obj_head"].shape, dtype=torch.int64),
    #         "edge_indices": torch.zeros(batch_size + envs.single_observation_space["edge_indices"].shape, dtype=torch.int64),
    #         "opponent_encoding": torch.zeros(batch_size + envs.single_observation_space["opponent_encoding"].shape, dtype=torch.int64),
    #         "accept_mask": torch.zeros(batch_size + envs.single_observation_space["accept_mask"].shape, dtype=torch.bool),
    #     },
    #     batch_size=batch_size,
    #     device=device,
    # )
    actions: torch.Tensor = torch.zeros(batch_size + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros(batch_size).to(device)
    rewards = torch.zeros(batch_size).to(device)
    dones = torch.zeros(batch_size).to(device)
    values = torch.zeros(batch_size).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    # next_obs = torch.Tensor(next_obs).to(device)
    next_obs = TensorDict(next_obs, batch_size=(args.num_envs,), device=device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # print("Collecting rollouts..")
        # episodic_return = .0
        # episodes_done = 0
        utility_all_agents = defaultdict(lambda: .0)
        count_all_agents = defaultdict(lambda: 0)
        log_metrics = defaultdict(lambda: [.0, 0])
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
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_bool = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = TensorDict(next_obs, batch_size=(args.num_envs,), device=device), torch.Tensor(next_done_bool).to(device)

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
                        # wandb.log({"rounds_played": info["rounds_played"]}, step=global_step)
                        # wandb.log({"self_accepted": info["self_accepted"]}, step=global_step)
                        # wandb.log({"found_agreement": info["found_agreement"]}, step=global_step)
                log_metrics["episode_reward_mean"][0] += reward[next_done_bool].sum()
                log_metrics["episode_reward_mean"][1] += next_done_bool.sum()
        
        if args.wandb:
            for metric, (value, count) in log_metrics.items():
                wandb.log({metric: value / count}, step=global_step)
                print(f"{metric}: {value / count}")
            # wandb.log({"episode_reward_mean": episodic_return / episodes_done}, step=global_step)
            for agent_id, utility in utility_all_agents.items():
                wandb.log({f"utility/{agent_id}": utility / count_all_agents[agent_id]}, step=global_step)

        # print("Calculating returns and advantages..")
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
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # print("Optimizing policy...")
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # print(f"Epoch {epoch}")
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                if args.DPO: # see DPO paper
                    is_pos = (mb_advantages >= 0.0).float()
                    r1 = ratio - 1.0
                    drift1 = F.relu(r1 * mb_advantages - 2 * F.tanh(r1 * mb_advantages / 2))
                    drift2 = F.relu(
                        logratio * mb_advantages - 0.6 * F.tanh(logratio * mb_advantages / 0.6)
                    )
                    drift = drift1 * is_pos + drift2 * (1 - is_pos)
                    pg_loss = -(ratio * mb_advantages - drift).mean()
                else:
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean() #NOTE: identical

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2 #NOTE: log clips, clip vloss directly
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    # v_loss = torch.pow(newvalue - b_returns[mb_inds], 2.0)
                    # v_loss = torch.clamp(v_loss, 0, 10.0).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
            

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if args.wandb:
            wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]}, global_step)
            wandb.log({"losses/total_loss": loss.item()}, global_step)
            wandb.log({"losses/value_loss": v_loss.item()}, global_step)
            wandb.log({"losses/policy_loss": pg_loss.item()}, global_step)
            wandb.log({"losses/entropy": entropy_loss.item()}, global_step)
            wandb.log({"losses/old_approx_kl": old_approx_kl.item()}, global_step)
            wandb.log({"losses/approx_kl": approx_kl.item()}, global_step)
            wandb.log({"losses/clipfrac": np.mean(clipfracs)}, global_step)
            wandb.log({"losses/explained_variance": explained_var}, global_step)
            wandb.log({"SPS": int(global_step / (time.time() - start_time))}, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

    envs.close()
