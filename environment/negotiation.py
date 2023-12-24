from collections import deque
from itertools import cycle
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Sequence
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from environment.agents.geniusweb import AGENTS
from environment.scenario import Scenario
from environment.deadline import Deadline
from environment.agents.rl_agent import RLAgent
from geniusweb.party.DefaultParty import DefaultParty


class NegotiationEnv(MultiAgentEnv):
    def __init__(self, env_config: dict):
        self.env_config = env_config
        self.agent_configs = env_config["agent_configs"]
        self._agent_ids = set([f"agent_{n}" for n in range(len(self.agent_configs))])
        self.used_agents = {a: AGENTS[a] for a in env_config["used_agents"]}
        super().__init__()

        # if env_config["wandb_enabled"]:
        #     # self.wandb = setup_wandb(env_config["wandb_config"])
        #     wandb.init(
        #         project=env_config["wandb_config"]["project"],
        #         group=env_config["wandb_config"]["group"],
        #     )

    def reset(self, *, seed=None, options=None):
        # Call super's `reset()` method to (maybe) set the given `seed`.
        super().reset(seed=seed, options=options)

        if self.env_config["scenario"] == "random":
            self.scenario = Scenario.create_random([100, 10000], self.np_random)
        else:
            self.scenario = Scenario.from_directory(Path(self.env_config["scenario"]))


        values_per_objective = [len(v) for v in self.scenario.objectives.values()]
        outcome_space = MultiDiscrete(values_per_objective, dtype=np.int32)
        # self.action_offset = np.cumsum(np.insert(outcome_space, 0, 0)[:-1])

        self.action_space = Dict(
            {
                "outcome": outcome_space,
                "accept": Discrete(2),
            }
        )
        # self.observation_space = Dict(
        #     {
        #         "my_outcome": outcome_space,
        #         "opp_outcome": outcome_space,
        #         "opponent_encoding": Discrete(len(self.used_agents)),
        #         "time": Box(-0, 1, dtype=np.float32),
        #     }
        # )
        # self.observation_space = Dict(
        #     {
        #         "head_node": Box(
        #             np.array([0, 0]), np.array([np.inf, 1]), dtype=np.float32
        #         ),  # #objectives, time (implicit num offers)
        #         "objective_nodes": Repeated(
        #             Box(np.array([0, 0]), np.array([np.inf, 1]), dtype=np.float32),
        #             max_len=10
        #         ),  # #values, weight,
        #         "value_nodes": Repeated(
        #             Box(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), dtype=np.float32),
        #             max_len=1000,
        #         ),  # weight, average offered, my outcome, opp outcome
        #     }
        # )
        self.observation_space = Dict(
            {
                "head_node": Box(
                    np.array([0, 0]), np.array([np.inf, 1]), dtype=np.float32
                ),  # #objectives, time (implicit num offers)
                "objective_nodes": Sequence(
                    Box(np.array([0, 0]), np.array([np.inf, 1]), dtype=np.float32),
                    stack=True,
                ),  # #values, weight,
                "value_nodes": Sequence(
                    Box(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), dtype=np.float32),
                    stack=True,
                ),  # weight, average offered, my outcome, opp outcome
            }
        )


        self.last_actions = deque(maxlen=len(self._agent_ids))

        # TODO: list based utility functions in scenario
        utility_functions = [self.scenario.utility_function_A, self.scenario.utility_function_B]


        if "rounds" in self.env_config["deadline"]:
            self.deadline = Deadline(rounds=self.env_config["deadline"]["rounds"])
        elif "ms" in self.env_config["deadline"]:
            self.deadline = Deadline(ms=self.env_config["deadline"]["ms"])
        else:
            raise ValueError("Deadline parameter not recognized")
        
        # self.opponent_encoding = np.array([0], dtype=np.int32)

        self.agents = []
        for agent_id, agent_config, utility_function in zip(sorted(self._agent_ids), self.agent_configs, utility_functions):
            if agent_config == "RL":
                agent_class = RLAgent
            elif agent_config == "random":
                agent_type, agent_class = self.np_random.choice(list(self.used_agents.items()))
                # self.opponent_encoding = self.env_config["used_agents"].index(agent_type)
            elif agent_config == "all":
                agent_type, agent_class = list(self.used_agents.items())[self.env_config.worker_index - 1]
                # self.opponent_encoding = self.env_config["used_agents"].index(agent_type)
            elif agent_config in self.used_agents:
                agent_class = self.used_agents[agent_config]
            else:
                raise ValueError("Agent config not recognized")
            
            agent = agent_class(agent_id, utility_function, self.deadline)
            self.agents.append(agent)

        if self.env_config["random_agent_order"]:
            self.np_random.shuffle(self.agents)

        self.agents_iter = cycle(self.agents)
        
        obs, _, _, _, infos = self.step(None)

        return obs, infos

    def register_action(self, action):
        self.last_actions.append(action)
        if self.current_agent.agent_id == self.agents[-1].agent_id:
            self.deadline.advance_round()

    def step(self, action_dict):
        if action_dict:
            if self.current_agent.agent_id not in action_dict:
                raise ValueError(f"{self.current_agent.agent_id} not in {action_dict}")
            action = action_dict[self.current_agent.agent_id]
            action["agent_id"] = self.current_agent.agent_id
            self.register_action(action)

        while not self.deadline.reached() and (len(self.last_actions) < 1 or not self.last_actions[-1]["accept"]):
            
            self.current_agent = next(self.agents_iter)
            
            if isinstance(self.current_agent, RLAgent):
                if self.env_config["offer_max_first"] and len(self.last_actions) < 2:
                    action = self.current_agent.get_first_action(self.last_actions)
                else:
                    obs = self.current_agent.get_observation(self.last_actions, self.deadline)
                    rews = {self.current_agent.agent_id: 0}
                    return obs, rews, {"__all__": False}, {"__all__": False}, {}
            elif isinstance(self.current_agent, DefaultParty):
                action, timeout = self.current_agent.select_action(self.last_actions)
                if timeout:
                    break
            else:
                raise ValueError(f"Agent type {self.current_agent} not recognized")
            
            self.register_action(action)


        if self.last_actions[-1]["accept"]:
            # NOTE: disabled the following assert because of rl actions
            # assert self.last_actions[-1]["outcome"] == self.last_actions[-2]["outcome"]
            rew = {
                agent.agent_id: np.float32(agent.utility_function.get_utility(self.last_actions[0]["outcome"]))
                for agent in self.agents
            }
        else:
            rew = {agent.agent_id: np.float32(0) for agent in self.agents}

        [agent.final(self.last_actions) for agent in self.agents if isinstance(agent, DefaultParty)]

        opponent_utility = {type(agent).__name__: rew[agent.agent_id] for agent in self.agents}

        infos = {"__common__": {"opponent_utility": opponent_utility}}

        # TODO: write finalizing episode code
        return {}, rew, {"__all__": True}, {"__all__": True}, infos
        # return observation, reward, terminated, truncated, info


