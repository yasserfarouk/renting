from collections import deque
from itertools import cycle
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from environment.agents.geniusweb import AGENTS
from environment.scenario import Scenario
from environment.deadline import Deadline
from environment.agents.rl_agent import RLAgent
from geniusweb.party.DefaultParty import DefaultParty


class NegotiationEnv(MultiAgentEnv):
    def __init__(self, env_config: dict):
        self.env_config = env_config
        self._agent_ids = set(env_config["RL_agents"])
        self.used_agents = {a: AGENTS[a] for a in env_config["used_agents"]}
        super().__init__()



    def reset(self, *, seed=None, options=None):
        # Call super's `reset()` method to (maybe) set the given `seed`.
        super().reset(seed=seed, options=options)

        if self.env_config["scenario"] == "random":
            self.scenario = Scenario.create_random([100, 10000], self.np_random)
        else:
            self.scenario = Scenario.from_directory(Path(self.env_config["scenario"]))


        values_per_objective = [len(v) for v in self.scenario.objectives.values()]
        max_num_values = max(values_per_objective)
        num_objectives = len(self.scenario.objectives)
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
        #         "time": Box(-0, 1, dtype=np.float32),
        #     }
        # )
        self.observation_space = Dict(
            {
                "head_node": Box(np.array([0, 0]), np.array([np.inf, 1]), dtype=np.float32),
                # #objectives, time (implicit num offers)
                "objective_nodes": Box(np.array([[1, 0]] * num_objectives), np.array([[np.inf, 1]] * num_objectives), shape=(num_objectives, 2), dtype=np.float32),
                # #values, weight,
                "value_nodes": Box(0, 1, shape=(num_objectives, max_num_values, 4), dtype=np.float32),
                # weight, average offered, my outcome, opp outcome
                "value_nodes_mask": Box(0, 1, shape=(num_objectives, max_num_values), dtype=bool),
                "opponent_encoding": Discrete(len(self.used_agents)),
                "accept_mask": Box(0, 1, shape=(2,), dtype=bool),
            }
        )
        # self.observation_space = Dict(
        #     {
        #         "head_node": Box(np.array([0, 0]), np.array([np.inf, 1]), dtype=np.float32),
        #         # #objectives, time (implicit num offers)
        #         "objective_nodes": Box(0, np.inf, shape=(2, 2), dtype=np.float32),
        #         # #values, weight,
        #         "value_nodes": Box(0, 1, shape=(2, 9, 4), dtype=np.float32),
        #         # weight, average offered, my outcome, opp outcome
        #         "value_nodes_mask": Box(0, 1, shape=(2, 9), dtype=bool),
        #     }
        # )


        # self.observation_space = Dict(
        #     {
        #         "head_node": Box(np.array([0, 0]), np.array([np.inf, 1]), dtype=np.float32),
        #         # #objectives, time (implicit num offers)
        #         "objective_nodes": Box(0, np.inf, shape=(num_objectives, 2), dtype=np.float32),
        #         # #values, weight,
        #         "value_nodes": Box(0, 1, shape=(num_objectives, max_num_values, 4), dtype=np.float32),
        #         # weight, average offered, my outcome, opp outcome
        #         "value_nodes_mask": Box(0, 1, shape=(num_objectives, max_num_values), dtype=bool),
        #     }
        # )
        # self.observation_space = Dict(
        #     {
        #         "head_node": Box(
        #             np.array([0, 0]), np.array([np.inf, 1]), dtype=np.float32
        #         ),  # #objectives, time (implicit num offers)
        #         "objective_nodes": Sequence(
        #             Dict({"test1": Box(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), dtype=np.float32)}),
        #             stack=True,
        #         ),  # #values, weight,
        #         "value_nodes": Sequence(
        #             Dict({"test1": Box(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), dtype=np.float32)}),
        #             stack=False,
        #         ),  # weight, average offered, my outcome, opp outcome
        #     }
        # )

        # self.observation_space = Dict({agent_id: self.observation_space for agent_id in self._agent_ids})
        # self.action_space = Dict({agent_id: self.action_space for agent_id in self._agent_ids})

        self.last_actions = deque(maxlen=2)

        # TODO: list based utility functions in scenario
        utility_functions = [self.scenario.utility_function_A, self.scenario.utility_function_B]
        playing_agents: list = self.env_config["always_playing"]
        if len(playing_agents) == 1:
            playing_agents.append(self.env_config["opponent"])
        assert len(playing_agents) == 2 #NOTE: dirty fix to enforce 2 agents, must deal with this later

        if "rounds" in self.env_config["deadline"]:
            self.deadline = Deadline(rounds=self.env_config["deadline"]["rounds"])
        elif "ms" in self.env_config["deadline"]:
            self.deadline = Deadline(ms=self.env_config["deadline"]["ms"])
        else:
            raise ValueError("Deadline parameter not recognized")
        
        self.opponent_encoding = 0

        self.agents = []
        playing_agent_ids = set()
        for player, utility_function in zip(playing_agents, utility_functions):
            if player.startswith("RL"):
                agent_class = RLAgent
                agent_id = player
            elif player == "random":
                selected_player, agent_class = self.np_random.choice(list(self.used_agents.items()))
                self.opponent_encoding = self.env_config["used_agents"].index(selected_player)
                agent_id = selected_player
            elif player == "all":
                selected_player, agent_class = list(self.used_agents.items())[self.env_config.worker_index - 1]
                self.opponent_encoding = self.env_config["used_agents"].index(selected_player)
                agent_id = selected_player
            elif player in self.used_agents:
                agent_class = self.used_agents[player]
                agent_id = player
            else:
                raise ValueError("Agent config not recognized")
            
            assert agent_id not in playing_agent_ids #NOTE ditry fix to enforce unique agent ids, must deal with this later
            playing_agent_ids.add(agent_id)

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
                    obs = self.current_agent.get_observation(self.last_actions, self.deadline, self.opponent_encoding)
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
            assert len(self.last_actions) == 2
            # NOTE: disabled the following assert because of rl actions
            # assert self.last_actions[-1]["outcome"] == self.last_actions[-2]["outcome"]
            rew_all_agents = {
                agent.agent_id: np.float32(agent.utility_function.get_utility(self.last_actions[0]["outcome"]))
                for agent in self.agents
            }
        else:
            rew_all_agents = {agent.agent_id: np.float32(0) for agent in self.agents}

        [agent.final(self.last_actions) for agent in self.agents if isinstance(agent, DefaultParty)]

        opponent_utility = {type(agent).__name__: rew_all_agents[agent.agent_id] for agent in self.agents}

        rew = {agent.agent_id: rew_all_agents[agent.agent_id] for agent in self.agents if isinstance(agent, RLAgent)}
        infos = {"__common__": {"opponent_utility": opponent_utility}}

        # TODO: write finalizing episode code
        return {}, rew, {"__all__": True}, {"__all__": True}, infos
        # return observation, reward, terminated, truncated, info


    def observation_space_contains(self, obs):
        return all([self.observation_space.contains(v) for v in obs.values()])
    
    def action_space_contains(self, action):
        return all([self.action_space.contains(v) for v in action.values()])
    
    def observation_space_sample(self, agent_ids: list = None):
        assert "RL_0" in self._agent_ids
        obs = self.observation_space.sample()
        obs["accept_mask"] = np.array([1, 0], dtype=bool)
        # sample_agents = agent_ids if agent_ids else self._agent_ids
        return {"RL_0": obs}
    
    def action_space_sample(self, agent_ids: list = None):
        assert "RL_0" in self._agent_ids
        action = self.action_space.sample()
        action["accept"] = np.int64(0)
        # sample_agents = agent_ids if agent_ids else )
        return {"RL_0": action}
