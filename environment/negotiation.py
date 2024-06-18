from collections import deque
from itertools import cycle
from pathlib import Path

import numpy as np
from geniusweb.party.DefaultParty import DefaultParty
from numpy.random import default_rng
from pettingzoo import ParallelEnv

from environment.agents.geniusweb import AGENTS
from environment.agents.rl_agent import GraphObs, HigaEtAl, RLAgent
from environment.deadline import Deadline
from environment.scenario import Scenario


REQUIRED_RL_AGENT = {
    "GNN": GraphObs,
    "HigaEtAl": HigaEtAl,
}

class NegotiationEnvZoo(ParallelEnv):
    metadata = {"name": "negotiation_env"}

    def __init__(self, env_config, render_mode=None):
        self.possible_agents = [a for a in env_config["agents"] if a.startswith("RL")]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

        if env_config["scenario"] == "random":
            self.scenario = Scenario.create_random([200, 1000], default_rng(0), 5)
        else:
            self.scenario = Scenario.from_directory(Path(env_config["scenario"]))

        self.env_config = env_config
        self.used_agents = {a: AGENTS[a] for a in env_config["used_agents"]}

    def observation_space(self, agent):
        return REQUIRED_RL_AGENT[agent.split("_")[1]].observation_space(self.scenario.utility_functions[0],len(self.used_agents))

    def action_space(self, agent):
        return REQUIRED_RL_AGENT[agent.split("_")[1]].action_space(self.scenario.utility_functions[0])

    def reset(self, *, seed=None, options=None):
        if not hasattr(self, "np_random"):
            self.np_random = default_rng(seed) if seed else default_rng(0)

        self.agents = self.possible_agents

        if self.env_config["scenario"] == "random":
            self.scenario = Scenario.create_random([200, 1000], self.np_random, 5)
        else:
            self.scenario = Scenario.from_directory(Path(self.env_config["scenario"]), self.np_random)

        self.last_actions = deque(maxlen=2)
        self.opponent_encoding = 0

        if "rounds" in self.env_config["deadline"]:
            self.deadline = Deadline(rounds=self.env_config["deadline"]["rounds"])
        elif "ms" in self.env_config["deadline"]:
            self.deadline = Deadline(ms=self.env_config["deadline"]["ms"])
        else:
            raise ValueError("Deadline parameter not recognized")
        
        self._agents = []
        self.observation_spaces = {}
        self.action_spaces = {}
        agent_ids = set()
        for agent, utility_function in zip(self.env_config["agents"], self.scenario.utility_functions):
            if agent.startswith("RL"):
                agent_class = REQUIRED_RL_AGENT[agent.split("_")[1]]
                agent_init = agent_class(agent, utility_function, len(self.used_agents))
                self.observation_spaces[agent] = agent_init.observation_space
                self.action_spaces[agent] = agent_init.action_space
            elif agent == "random":
                selected_agent, agent_class = self.np_random.choice(list(self.used_agents.items()))
                self.opponent_encoding = self.env_config["used_agents"].index(selected_agent)
                agent_init = agent_class(selected_agent, utility_function, self.deadline)
            elif agent == "all":
                used_agents_list = list(self.used_agents.items())
                selected_agent, agent_class = used_agents_list[self.worker_id % len(used_agents_list)]
                self.opponent_encoding = self.env_config["used_agents"].index(selected_agent)
                agent_init = agent_class(selected_agent, utility_function, self.deadline)
            elif agent in self.used_agents:
                agent_class = self.used_agents[agent]
                agent_init = agent_class(agent, utility_function, self.deadline)
            else:
                raise ValueError("Agent not recognized")

            self._agents.append(agent_init)

            #NOTE: for now, force unique agents. Have to think about dealing with duplicates later.
            assert agent_init.agent_id not in agent_ids
            agent_ids.add(agent_init.agent_id)

        if self.env_config["random_agent_order"]:
            self.np_random.shuffle(self._agents)

        self.agents_iter = cycle(self._agents)
        
        obs, _, _, _, infos = self.step(None)

        return obs, infos

    def register_action(self, action):
        self.last_actions.append(action)
        if self.current_agent.agent_id == self._agents[-1].agent_id:
            self.deadline.advance_round()

    def step(self, action_dict):
        if action_dict:
            if self.current_agent.agent_id not in action_dict:
                raise ValueError(f"{self.current_agent.agent_id} not in {action_dict}")
            action = action_dict[self.current_agent.agent_id]
            if not isinstance(action, dict):
                action = {"agent_id": self.current_agent.agent_id, "accept": action[0], "outcome": action[1:]}
            else:
                action["agent_id"] = self.current_agent.agent_id
            self.register_action(action)

        while not self.deadline.reached() and (len(self.last_actions) < 1 or not self.last_actions[-1]["accept"]):
            
            self.current_agent = next(self.agents_iter)
            
            if isinstance(self.current_agent, RLAgent):
                if hasattr(self.current_agent, "offer_max_first") and self.current_agent.offer_max_first and len(self.last_actions) < 2:
                    action = self.current_agent.get_first_action(self.last_actions)
                else:
                    obs = self.current_agent.get_observation(self.last_actions, self.deadline, self.opponent_encoding)
                    obs = {self.current_agent.agent_id: obs}
                    rews = {self.current_agent.agent_id: 0}
                    return obs, rews, {self.current_agent.agent_id: False}, {self.current_agent.agent_id: False}, {}
            elif isinstance(self.current_agent, DefaultParty):
                action, timeout = self.current_agent.select_action(self.last_actions)
                if timeout:
                    break
            else:
                raise ValueError(f"Agent type {self.current_agent} not recognized")
            
            self.register_action(action)


        if self.last_actions[-1]["accept"]:
            assert len(self.last_actions) == 2 #Check that RL agent did not make a unvailable action
            utility_all_agents = {
                agent.agent_id: np.float32(agent.utility_function.get_utility(self.last_actions[0]["outcome"]))
                for agent in self._agents
            }
        else:
            utility_all_agents = {agent.agent_id: np.float32(0) for agent in self._agents}

        [agent.final(self.last_actions) for agent in self._agents if isinstance(agent, DefaultParty)]

        rew = {agent.agent_id: utility_all_agents[agent.agent_id] for agent in self._agents if isinstance(agent, RLAgent)}

        #NOTE: the following line does likely not lead to a correct observation for the 
        # agent. However, this observation is not used for training, so it should not matter.
        # We prevent RLLib from sampling a random obs from an old obs_space instead, which causes errors.
        obs = {agent.agent_id: agent.get_observation(self.last_actions, self.deadline, self.opponent_encoding) for agent in self._agents if isinstance(agent, RLAgent)}

        infos = {agent.agent_id: {"utility_all_agents": utility_all_agents, "rounds_played": self.deadline.round, "self_accepted": (self.last_actions[-1]["accept"] and self.last_actions[-1]["agent_id"] == agent.agent_id), "found_agreement": bool(self.last_actions[-1]["accept"])} for agent in self._agents if isinstance(agent, RLAgent)}

        return obs, rew, {a: True for a in self.agents}, {a: True for a in self.agents}, infos
