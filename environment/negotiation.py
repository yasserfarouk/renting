from collections import deque
from itertools import cycle
from pathlib import Path

import numpy as np
from geniusweb.party.DefaultParty import DefaultParty
from gymnasium.spaces import Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from environment.agents.geniusweb import AGENTS
from environment.agents.rl_agent import (HigaEtAl, RLAgent, RLAgentGraphObs, RLAgentGraphObs2,
                                         RLAgentStackedObs, RLAgentGraphObsPartialAdj, RLAgentGraphObsPartialAdjFixedLargeAction, Test)
from environment.deadline import Deadline
from environment.scenario import Scenario


REQUIRED_RL_AGENT = {
    "GraphToFixed": RLAgentStackedObs,
    "GraphToGraph": RLAgentStackedObs,
    "GraphToGraph2": RLAgentGraphObsPartialAdj,
    "AttentionGraphToGraph": RLAgentGraphObsPartialAdj,
    "AttentionGraphToGraph2": RLAgentGraphObs2,
    "PureGNN": RLAgentGraphObs,
    "HigaEtAl": HigaEtAl,
    "FixedToFixed": HigaEtAl,
    "FixedToFixed2": HigaEtAl,
    "GraphToGraphLargeFixedAction": RLAgentGraphObsPartialAdjFixedLargeAction,
    "Test": Test,
}

class NegotiationEnv(MultiAgentEnv):
    def __init__(self, env_config: dict):
        self.env_config = env_config
        self._agent_ids = set(a for a in env_config["agents"] if a.startswith("RL"))
        self.used_agents = {a: AGENTS[a] for a in env_config["used_agents"]}
        super().__init__()



    def reset(self, *, seed=None, options=None):
        # Call super's `reset()` method to (maybe) set the given `seed`.
        super().reset(seed=seed, options=options)


        self.scenario = Scenario.from_directory(Path(self.env_config["scenario"]))

        self.last_actions = deque(maxlen=2)
        self.opponent_encoding = 0


        if "rounds" in self.env_config["deadline"]:
            self.deadline = Deadline(rounds=self.env_config["deadline"]["rounds"])
        elif "ms" in self.env_config["deadline"]:
            self.deadline = Deadline(ms=self.env_config["deadline"]["ms"])
        else:
            raise ValueError("Deadline parameter not recognized")
        

        self.agents = []
        self.observation_space = Dict({})
        self.action_space = Dict({})
        agent_ids = set()
        for agent, utility_function in zip(self.env_config["agents"], self.scenario.utility_functions):
            if agent.startswith("RL"):
                agent_class = REQUIRED_RL_AGENT[agent.split("_")[1]]
                agent_init = agent_class(agent, utility_function, len(self.used_agents))
                self.observation_space[agent] = agent_init.observation_space
                self.action_space[agent] = agent_init.action_space
            elif agent == "random":
                selected_agent, agent_class = self.np_random.choice(list(self.used_agents.items()))
                self.opponent_encoding = self.env_config["used_agents"].index(selected_agent)
                agent_init = agent_class(selected_agent, utility_function, self.deadline)
            elif agent == "all":
                selected_agent, agent_class = list(self.used_agents.items())[self.env_config.worker_index - 1]
                self.opponent_encoding = self.env_config["used_agents"].index(selected_agent)
                agent_init = agent_class(selected_agent, utility_function, self.deadline)
            elif agent in self.used_agents:
                agent_class = self.used_agents[agent]
                agent_init = agent_class(agent, utility_function, self.deadline)
            else:
                raise ValueError("Agent not recognized")

            self.agents.append(agent_init)

            #NOTE: for now, force unique agents. Have to think about dealing with duplicates later.
            assert agent_init.agent_id not in agent_ids
            agent_ids.add(agent_init.agent_id)

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
            if not isinstance(action, dict):
                action = {"agent_id": self.current_agent.agent_id, "accept": action[0], "outcome": action[1:]}
            else:
                action["agent_id"] = self.current_agent.agent_id
            self.register_action(action)

        while not self.deadline.reached() and (len(self.last_actions) < 1 or not self.last_actions[-1]["accept"]):
            
            self.current_agent = next(self.agents_iter)
            
            if isinstance(self.current_agent, RLAgent):
                if self.env_config["offer_max_first"] and len(self.last_actions) < 2:
                    action = self.current_agent.get_first_action(self.last_actions)
                else:
                    obs = self.current_agent.get_observation(self.last_actions, self.deadline, self.opponent_encoding)
                    obs = {self.current_agent.agent_id: obs}
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
            assert len(self.last_actions) == 2 #Check that RL agent did not make a unvailable action
            utility_all_agents = {
                agent.agent_id: np.float32(agent.utility_function.get_utility(self.last_actions[0]["outcome"]))
                for agent in self.agents
            }
        else:
            utility_all_agents = {agent.agent_id: np.float32(0) for agent in self.agents}

        [agent.final(self.last_actions) for agent in self.agents if isinstance(agent, DefaultParty)]

        rew = {agent.agent_id: utility_all_agents[agent.agent_id] for agent in self.agents if isinstance(agent, RLAgent)}

        #NOTE: the following line does likely not lead to a correct observation for the 
        # agent. However, this observation is not used for training, so it should not matter.
        # We prevent RLLib from sampling a random obs from an old obs_space instead, which causes errors.
        obs = {agent.agent_id: agent.get_observation(self.last_actions, self.deadline, self.opponent_encoding) for agent in self.agents if isinstance(agent, RLAgent)}

        self.scenario.calculate_specials()
        infos = {"__common__": {"utility_all_agents": utility_all_agents}}

        return obs, rew, {"__all__": True}, {"__all__": True}, infos
        # return observation, reward, terminated, truncated, info


    # def observation_space_contains(self, obs):
    #     return all([self.observation_space.contains(v) for v in obs.values()])
    
    # def action_space_contains(self, action):
    #     return all([self.action_space.contains(v) for v in action.values()])
    
    def observation_space_sample(self, agent_ids: list = None):
        sample = self.observation_space.sample()
        if agent_ids:
            sample = {a: obs for a, obs in sample.items() if a in agent_ids}
        for obs in sample.values():
            if "accept_mask" in obs:
                obs["accept_mask"] = np.array([1, 0], dtype=bool)
        return sample
    
    def action_space_sample(self, agent_ids: list = None):
        sample = self.action_space.sample()
        if agent_ids:
            sample = {a: obs for a, obs in sample.items() if a in agent_ids}
        for action in sample.values():
            if "accept" in action:
                action["accept"] = np.int64(0)
            else:
                action[0] = 0
        return sample
