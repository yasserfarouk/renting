import time
from collections import deque
from itertools import cycle
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase
from typing import Union

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from environment.agents.geniusweb import AGENTS
from environment.scenario import Scenario


class NegotiationEnv(MultiAgentEnv):
    def __init__(self, env_config: dict):
        self.env_config = env_config
        self.agent_configs = env_config["agent_configs"]
        self._agent_ids = set([f"agent_{n}" for n in range(len(self.agent_configs))])
        super().__init__()

    def reset(self, seed=None, options=None):
        # Call super's `reset()` method to (maybe) set the given `seed`.
        super().reset(seed=seed, options=options)

        if "random" in self.env_config["scenario"]:
            self.scenario = Scenario.create_random([100, 10000], self.np_random)
        else:
            self.scenario = Scenario.from_directory(Path(self.env_config["scenario"]))

        offer_action_space = MultiDiscrete([len(v) for v in self.scenario.objectives.values()], dtype=np.int32)

        self.action_space = Dict(
            {
                "offer": offer_action_space,
                "accept": Discrete(2),
            }
        )
        self.observation_space = Dict(
            {
                "my_offer": offer_action_space,
                "opp_offer": offer_action_space,
                "time": Box(-0.1, 1.1, dtype=np.float32),
            }
        )


        self.last_actions = deque(maxlen=len(self._agent_ids))

        # TODO: list based utility functions in scenario
        utility_functions = [self.scenario.utility_function_A, self.scenario.utility_function_B]
        # for utility_function, agent_config in zip(utility_functions, self.agent_configs):
        #     agent_config["utility_function"] = utility_function


        if "rounds" in self.env_config["deadline"]:
            self.deadline = Deadline(rounds=self.env_config["deadline"]["rounds"])
        elif "ms" in self.env_config["deadline"]:
            self.deadline = Deadline(ms=self.env_config["deadline"]["ms"])
        else:
            raise ValueError("Deadline parameter not recognized")

        self.agents = []
        self.rl_agents = set()
        for agent_id, agent_config, utility_function in zip(self._agent_ids, self.agent_configs, utility_functions):
            if agent_config == "RL":
                self.rl_agents.add(agent_id)
                agent = None
            elif agent_config == "random":
                agent_class = self.np_random.choice(list(AGENTS.values()))
                agent = agent_class(agent_id, utility_function, self.deadline)
            elif agent_config in AGENTS:
                agent_class = AGENTS[agent_config]
                agent = agent_class(agent_id, utility_function, self.deadline)
            else:
                raise ValueError("Agent config not recognized")

            self.agents.append((agent_id, agent, utility_function))

        if self.env_config["random_agent_order"]:
            self.np_random.shuffle(self.agents)

        self.agents_iter = cycle(self.agents)
        
        obs, _, _, _, infos = self.step(None)

        return obs, infos


    def step(self, action_dict):
        if action_dict:
            if self.current_agent_id not in action_dict:
                raise ValueError(f"{self.current_agent_id} not in {action_dict}")
            action = action_dict[self.current_agent_id]
            action["agent_id"] = self.current_agent_id
            self.last_actions.append(action)

        while not self.deadline.reached() and (len(self.last_actions) < 1 or not self.last_actions[-1]["accept"]):
            
            self.current_agent_id, agent, utility_function = next(self.agents_iter)
            
            if self.current_agent_id in self.rl_agents:
                if self.env_config["offer_max_first"] and len(self.last_actions) < 2:
                    offer = np.array(utility_function.get_max_utility_bid(), dtype=np.int32)
                    action = {"agent_id": self.current_agent_id, "offer": offer, "accept": 0}
                else:
                    obs = self.get_observation(self.current_agent_id, self.last_actions)
                    rews = {self.current_agent_id: 0}
                    return obs, rews, {"__all__": False}, {"__all__": False}, {}
            else:
                action: dict = agent.select_action(self.last_actions)

            self.last_actions.append(action)


        if self.last_actions[-1]["accept"]:
            # NOTE: disabled the following assert because of rl actions
            # assert self.last_actions[-1]["offer"] == self.last_actions[-2]["offer"]
            rew = {
                agent_id: np.float32(utility_function.get_utility(self.last_actions[-1]["offer"]))
                for agent_id, _, utility_function in self.agents
            }
        else:
            rew = {agent_id: np.float32(0) for agent_id, _, _ in self.agents}

        [agent.final(self.last_actions) for _, agent, _ in self.agents if agent]


        # TODO: write finalizing episode code
        return {}, rew, {"__all__": True}, {"__all__": True}, {}
        # return observation, reward, terminated, truncated, info

    # def step(self, action_dict):
    #     if self.current_agent_id not in action_dict:
    #         raise ValueError(f"{self.current_agent_id} not in {action_dict}")
    #     action = action_dict[self.current_agent_id]
    #     action["agent_id"] = self.current_agent_id
    #     self.last_actions.append(action)
    #     self.negotiate()

    def get_observation(self, agent_id, last_actions):
        #TODO: construct proper observation
        obs = {
            "my_offer": last_actions[0]["offer"],
            "opp_offer": last_actions[1]["offer"],
            "time": np.array([self.deadline.get_progress()], dtype=np.float32)
        }
        return {agent_id: obs}


class BidHistory:
    def __init__(self) -> None:
        self.test = None


class Deadline:
    #TODO: fix infinite deadline, currently it is > 1 year
    def __init__(self, ms: int=2**35, rounds: int = None):
        assert ms or rounds
        if ms and ms <= 0:
            raise ValueError(f"ms must be positive but is {ms}")
        if rounds and rounds <= 2:
            raise ValueError(f"rounds must be at least 3 but is {rounds}")

        self.start_time_ms = time.time() * 1000
        self.ms = ms
        self.rounds = rounds
        self.round = 0

    def get_progress(self) -> Union[float, int]:
        if self.rounds:
            progress = self.round / self.rounds
        else:
            progress = (time.time() * 1000 - self.start_time_ms) / self.ms
        
        # clip progress to [0, 1]
        return min(max(progress, 0), 1)

    def reached(self) -> bool:
        return True if self.get_progress() == 1 else False

    def advance_round(self):
        self.round += 1

