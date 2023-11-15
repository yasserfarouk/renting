from itertools import cycle
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from environment.agents.geniusweb import AGENTS
from environment.scenario import Scenario
from environment.deadline import Deadline


class NegotiationEnv(MultiAgentEnv):
    def __init__(self, env_config: dict):
        print(env_config)
        self.env_config = env_config
        self.agent_configs = env_config["agent_configs"]
        self._agent_ids = set([f"agent_{n}" for n in range(len(self.agent_configs))])

        # self.action_space = spaces.Box(low=0, high=1, shape=(2,))
        # self.observation_space = spaces.Box(low=0, high=1, shape=(6,))

        # self._env = env
        # self._num_players = len(self._env.observation_spec())
        # self._ordered_agent_ids = [
        #     PLAYER_STR_FORMAT.format(index=index) for index in range(self._num_players)
        # ]
        # # RLLib requires environments to have the following member variables:
        # # observation_space, action_space, and _agent_ids
        # self._agent_ids = set(self._ordered_agent_ids)

        # # RLLib expects a dictionary of agent_id to observation or action,
        # # Melting Pot uses a tuple, so we convert them here
        # self.observation_space = self._convert_spaces_tuple_to_dict(
        #     utils.spec_to_space(self._env.observation_spec()),
        #     remove_world_observations=True,
        # )
        # self.action_space = self._convert_spaces_tuple_to_dict(
        #     utils.spec_to_space(self._env.action_spec())
        # )
        super().__init__()

    def reset(self, seed=None, options=None):
        # Call super's `reset()` method to (maybe) set the given `seed`.
        super().reset(seed=seed, options=options)
        self.accept = None

        if "random" in self.env_config["scenario"]:
            self.scenario = Scenario.create_random("random", [100, 10000], self.np_random)
        else:
            self.scenario = Scenario(self.env_config["scenario"])


        self.action_space = spaces.Dict(
            {
                "offer": self.scenario.action_space,
                "accept": spaces.Discrete(2),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "my_offer": self.scenario.action_space,
                "opp_offer": self.scenario.action_space,
                "time": spaces.Box(0.0, 1.0, dtype=np.float32),
            }
        )

        if self.env_config["random_agent_order"]:
            self.np_random.shuffle(self.agent_configs)

        # TODO: list based utility functions in scenario
        utility_functions = [self.scenario.utility_function_A, self.scenario.utility_function_B]
        for utility_function, agent_config in zip(utility_functions, self.agent_configs):
            agent_config["utility_function"] = utility_function


        if "rounds" in self.env_config["deadline"]:
            self.deadline = Deadline(rounds=self.env_config["deadline"]["rounds"])
        elif "ms" in self.env_config["deadline"]:
            self.deadline = Deadline(ms=self.env_config["deadline"]["ms"])
        else:
            raise ValueError("Deadline parameter not recognized")

        self.agents = []
        for agent_id, agent_config in zip(self._agent_ids, self.agent_configs):
            if "RL" in agent_config and agent_config["RL"]:
                self.agents.append({"RL": None})
                continue
            elif "class" in agent_config:
                agent_name = agent_config["class"]
                agent_class = AGENTS[agent_config["class"]]
            elif "random" in agent_config and agent_config["random"]:
                agent_name, agent_class = self.np_random.choice(list(AGENTS.items()))
            else:
                raise ValueError("Agent config not recognized")
            agent_name = agent_name.replace(".", "_")
            agent = agent_class(agent_name, agent_config["utility_function"], self.deadline)
            self.agents.append({agent_id: agent})

        self.agents_iter = cycle(self.agents.items())
        obs, _, _, _, infos = self.negotiate(None)

        # obs = {name: np.zeros(0) for name in self._agent_ids}
        # infos = {}

        return obs, infos
    
    def increment_agent_index(self):
        self.agent_index = (self.agent_index + 1) % len(self.agents)

    def negotiate(self, action):
        while not self.deadline.reached() and not self.accept:
            agent_id, agent = next(self.agents_iter)

            # self.agent_index = (self.agent_index + 1) % len(self.agents)
            
            if agent_id == "RL" :
                obs = action
                obs, rew = 0, 0
                return obs, rew, {"__all__": False}, {"__all__": False}, {}
            else:
                action = agent.select_action()


            if accept:
                pass
            self.check_action = action
            self.action_count += 1

            # if isinstance(agent, RLAgent):
            #     obs = agent.generate_observation()
            #     obs = {agent.name: obs}
            #     return obs, {}#info
            # else:

            [agent.communicate_action(action) for agent in self.agents]
            self.agent_index = (self.agent_index + 1) % len(self.agents)

            if isinstance(action, Accept):
                self.accept = action
                break

            # self.agent_index = (self.agent_index + 1) % len(self.agents)

            # next_agent = self.agents[self.agent_index]
            # if isinstance(next_agent, RLAgent):
            #     obs = next_agent.generate_observation()
            #     obs = {next_agent.name: obs}
            #     rew = {next_agent.name: np.array(0, dtype=np.float32)}
            #     return obs, rew, {"__all__": False}, {"__all__": False}, {}

        if self.accept:
            rew = {
                agent.name: np.float32(agent.profile.getUtility(self.accept._bid))
                for agent in self.agents
            }
            social_welfare = sum(rew.values())
            rew = {name: reward * social_welfare for name, reward in rew.items()}
            agreements = Agreements(
                {agent.ID: self.accept._bid for agent in self.agents}
            )
        else:
            rew = {agent.name: np.float32(0) for agent in self.agents}
            agreements = Agreements({})

        obs = {
            agent.name: agent.generate_observation()
            for agent in self.agents
            if isinstance(agent, RLAgent)
        }
        [agent.finish(agreements) for agent in self.agents]

        self.postprocessed = True

        # TODO: write finalizing episode code
        return obs, rew, {"__all__": True}, {"__all__": True}, {}
        # return observation, reward, terminated, truncated, info

    def step(self, action_dict):
        current_agent_name = self.agents[self.agent_index].name
        if current_agent_name not in action_dict:
            raise ValueError(f"{current_agent_name} not in {action_dict}")
        # assert current_agent_name in rl_action
        return self.negotiate(action_dict[current_agent_name])

    # def step(self, action_dict):
    #     observations, rewards, terminated, truncated, info = {}, {}, {}, {}, {}
    #     return observations, rewards, terminated, truncated, info



# class NegotiationEnvOld(MultiAgentEnv):
#     """Custom Environment that follows gym interface"""

#     metadata = {"render.modes": ["human"]}

#     def __init__(self, env_config: dict):
#         super().__init__()
#         self.num_rl_opponents = env_config["num_rl_opponents"]

#         self.available_agents = AGENTS.copy()

#         rl_agents = {f"RLAgent_{i}": RLAgent for i in range(self.num_rl_opponents)}
#         self.available_agents.update(rl_agents)

#         self.deadline_ms = env_config["deadline_ms"]

#         if "domain_size" in env_config:
#             self.fixed_domain_size = env_config["domain_size"]
#         else:
#             self.fixed_domain_size = None

#         self.action_space = Dict(
#             {
#                 "util_goal": Box(0.0, 1.0, dtype=np.float32),
#                 # "opp_util_goal": Box(0.0, 1.0, dtype=np.float32),
#                 "accept": Discrete(2),
#             }
#         )
#         self.observation_space = Box(0.0, 1.0, shape=(5,), dtype=np.float32)

#         self._agent_ids = set(["RLAgent"] + [f"RLAgent_{i}" for i in range(self.num_rl_opponents)])

#     def reset(self, seed=0, options=None) -> dict:
#         self.agent_index = 0
#         self.action_count = 0
#         self.accept = None
#         self.postprocessed = False
#         self.check_action = None
#         # self.agreement = Agreements({})

#         if not self.fixed_domain_size:
#             domain_size = self.np_random.randint(200, 2000)
#         else:
#             domain_size = self.fixed_domain_size

#         self.scenario = Scenario.create_random("scenario", domain_size)

#         agent_B_name, agent_B_class = self.np_random.choice(list(self.available_agents.items()))
#         agent_B_name = agent_B_name.replace(".", "_")  # TODO: make sure a . works

#         # TODO: this does not work, wrapped by default for now
#         # if isinstance(agent_B_class, DefaultParty):
#         #     agent_B_class = geniusweb_wrapper(agent_B_class)

#         self.deadline = Deadline(self.deadline_ms)

#         self.agents = [
#             geniusweb_wrapper(RLAgent)(
#                 "RLAgent", self.scenario.preferences_A, self.deadline
#             ),
#             geniusweb_wrapper(agent_B_class)(
#                 agent_B_name, self.scenario.preferences_B, self.deadline
#             ),
#         ]

#         self.np_random.shuffle(self.agents)#TODO: fix
#         # self.agents = self.agents[::-1]

#         obs, _, _, _, infos = self.negotiate()

#         return obs, infos

#     def increment_agent_index(self):
#         self.agent_index = (self.agent_index + 1) % len(self.agents)

#     def negotiate(self, rl_action=None):
#         self.current_rl_action = rl_action
#         while not self.deadline.reached() and not self.accept:
#             agent = self.agents[self.agent_index]

#             # self.agent_index = (self.agent_index + 1) % len(self.agents)
            
#             if isinstance(agent, RLAgent) and not self.current_rl_action:
#                 obs = agent.generate_observation()
#                 obs = {agent.name: obs}
#                 rew = {agent.name: np.array(0, dtype=np.float32)}
#                 return obs, rew, {"__all__": False}, {"__all__": False}, {}
#             elif isinstance(agent, RLAgent) and self.current_rl_action:
#                 action = agent.translate_action(self.current_rl_action)
#                 self.current_rl_action = None
#             else:
#                 action = agent.select_action()

#             self.check_action = action
#             self.action_count += 1

#             # if isinstance(agent, RLAgent):
#             #     obs = agent.generate_observation()
#             #     obs = {agent.name: obs}
#             #     return obs, {}#info
#             # else:

#             [agent.communicate_action(action) for agent in self.agents]
#             self.agent_index = (self.agent_index + 1) % len(self.agents)

#             if isinstance(action, Accept):
#                 self.accept = action
#                 break

#             # self.agent_index = (self.agent_index + 1) % len(self.agents)

#             # next_agent = self.agents[self.agent_index]
#             # if isinstance(next_agent, RLAgent):
#             #     obs = next_agent.generate_observation()
#             #     obs = {next_agent.name: obs}
#             #     rew = {next_agent.name: np.array(0, dtype=np.float32)}
#             #     return obs, rew, {"__all__": False}, {"__all__": False}, {}

#         if self.accept:
#             rew = {
#                 agent.name: np.float32(agent.profile.getUtility(self.accept._bid))
#                 for agent in self.agents
#             }
#             social_welfare = sum(rew.values())
#             rew = {name: reward * social_welfare for name, reward in rew.items()}
#             agreements = Agreements(
#                 {agent.ID: self.accept._bid for agent in self.agents}
#             )
#         else:
#             rew = {agent.name: np.float32(0) for agent in self.agents}
#             agreements = Agreements({})

#         obs = {
#             agent.name: agent.generate_observation()
#             for agent in self.agents
#             if isinstance(agent, RLAgent)
#         }
#         [agent.finish(agreements) for agent in self.agents]

#         self.postprocessed = True

#         # TODO: write finalizing episode code
#         return obs, rew, {"__all__": True}, {"__all__": True}, {}
#         # return obs, rew, done, truncated, info

#     def step(self, rl_action):
#         current_agent_name = self.agents[self.agent_index].name
#         if current_agent_name not in rl_action:
#             raise ValueError(f"{current_agent_name} not in {rl_action}")
#         # assert current_agent_name in rl_action
#         return self.negotiate(rl_action[current_agent_name])
