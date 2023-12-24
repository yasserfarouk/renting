import numpy as np
import pytest

from environment.agents.geniusweb import AGENTS
from environment.negotiation import NegotiationEnv

DEADLINE = 20

@pytest.fixture
def env() -> NegotiationEnv:
    env_config = {
        "agent_configs": ["RL", "BASIC.HardlinerAgent"],
        "used_agents": AGENTS.keys(),
        "scenario": "random",
        "deadline": {"rounds": DEADLINE, "ms": 10000},
        "random_agent_order": False,
        "offer_max_first": True,
    }
    env = NegotiationEnv(env_config)
    return env


def test_reset(env: NegotiationEnv):
    obs, _ = env.reset()
    obs = next(iter(obs.values()))
    assert env.observation_space.contains(obs)


def test_observation(env: NegotiationEnv):
    obs, _ = env.reset()
    agent_id = next(iter(obs.keys()))
    outcome = np.array(env.agents[0].utility_function.max_utility_outcome, dtype=np.int32)
    action = {"agent_id": agent_id, "outcome": outcome, "accept": 0}
    obs, rews, terminated, truncated, _ = env.step({agent_id: action})
    obs = next(iter(obs.values()))
    assert env.observation_space.contains(obs)


def test_deadline(env: NegotiationEnv):
    obs, _ = env.reset()
    agent_id = next(iter(obs.keys()))
    outcome = np.array(env.agents[0].utility_function.max_utility_outcome, dtype=np.int32)
    max_action = {"agent_id": agent_id, "outcome": outcome, "accept": 0}
    for i in range(1, DEADLINE):
        assert env.deadline.round == i
        obs, rews, terminated, truncated, _ = env.step({agent_id: max_action})
    assert terminated["__all__"]
    assert rews[agent_id] == 0
    assert env.deadline.reached()
