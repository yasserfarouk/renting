import numpy as np
import pytest

from environment.negotiation import NegotiationEnv


@pytest.fixture
def env() -> NegotiationEnv:
    env_config = {
        "agent_configs": ["RL", "ANL2022.Agent007"],
        "scenario": {"random"},
        "deadline": {"rounds": 20, "ms": 10000},
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
    offer = np.array(env.agents[0][2].get_max_utility_bid(), dtype=np.int32)
    action = {"agent_id": agent_id, "offer": offer, "accept": 0}
    obs, rews, terminated, truncated, _ = env.step({agent_id: action})
    obs = next(iter(obs.values()))
    assert env.observation_space.contains(obs)
