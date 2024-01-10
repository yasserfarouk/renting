import pytest

from environment.negotiation import NegotiationEnv
from environment.agents.geniusweb import AGENTS


@pytest.fixture
def env() -> NegotiationEnv:
    env_config = {
        "agent_configs": ["ANL2022.Agent007", "ANL2022.Agent4410"],
        "used_agents": AGENTS.keys(),
        "scenario": "random",
        "deadline": {"rounds": 20, "ms": 10000},
        "random_agent_order": True,
    }
    env = NegotiationEnv(env_config)
    return env


def test_smoke(env: NegotiationEnv):
    env


def test_reset(env: NegotiationEnv):
    env.reset()


def test_load_scenario():
    env_config = {
        "agent_configs": ["ANL2022.Agent007", "RL"],
        "used_agents": AGENTS.keys(),
        "scenario": "environment/scenarios/scenario_0000",
        "deadline": {"rounds": 20, "ms": 10000},
        "random_agent_order": False,
    }
    env = NegotiationEnv(env_config)
    env.reset()
