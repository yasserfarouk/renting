import pytest
from environment.negotiation import NegotiationEnv


@pytest.fixture
def env() -> NegotiationEnv:
    env_config = {
        "agent_configs": [
            {"class": "ANL2022.Agent007"},
            {"class": "ANL2022.Agent4410"},
        ],
        "scenario": {"random"},
        "deadline": {"rounds": 20, "ms": 60000},
        "random_agent_order": True,
    }
    env = NegotiationEnv(env_config)
    return env


def test_smoke(env: NegotiationEnv):
    env


def test_reset(env: NegotiationEnv):
    env.reset()


def test_geniusweb_agent(env: NegotiationEnv):
    obs = env.reset()
    pass
