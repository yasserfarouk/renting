import pytest

from environment.negotiation import NegotiationEnv


@pytest.fixture
def env() -> NegotiationEnv:
    env_config = {
        "agent_configs": ["ANL2022.Agent007", "ANL2022.Agent4410"],
        "scenario": {"random"},
        "deadline": {"rounds": 20, "ms": 10000},
        "random_agent_order": True,
        "offer_max_first": True,
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
