from collections import deque

import pytest
import numpy as np
from numpy.random import default_rng

from environment.agents.geniusweb import TRAINING_AGENTS
from environment.negotiation import Deadline
from environment.scenario import Scenario


@pytest.mark.parametrize("agent_class", TRAINING_AGENTS.values())
def test_initialisation(agent_class):
    scenario = Scenario.create_random(400, default_rng())
    agent_class("test", scenario.utility_functions[0], Deadline(10000), {})


@pytest.mark.parametrize("agent_class", TRAINING_AGENTS.values())
def test_opening_bid(agent_class):
    scenario = Scenario.create_random(400, default_rng())
    agent = agent_class("test", scenario.utility_functions[0], Deadline(10000), {})
    last_actions = deque()
    agent.select_action(last_actions)


@pytest.mark.parametrize("agent_class", TRAINING_AGENTS.values())
def test_accept_first_offer(agent_class):
    last_actions = deque()
    scenario = Scenario.create_random(400, default_rng())
    agent = agent_class("test", scenario.utility_functions[0], Deadline(10000), {})
    last_actions = deque()
    action, _ = agent.select_action(last_actions)
    assert isinstance(action, dict)
    last_actions.append(action)
    action = action.copy()
    action["accept"] = 1
    action["agent_id"] = "test_opponent"
    last_actions.append(action)
    agent.final(last_actions)


@pytest.mark.parametrize("agent_class", TRAINING_AGENTS.values())
def test_round_deadline(agent_class):
    scenario = Scenario.create_random(400, default_rng())
    outcome = np.array(
        scenario.utility_functions[1].max_utility_outcome, dtype=np.int64
    )
    last_actions = deque()
    bid = {"agent_id": "opponent", "outcome": outcome, "accept": 0}
    agent = agent_class("test", scenario.utility_functions[0], Deadline(10000, 10), {})
    for i in range(9):
        agent.select_action(last_actions)
        if i == 0:
            last_actions.append(bid)
    if hasattr(agent, "progress"):
        assert agent.progress.get(None) < 1
    elif hasattr(agent, "_progress"):
        assert agent._progress.get(None) < 1
    elif hasattr(agent, "_session_progress"):
        assert agent._session_progress.get(None) < 1
    else:
        raise ValueError("Agent does not have progress attribute")

    agent.select_action(last_actions)

    if hasattr(agent, "progress"):
        assert agent.progress.get(None) == 1
    elif hasattr(agent, "_progress"):
        assert agent._progress.get(None) == 1
    elif hasattr(agent, "_session_progress"):
        assert agent._session_progress.get(None) == 1
