from typing import cast

import pytest
from numpy.random import default_rng
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.ActionWithBid import ActionWithBid
from geniusweb.actions.PartyId import PartyId
from geniusweb.inform.Agreements import Agreements
from geniusweb.party.DefaultParty import DefaultParty

from environment.agents.geniusweb import AGENTS
from environment.agents.geniusweb.wrapper import geniusweb_wrapper, DummyReporter
from environment.deadline import Deadline
from environment.scenario import Scenario


@pytest.mark.parametrize("agent_class", AGENTS.values())
def test_initialisation(agent_class):
    scenario = Scenario.create_random("test", 400, default_rng())
    agent: DefaultParty = agent_class("test", scenario.utility_function_A, Deadline(10000), {})


# @pytest.mark.parametrize("agent_class", AGENTS.values())
# def test_wrapper(agent_class):
#     scenario = Scenario.create_random("test", 400)
#     agent_class = geniusweb_wrapper(agent_class)
#     agent = agent_class("test", scenario.preferences_A, Deadline(10000), {})
#     agent.finish()


@pytest.mark.parametrize("agent_class", AGENTS.values())
def test_opening_bid(agent_class):
    scenario = Scenario.create_random("test", 400, default_rng())
    agent: DefaultParty = agent_class("test", scenario.utility_function_A, Deadline(10000), {})
    action: ActionWithBid = agent.select_action()
    assert isinstance(action, Action)
    action._actor = PartyId("test_opponent")
    accept = cast(Accept, action)
    agent.communicate_action(accept)
    agent.finish(Agreements({}))
