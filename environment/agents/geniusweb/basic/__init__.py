from geniusweb.party.DefaultParty import DefaultParty

from .boulware_agent.boulware_agent import BoulwareAgent
from .conceder_agent.conceder_agent import ConcederAgent
from .hardliner_agent.hardliner_agent import HardlinerAgent
from .linear_agent.linear_agent import LinearAgent
from .random_agent.random_agent import RandomAgent
from .stupid_agent.stupid_agent import StupidAgent

AGENTS: dict[str, DefaultParty] = {
    "BoulwareAgent": BoulwareAgent,
    "ConcederAgent": ConcederAgent,
    "HardlinerAgent": HardlinerAgent,  # NOTE: disabled because of useless behaviour
    "LinearAgent": LinearAgent,
    "RandomAgent": RandomAgent,
    "StupidAgent": StupidAgent,  # NOTE: disabled because test agent
}
