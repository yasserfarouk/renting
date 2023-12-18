from typing import Any
from environment.agents.geniusweb.wrapper import geniusweb_wrapper

from .ANL2022 import AGENTS as ANL2022_AGENTS
from .ANL2023 import AGENTS as ANL2023_AGENTS
from .basic import AGENTS as BASIC_AGENTS
from .CSE3210 import AGENTS as CSE3210_AGENTS

AGENTS: dict[str, Any] = {}

AGENTS.update({f"ANL2022.{k}": geniusweb_wrapper(v) for k, v in ANL2022_AGENTS.items()})
AGENTS.update({f"ANL2023.{k}": geniusweb_wrapper(v) for k, v in ANL2023_AGENTS.items()})
AGENTS.update({f"BASIC.{k}": geniusweb_wrapper(v) for k, v in BASIC_AGENTS.items()})
AGENTS.update({f"CSE3210.{k}": geniusweb_wrapper(v) for k, v in CSE3210_AGENTS.items()})
