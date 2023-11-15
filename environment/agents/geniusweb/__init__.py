from typing import Any
from environment.agents.geniusweb.wrapper import geniusweb_wrapper

from .ANL2022 import AGENTS as ANL2022
from .ANL2023 import AGENTS as ANL2023
from .basic import AGENTS as BASIC
from .CSE3210 import AGENTS as CSE3210

AGENTS: dict[str, Any] = {}

AGENTS.update({f"ANL2022.{k}": geniusweb_wrapper(v) for k, v in ANL2022.items()})
AGENTS.update({f"ANL2023.{k}": geniusweb_wrapper(v) for k, v in ANL2023.items()})
AGENTS.update({f"BASIC.{k}": geniusweb_wrapper(v) for k, v in BASIC.items()})
AGENTS.update({f"CSE3210.{k}": geniusweb_wrapper(v) for k, v in CSE3210.items()})
