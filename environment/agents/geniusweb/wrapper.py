import json
import shutil
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.ActionWithBid import ActionWithBid
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Agreements import Agreements
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValue import DiscreteValue
from geniusweb.profile.Profile import Profile
from geniusweb.progress.ProgressRounds import ProgressRounds
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from geniusweb.references.ProfileRef import ProfileRef
from geniusweb.references.ProtocolRef import ProtocolRef
from pyson.ObjectMapper import ObjectMapper
from tudelft_utilities_logging.Reporter import Reporter
from uri.uri import URI

from environment.scenario import UtilityFunction


def geniusweb_wrapper(base):
    class GeniusWebAgent(base): #TODO: set to base
        def __init__(self, agent_id: str, utility_function, deadline, parameters: dict = {}) -> None:
            super().__init__(DummyReporter())
            self.agent_id = agent_id
            self.action = None

            self.connection = DummyConnection()


            self.tmp_dir = Path(tempfile.gettempdir()) / "geniusweb" / str(uuid4())
            self.tmp_dir.mkdir(parents=True)
            self.tmp_profile_fn = self.tmp_dir / "UtilityFunction.json"

            parameters["storage_dir"] = str(self.tmp_dir)

            profile = convert_utility_to_geniusweb(utility_function)
            with self.tmp_profile_fn.open("w") as f:
                f.write(json.dumps(profile))
            profile_uri = URI(f"file:{self.tmp_profile_fn}")
            profile_ref = ProfileRef(profile_uri)

            #TODO: make sure that all agents can handle round based deadlines
            if deadline.rounds:
                progress = ProgressRounds(deadline.rounds, 0, datetime.fromtimestamp((deadline.start_time_ms + deadline.ms) / 1000))
            else:
                progress = ProgressTime(deadline.ms, datetime.fromtimestamp(deadline.start_time_ms / 1000))

            protocol = ProtocolRef(URI("SAOP"))

            self.ID = PartyId(agent_id)

            settings = Settings(
                self.ID,
                profile_ref,
                protocol,
                progress,
                Parameters(parameters),
            )

            self.notifyChange(settings)

            if not hasattr(self, "profile"):
                self.profile = ObjectMapper().parse(profile, Profile)

        def send_action(self, action: Action):
            self.action = action

        def getConnection(self):
            return self.connection

        def select_action(self, last_actions: deque[dict]) -> dict:
            for prev_action in last_actions:
                if prev_action["agent_id"] != self.agent_id:
                    self.communicate_action(prev_action)
            
            self.notifyChange(YourTurn())
            if self.connection.action:
                action = self.connection.action
                self.connection.reset()
            elif self.action:
                action = self.action
                self.action = None
            else:
                raise ValueError(f"Action cannot be None, agent_id: {type(self).__name__}")

            self.notifyChange(ActionDone(action))

            return self._geniusweb_action_to_dict_action(action)

        def communicate_action(self, action):
            action = self._dict_action_to_geniusweb_action(action)
            self.notifyChange(ActionDone(action))

        def final(self, last_actions: deque[dict]):
            last_action = last_actions[-1]
            if last_action["agent_id"] != self.agent_id:
                self.communicate_action(last_action)
            if last_action["accept"] == 1:
                bid = self._dict_action_to_geniusweb_action(last_action).getBid()
                agreements = Agreements({PartyId(action["agent_id"]): bid for action in last_actions})
            else:
                agreements = Agreements({})

            self.notifyChange(Finished(agreements))
            if self.tmp_dir.exists():
                shutil.rmtree(self.tmp_dir)

        def _geniusweb_action_to_dict_action(self, action: Action) -> list:
            if isinstance(action, Offer):
                action_dict = {"accept": np.int32(0)}
            elif isinstance(action, Accept):
                action_dict = {"accept": np.int32(1)}
            else:
                raise ValueError(f"Action {action} not supported")
            
            issue_values = action._bid._issuevalues
            offer = [int(issue_values[str(i)]._value) for i in range(len(issue_values))]
            # bid_dict = {int(i): int(v._value) for i, v in action._bid._issuevalues.items()}
            action_dict["offer"] = np.array(offer, dtype=np.int32)
            action_dict["agent_id"] = self.agent_id

            return action_dict

        def _dict_action_to_geniusweb_action(self, action: dict) -> ActionWithBid:
            bid = Bid({str(i):  DiscreteValue(str(v)) for i, v in enumerate(action["offer"])})
            if action["accept"] == 0:
                action_GW = Offer(PartyId(action["agent_id"]), bid)
            elif action["accept"] == 1:
                action_GW = Accept(PartyId(action["agent_id"]), bid)
            else:
                raise ValueError(f"Action {action} not supported")
            
            return action_GW
        
    GeniusWebAgent.__name__ = base.__name__

    return GeniusWebAgent

class DummyConnection:
    def __init__(self) -> None:
        self.action = None

    def send(self, action):
        self.action = action

    def reset(self):
        self.action = None


class DummyReporter(Reporter):
    def log(self, *args, **kwargs):
        pass


def convert_utility_to_geniusweb(utility_function: UtilityFunction):
    issue_weights = {str(iss): w for iss, w in utility_function.objective_weights.items()}
    value_weights = {str(iss): {str(v): w for v, w in values.items()} for iss, values in utility_function.value_weights.items()}

    issuesValues = {issue: {"values": list(values.keys())} for issue, values in value_weights.items()}
    domain = {"name": "tmp_domain", "issuesValues": issuesValues}

    issue_utilities = {
        i: {"DiscreteValueSetUtilities": {"valueUtilities": v}} for i, v in value_weights.items()
    }
    profile = {
        "LinearAdditiveUtilitySpace": {
            "issueUtilities": issue_utilities,
            "issueWeights": issue_weights,
            "domain": domain,
            "name": "tmp_profile",
        }
    }

    return profile
