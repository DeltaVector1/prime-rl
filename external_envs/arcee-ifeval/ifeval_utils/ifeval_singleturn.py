from typing import Literal

import verifiers as vf


class IFEvalSingleTurnEnv(vf.SingleTurnEnv):
    """
    Compatibility wrapper around the current verifiers single-turn env.
    """

    def __init__(self, message_type: Literal["chat", "completion"] = "chat", **kwargs):
        super().__init__(message_type=message_type, **kwargs)
        self.message_type = message_type
