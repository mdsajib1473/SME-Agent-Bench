"""Architecture registry: select an agent by name."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.plan_execute import PlanExecuteAgent
from agents.react import ReActAgent
from agents.supervisor import SupervisorAgent

ARCHITECTURES = {
    "react": ReActAgent,
    "plan_execute": PlanExecuteAgent,
    "supervisor": SupervisorAgent,
}


def build_agent(name):
    try:
        return ARCHITECTURES[name]()
    except KeyError:
        raise ValueError(
            f"unknown architecture {name!r}; choose one of {', '.join(ARCHITECTURES)}"
        ) from None
