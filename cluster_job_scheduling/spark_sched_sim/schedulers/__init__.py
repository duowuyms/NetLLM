from copy import deepcopy

from .neural import *
from .heuristic import *


def make_scheduler(agent_cfg):
    glob = globals()
    agent_cls = agent_cfg['agent_cls']
    assert agent_cls in glob, \
        f"'{agent_cls}' is not a valid scheduler."
    return glob[agent_cls](**deepcopy(agent_cfg))