from enum import Enum
from dataclasses import dataclass


class NodeState(Enum):
    FILE = 1
    DIR_OPEN = 2
    DIR_CLOSED = 3


@dataclass
class Config:
    indent_width: int
    node_state_symbols: dict[NodeState, str]
    node_state_symbol_width: int


def default_config() -> Config:
    return Config(
        indent_width = 4,

        node_state_symbols = {
            NodeState.FILE:       "|",
            NodeState.DIR_OPEN:   "-",
            NodeState.DIR_CLOSED: "+",
        },

        # All node_state_symbols must consist of the same number of glyphs
        node_state_symbol_width = 1,
    )
