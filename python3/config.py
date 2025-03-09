from enum import Enum
from dataclasses import dataclass
from logger import LogLevel


class NodeState(Enum):
    FILE = 1
    DIR_OPEN = 2
    DIR_CLOSED = 3


@dataclass
class Config:
    confirm_changes: bool
    indent_width: int
    node_state_symbols: dict[NodeState, str]
    node_state_symbol_width: int
    enable_log: bool
    log_level: LogLevel


def default_config() -> Config:
    return Config(
        confirm_changes = False,
        indent_width = 4,

        node_state_symbols = {
            NodeState.FILE:       "|",
            NodeState.DIR_OPEN:   "-",
            NodeState.DIR_CLOSED: "+",
        },

        # All node_state_symbols must consist of the same number of glyphs
        node_state_symbol_width = 1,

        enable_log = False,
        log_level = LogLevel.INFO,
    )
