from pathlib import Path

from dataclasses import (
    dataclass,
    field
)
from typing import (
    Optional,
    Callable,
    TypeVar
)

from config import Config, NodeState
from dir_tree import (
    DirTree,
    DirView,
    DirNode,
    NodeKind,
    LinkStatus,
    NodeID,
    NodeDetails
)


# Parser
#===============================================================================
@dataclass
class ParseState:
    text: str
    pos: int = 0


    def char(self) -> Optional[str]:
        return None if self.done() else self.text[self.pos]

    def check_char(self, f: Callable[[str], bool]) -> bool:
        return False if self.done() else f(self.text[self.pos])

    def tail(self) -> str:
        return self.text[self.pos:] # TODO This creates a copy

    def done(self) -> bool:
        return self.pos == len(self.text)

    def advance(self, offset: int = 1):
        self.pos = min(self.pos + offset, len(self.text))


def skip_whitespace(parser: ParseState):
    while parser.check_char(lambda ch: ch in [" ", "\t"]):
        parser.advance()


def try_parse_int(parser: ParseState) -> Optional[int]:
    start = parser.pos
    while not parser.done() and is_digit(unwrap(parser.char())):
        parser.advance()

    if parser.pos == start:
        return None

    return int(parser.text[start:parser.pos])


def parse_int(parser: ParseState, error_context: str) -> int:
    val = try_parse_int(parser)
    if val is None:
        raise Exception("Error: expected " + error_context)

    return val


def try_consume(parser: ParseState, s: str) -> bool:
    if parser.tail().startswith(s):
        parser.advance(len(s))
        return True

    return False


def consume(parser: ParseState, s: str):
    if not parser.tail().startswith(s):
        raise Exception("Parsing failed, expected: " + s)

    parser.advance(len(s))

def consume_n(parser: ParseState, n: int) -> str:
    result = ""
    while n > 0:
        if parser.done():
            raise Exception("Parsing failed: unexpected end")

        result += unwrap(parser.char())
        parser.advance()
        n -= 1

    return result


def consume_until(parser: ParseState, chars: list[str]) -> str:
    start = parser.pos
    while not parser.done() and unwrap(parser.char()) not in chars:
        parser.advance()

    if parser.pos == start:
        raise Exception("Expected something: " + parser.text[start:])

    return parser.text[start:parser.pos]


def is_digit(s: str) -> bool:
    for c in s:
        n = ord(c)
        if n < 48 or n > 57:
            return False

    return True


T = TypeVar("T")
def unwrap(v: Optional[T]) -> T:
    if v is None:
        raise Exception("unwrapping None")

    return v


# Parsing directory trees
#===============================================================================
@dataclass
class Span:
    start: int
    end: int # exclusive

def empty_span():
    return Span(0, 0)

@dataclass
class LineSegments:
    details: Optional[Span] = None
    node_state: Optional[Span] = None
    name: Span = field(default_factory=empty_span)


@dataclass
class DecoratedNodeID:
    id: NodeID
    is_executable: bool


@dataclass
class NewNode:
    name: str
    kind: NodeKind


def compute_indent(config: Config, segments: LineSegments, indent_offset: int) -> int:
    first_significant_char = 0
    if segments.node_state is None:
        # Even if there is no node state we calculate the indentation as if there was
        first_significant_char = segments.name.start - config.node_state_symbol_width - 1
    else:
        first_significant_char = segments.node_state.start

    indent = max(first_significant_char - indent_offset, 0) // config.indent_width
    return indent


def parse_tree(config: Config, root_dir: Path, lines: list[str], indent_offset: int) -> DirTree:
    tree = DirTree(-1, root_dir)
    parent_stack: list[DirNode] = [tree.root]
    prev_indent = 0
    prev_node: Optional[DirNode] = None
    for line in lines:
        if not line.strip():
            continue

        parsed_node, segments = parse_line(config, line)
        indent = compute_indent(config, segments, indent_offset)

        # Update parent stack if indention has changed
        if indent < prev_indent:
            for i in range(prev_indent - indent):
                parent_stack.pop()
            prev_indent = indent
        elif indent > prev_indent:
            if not prev_node:
                raise Exception("First line cannot be indented")
            if not prev_node.is_dir():
                raise Exception("Parent is not a directory (indent = " + str(indent))
            parent_stack.append(prev_node)
            prev_indent += 1

        parent = parent_stack[-1]
        match parsed_node:
            case DirNode() as node:
                node.parent = parent
                parent.children.append(node)
                prev_node = node

            case NewNode() as new_node:
                node = tree.lookup_or_create(parent.filepath() / new_node.name, new_node.kind)
                prev_node = node

    return tree


# A single line consists of the following fields:
#
# 1. Node details: "[drwxr-xr-x user group size mtime]"
# 2. Node state:   "|" (file node) or
#                  "+" (collapsed directory node) or
#                  "-" (expanded directory node)
# 3. Node ID:      "X:Y_" where X denotes the generation ID, Y the actual node ID, and _ is an optional flag
# 4. Node name:    "filename"
# 5. Link target   " -> link_target_name"
#
# During parsing, we distinguish between two kind of nodes:
# - An *existing* node is a node that has an ID and exists in the DirTree.
# - A *new* node does not exist in the DirTree
#
# For existing nodes, we require that fields (2) - (4) can be parsed
# successfully (fields (1) and (5) are optional). For new nodes, we require
# that *only* (4), i.e. the node name, is present.
def parse_line(config: Config, line: str) -> tuple[DirNode|NewNode, LineSegments]:
    parser = ParseState(line)
    segments = LineSegments()

    details = try_parse_details(parser, segments)
    node_state = try_parse_node_state(config, parser, segments)

    # If neither the node details or the node state are present we consider
    # this to be a new node
    is_new_node = details is None and node_state is None
    if is_new_node:
        node_kind, node_name = parse_node_name(parser, segments)

        # Don't allow any junk at the end
        junk_start = parser.pos
        skip_whitespace(parser)
        if not parser.done():
            raise Exception("Unexpected chars after filename: " + parser.text[junk_start:])

        return NewNode(name = node_name, kind = node_kind), segments


    # If we get here we are parsing an existing node
    node_id = parse_node_id(parser)
    node_kind, node_name = parse_node_name(parser, segments)
    if (node_kind == NodeKind.FILE) != (node_state == NodeState.FILE):
        raise Exception("Folder states only applicable to folders")

    link_status, link_target = parse_link_status(parser)
    node = DirNode(
        id = node_id.id,
        name = node_name,
        kind = node_kind,
        is_executable = node_id.is_executable,
        link_target = link_target,
        is_expanded = node_state == NodeState.DIR_OPEN,
        link_status = link_status,
        details = details,
    )

    return node, segments


def try_parse_details(parser: ParseState, segments: LineSegments) -> Optional[NodeDetails]:
    if not try_consume(parser, "["):
        return None

    start_pos = parser.pos
    mode = consume_until(parser, [" "]).strip()

    skip_whitespace(parser)
    user = consume_until(parser, [" "]).strip()

    skip_whitespace(parser)
    group = consume_until(parser, [" "]).strip()

    skip_whitespace(parser)
    size = consume_until(parser, [" "]).strip()

    skip_whitespace(parser)
    mtime = consume_until(parser, ["]"]).strip() # TODO Proper datetime parsing

    skip_whitespace(parser)
    consume(parser, "]")

    segments.details = Span(start_pos, parser.pos)

    return NodeDetails(
        user = user,
        group = group,
        mode = mode,
        size = size,
        mtime = mtime,
    )


def try_parse_node_state(config: Config, parser: ParseState, segments: LineSegments) -> Optional[NodeState]:
    skip_whitespace(parser)
    start_pos = parser.pos

    for node_state, node_state_sym in config.node_state_symbols.items():
        if try_consume(parser, node_state_sym):
            segments.node_state = Span(start_pos, parser.pos)
            return node_state

    return None


NODE_FLAGS_LEN = 1
def parse_node_id(parser: ParseState) -> DecoratedNodeID:
    gen_id = parse_int(parser, "generation ID")
    consume(parser, ":")
    node_id = parse_int(parser, "node ID")

    is_executable = False
    flags = consume_n(parser, NODE_FLAGS_LEN)
    if flags == "x":
        is_executable = True
    elif flags != "_":
        raise Exception("Invalid node flag: " + flags)

    consume(parser, " ")

    return DecoratedNodeID(
        id = (gen_id, node_id),
        is_executable = is_executable,
    )


def parse_node_name(parser: ParseState, segments: Optional[LineSegments]) -> tuple[NodeKind, str]:
    skip_whitespace(parser)

    if parser.char() in ["'", '"']:
        raise Exception("TODO: quoted names")

    else:
        name_start = parser.pos
        while not parser.done() and not parser.tail().startswith("->"):
            parser.advance()

        name_end = parser.pos
        if name_start == name_end:
            raise Exception("Expected node name")

        if segments is not None:
            segments.name = Span(name_start, name_end)

        name = parser.text[name_start:name_end].strip()
        kind = NodeKind.FILE
        if name.endswith("/"):
            if len(name) > 1:
                name = name[:-1]

            kind = NodeKind.DIRECTORY

        return (kind, name)


def parse_link_status(parser: ParseState) -> tuple[LinkStatus, Optional[Path]]:
    link_target = None
    link_status = LinkStatus.NO_LINK
    skip_whitespace(parser)
    if try_consume(parser, "->"):
        if try_consume(parser, "!"):
            link_status = LinkStatus.BROKEN
        elif try_consume(parser, "?"):
            link_status = LinkStatus.UNKNOWN
        else:
            link_status = LinkStatus.GOOD

        link_target = Path(parse_node_name(parser, None)[1])

    return link_status, link_target
