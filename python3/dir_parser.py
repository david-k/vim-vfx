from pathlib import Path

from dataclasses import (
    dataclass,
    field
)
from typing import (
    Optional,
    Callable,
    TypeVar,
    Any
)

from config import Config, NodeState
from dir_tree import *
import log


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
    start: int = 0
    end: int = 0 # exclusive


@dataclass
class LineSegments:
    details: Optional[Span] = None
    node_state: Optional[Span] = None
    name: Span = field(default_factory=Span)


# Describes the offset (in spaces) from the start of a line to the first significant character on
# that line.
#
# If base_offset is not None, then the first significant character is at position base_offset + entry_offset.
# Otherwise,
#
# What is the difference between `base_offset == None` and `base_offset == 0`? Consider the
# following example:
#
#    ·················································new_dir/
#    ·····················································new_sub
#    [drwxr-xr-x david david 4.0K 2025-01-28 09:25]·····+1_ sub/
#
# Here, · denotes a space that is relevant for computing the indentation of an entry, and we assume
# a indent_width of 4. Parsing starts at the top.
# - Since the first line has no details, we set base_offset=None and entry_offset=47 (see
#   EntryOffset.from_segments())
# - For the second line, we set base_offset=None and entry_offset=51.
#   Computing the diff between these to EntryOffset gives 4, which means the indention level of
#   new_sub is higher (by one) than new_dir
# - For the third line we set base_offset=46 and entry_offset=4. Because the previous line has no
#   base_offset, we need to rebase it before we can compute the diff between them.

@dataclass
class EntryOffset:
    base_offset: Optional[int] = None
    entry_offset: int = 0

    @staticmethod
    def from_segments(segments: LineSegments, config: Config) -> "EntryOffset":
        offset = EntryOffset()

        if segments.node_state is None:
            offset.entry_offset = segments.name.start
        else:
            # Even if there is a node state we calculate the indentation as if there wasn't
            offset.entry_offset = segments.node_state.start + config.node_state_symbol_width + 1

        if segments.details is not None:
            offset.rebase(segments.details.end + NUM_SPACES_IGNORED_AFTER_DETAILS)

        return offset


    def diff(self, other: "EntryOffset") -> int:
        return self._abs_offset(other.base_offset) - other._abs_offset(self.base_offset)


    def rebase(self, new_base_offset: int):
        assert self.base_offset is None # Ignore this case as I don't need it
        self.base_offset = new_base_offset
        self.entry_offset = max(self.entry_offset - self.base_offset, 0)


    def _abs_offset(self, other_base_offset: Optional[int]) -> int:
        if self.base_offset is not None:
            return self.entry_offset

        if other_base_offset is not None:
            return max(self.entry_offset - other_base_offset, 0)

        return self.entry_offset


@dataclass
class DecoratedNodeID:
    id: NodeIDKnown
    is_executable: bool


@dataclass
class NewNode:
    name: str
    kind: NodeKind


def parse_buffer(config: Config, lines: list[str]) -> DirTree:
    with log.scope("Parsing buffer"):
        assert NUM_LINES_BEFORE_TREE == 1
        options = parse_options(lines[0])
        tree = parse_tree(config,
            options["path"],
            options["show_details"],
            options["show_dotfiles"],
            lines[1:],
        )
        tree.root.id = NodeIDKnown(options["root_id"])
        return tree


def parse_options(line: str):
    [options_str, root_path] = line.split("|", maxsplit=1)

    if not options_str.startswith(">"):
        raise Exception("Options line must start with '>'")

    options: dict[str, Any] = {
        "path": Path(root_path.strip()),
    }
    for option in options_str[1:].split():
        [name, value] = option.split("=")
        options[name] = parse_option_value(value)

    return options


def parse_option_value(value: str) -> bool|int:
    if value == "True":
        return True

    if value == "False":
        return False

    if value.isdecimal():
        return int(value)

    raise Exception("Invalid option value: " + value)



def parse_tree(
    config: Config,
    root_dir: Path,
    has_details: bool,
    has_dotfiles: bool,
    lines: list[str],
) -> DirTree:
    tree = DirTree(root_dir, has_details, has_dotfiles)
    tree.root.is_expanded = True
    parent_stack: list[DirNode] = [tree.root]
    prev_node: Optional[DirNode] = None
    prev_offset: Optional[EntryOffset] = None
    for line_idx, line in enumerate(lines):
        if not line.strip():
            continue

        errors: list[str] = []
        parsed_node, segments = parse_line(config, line)
        offset = EntryOffset.from_segments(segments, config)

        # Handle changes in indentation
        if prev_offset is None:
            prev_offset = offset
        else:
            # Compute indent of the current line relative to the previous line
            if prev_offset.base_offset is None and offset.base_offset is not None:
                prev_offset.rebase(offset.base_offset)

            relative_indent = int(offset.diff(prev_offset) / config.indent_width)

            # Update parent stack if indention has changed
            if relative_indent < 0:
                for i in range(-relative_indent):
                    parent_stack.pop()
                prev_offset.entry_offset = offset.entry_offset
            elif relative_indent > 0:
                if not prev_node:
                    errors.append("First line cannot be indented")
                elif not prev_node.is_dir():
                    errors.append("Parent is not a directory")
                else:
                    parent_stack.append(prev_node)
                    prev_offset.entry_offset += config.indent_width

        parent = parent_stack[-1]
        match parsed_node:
            case DirNode() as node:
                if node.details is None and has_details:
                    errors.append("Nodes are expected to have details")

                node.parent = parent
                node.line_no = line_idx + NUM_LINES_BEFORE_TREE + 1
                for error in errors:
                    node.add_error(error)
                parent.children.append(node)
                prev_node = node

            case NewNode() as new_node:
                node = tree.lookup_or_create_new(parent.filepath() / new_node.name, new_node.kind)
                for error in errors:
                    node.add_error(error)
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
    seq_no = parse_int(parser, "node sequence no")

    is_executable = False
    flags = consume_n(parser, NODE_FLAGS_LEN)
    if flags == "x":
        is_executable = True
    elif flags != "_":
        raise Exception("Invalid node flag: " + flags)

    consume(parser, " ")

    return DecoratedNodeID(
        id = NodeIDKnown(seq_no),
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

        skip_whitespace(parser)
        if not parser.done():
            link_target = Path(parse_node_name(parser, None)[1])

    return link_status, link_target
