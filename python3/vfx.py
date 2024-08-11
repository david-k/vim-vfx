import vim # type: ignore
import io
import re
import subprocess
from pathlib import Path

from enum import Enum
from dataclasses import dataclass, field
from typing import NewType, Optional, Callable, TextIO, TypeVar

from dir_tree import DirTree, DirView, DirNode, EntryKind, LinkStatus, UniqueEntryID, NodeDetails

from pprint import pprint


# Globals
#===============================================================================
BufNo = NewType("BufNo", int)

@dataclass
class Session:
    buf_no: BufNo
    alternate_buf: BufNo
    view: DirView
    indent_offset: int = 0


@dataclass
class GlobalTreeState:
    root_dir: Optional[Path] = None
    expanded_dirs: set[Path] = field(default_factory=set)
    show_dotfiles: bool = False
    show_details: bool = True
    cursor_line: int = 1


GLOBAL_STATE = GlobalTreeState()
LAST_GEN_ID = 0
SESSIONS: dict[int, Session] = {}

# Number of spaces used for indentation. TODO Read from 'softtabstop' or 'shiftwidth'?
INDENT_WIDTH = 4


# Utils
#===============================================================================
def next_gen_id() -> int:
    global LAST_GEN_ID

    LAST_GEN_ID += 1
    return LAST_GEN_ID


def get_session() -> Optional[Session]:
    buf_no = BufNo(vim.current.buffer.number)
    return SESSIONS.get(buf_no)


# Should be called right before the buffer is closed
def on_exit(s: Session):
    global GLOBAL_STATE

    GLOBAL_STATE.root_dir = s.view.root_dir()
    GLOBAL_STATE.expanded_dirs = s.view.expanded_dirs
    GLOBAL_STATE.show_dotfiles = s.view.show_dotfiles
    GLOBAL_STATE.show_details = s.view.show_details
    GLOBAL_STATE.cursor_line = vim.eval('line(".")')

    del SESSIONS[s.buf_no]


def restore_alternate_buf(s: Session):
    if int(vim.eval(f"buflisted({s.alternate_buf})")):
        vim.command(f"let @# = {s.alternate_buf}")


def default_app_for(filename: Path) -> str:
    mime = subprocess.run(
        ["xdg-mime", "query", "filetype", filename],
        capture_output=True,
        check=True
    ).stdout.decode().strip()

    default_app = subprocess.run(
        ["xdg-mime", "query", "default", mime],
        capture_output=True,
        check=True
    ).stdout.decode().strip()

    return default_app


def should_open_in_vim(default_app: str) -> bool:
    return default_app in ["", "vim.desktop", "gvim.desktop"]


def is_dir_empty(path: Path) -> bool:
    for c in path.iterdir():
        return False

    return True


# Vim helpers
#-----------------------------------------------------------
def escape_for_vim_expr(text: str):
    return "'" + text.replace("'", "''") + "'"

def escape_for_vim_command(text: str):
    return re.sub(r'(\n| |\t|\r|\\)', r'\\\1', text)


def vim_set_buffer_contents(lines: list[str]):
    line_no = vim.eval('line(".")')
    buf = vim.current.buffer
    buf[:] = lines
    vim.command(f'normal {line_no}G')


def vim_get_line(line_no) -> str:
    return vim.eval(f'getline({line_no})')


def vim_get_line_no() -> int:
    return int(vim.eval('line(".")'))


def is_buf_name_available(buf_name: str) -> bool:
    return int(vim.eval(f'bufexists( {escape_for_vim_expr(buf_name)} )')) == 0


def make_buf_name(path: str) -> str:
    def add_counter_to_name(name: str, counter: int) -> str:
        if counter == 0:
            return name
        return f"{name} [{counter}]"

    counter = 0
    while True:
        buf_name = add_counter_to_name("[Directory] " + path, counter)
        if is_buf_name_available(buf_name):
            return buf_name

        counter += 1


def rename_buffer(new_name: str):
    # `:file NEW_NAME` can be used to rename a buffer.
    # However, this creates a new unlisted buffer with the old name and sets the
    # alternate file to that buffer. Thus, we need to do some extra work to make
    # it appear as if this does not happen.

    alt_buf = vim.eval('bufnr("#")')

    vim.command('file ' + escape_for_vim_command(new_name))
    prev_name_buf = vim.eval('bufnr("#")')
    vim.command('bwipeout ' + prev_name_buf)

    vim.command(f"let @# = {alt_buf}")


def is_buf_modified() -> bool:
    return bool(int(vim.eval("&modified")))


# Working with filepaths in the buffer
#-----------------------------------------------------------
def get_path_at(line_no: int, indent_offset: int) -> Path:
    parts = [parse_line(vim_get_line(line_no)).name]
    while line_no := go_to_parent(line_no, indent_offset):
        parts.append(parse_line(vim_get_line(line_no)).name)

    head, *tail = reversed(parts)
    return Path(head).joinpath(*tail)


def go_to_parent(line_no: int, indent_offset: int) -> int:
    cur_indent = get_indent_at(line_no, indent_offset)
    for cur_line_no in range(line_no - 1, 0, -1):
        if get_indent_at(cur_line_no, indent_offset) < cur_indent:
            return cur_line_no

    return 0


def get_indent_at(line_no: int, indent_offset: int) -> int:
    return get_indent(vim_get_line(line_no), indent_offset)


def get_indent(line: str, indent_offset: int) -> int:
    if not line:
        return 0

    idx = 0
    indent = 0
    if line[0] == "[":
        while idx < len(line) and line[idx] != "]":
            idx += 1
        idx += 2
    else:
        indent = -indent_offset

    while idx < len(line):
        if line[idx] == " ":
            indent += 1
        else:
            break

        idx += 1

    if idx < len(line) and line[idx] not in ["|", "+", "-"]:
        indent -= 2

    return max(indent, 0) // INDENT_WIDTH


def parse_line(line: str, parent: Optional[DirNode] = None) -> DirNode:
    opt_entry = try_parse_line(line, parent)
    if not opt_entry:
        raise Exception("Parsing entry failed")

    return opt_entry


# Parser
#-----------------------------------------------------------
@dataclass
class ParseState:
    text: str
    pos: int = 0

    # TODO Reduce copying slices

    def char(self) -> Optional[str]:
        return None if self.done() else self.text[self.pos]

    def check_char(self, f: Callable[[str], bool]) -> bool:
        return False if self.done() else f(self.text[self.pos])

    def tail(self) -> str:
        return self.text[self.pos:]

    def done(self) -> bool:
        return self.pos == len(self.text)

    def advance(self, offset: int = 1):
        self.pos = min(self.pos + offset, len(self.text))


def skip_whitespace(parser: ParseState):
    while parser.check_char(lambda ch: ch in [" ", "\t", "\r"]):
        parser.advance()


def try_parse_int(parser: ParseState) -> Optional[int]:
    start = parser.pos
    while not parser.done() and is_digit(unwrap(parser.char())):
        parser.advance()

    if parser.pos == start:
        return None

    return int(parser.text[start:parser.pos])


def try_consume(parser: ParseState, s: str) -> bool:
    if parser.tail().startswith(s):
        parser.advance(len(s))
        return True

    return False


def consume(parser: ParseState, s: str):
    if not parser.tail().startswith(s):
        raise Exception("Parsing failed, expected: " + s)

    parser.advance(len(s))


def parse_identifier(parser: ParseState) -> str:
    start = parser.pos
    while not parser.done() and unwrap(parser.char()).isalnum():
        parser.advance()

    if parser.pos == start:
        raise Exception("Expected identifier: " + parser.text[start:])

    return parser.text[start:parser.pos]


def parse_until(parser: ParseState, chars: list[str]) -> str:
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


# Line parsing
#-----------------------------------------------------------
# Parsing a single line:
# 1. parse details
# 2. parse indent
# 3. parse folder state
# 4. parse ID
# 5. parse name
# 6. parse link target

class NodeState(Enum):
    FILE = 1
    DIR_OPEN = 2
    DIR_CLOSED = 3


def try_parse_line(line: str, parent: Optional[DirNode] = None) -> Optional[DirNode]:
    parser = ParseState(line)

    details = try_parse_details(parser)

    skip_whitespace(parser)
    node_state = try_parse_node_state(parser)
    if node_state is None:
        return None

    unique_id_and_x = try_parse_id(parser)
    if unique_id_and_x is None:
        return None

    unique_id = unique_id_and_x[0:2]
    is_executable = unique_id_and_x[2]

    name_and_kind = try_parse_filename(parser)
    if name_and_kind is None:
        raise Exception("Expected filename")

    [kind, name] = name_and_kind

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

        link_target_and_kind = try_parse_filename(parser)
        if link_target_and_kind is None:
            raise Exception("Expected link target")

        link_target = Path(link_target_and_kind[1])

    if (kind == EntryKind.FILE) != (node_state == NodeState.FILE):
        raise Exception("Folder states only applicable to folders")

    node = DirNode(
        id = unique_id,
        name = name,
        kind = kind,
        is_executable = is_executable,
        link_target = link_target,
        parent = parent,
        is_expanded = node_state == NodeState.DIR_OPEN,
        link_status = link_status,
        details = details,
    )
    if parent:
        parent.children.append(node)

    return node


def try_parse_details(parser: ParseState) -> Optional[NodeDetails]:
    if not try_consume(parser, "["):
        return None

    mode = parse_until(parser, [" "]).strip()

    skip_whitespace(parser)
    user = parse_identifier(parser).strip()

    skip_whitespace(parser)
    group = parse_identifier(parser).strip()

    skip_whitespace(parser)
    size = parse_until(parser, [" "]).strip()

    skip_whitespace(parser)
    mtime = parse_until(parser, ["]"]).strip()

    skip_whitespace(parser)
    consume(parser, "]")

    return NodeDetails(
        user = user,
        group = group,
        mode = mode,
        size = size,
        mtime = mtime,
    )


def try_parse_node_state(parser: ParseState) -> Optional[NodeState]:
    if try_consume(parser, "|"):
        return NodeState.FILE

    if try_consume(parser, "+"):
        return NodeState.DIR_CLOSED

    if try_consume(parser, "-"):
        return NodeState.DIR_OPEN

    return None


def try_parse_id(parser: ParseState) -> Optional[tuple[int, int, bool]]:
    gen_id = try_parse_int(parser)
    if gen_id is None:
        return None

    if not try_consume(parser, ":"):
        return None

    node_id = try_parse_int(parser)
    if node_id is None:
        return None

    is_executable = False
    if try_consume(parser, "x"):
        is_executable = True

    consume(parser, " ")

    return (gen_id, node_id, is_executable)


def try_parse_filename(parser: ParseState) -> Optional[tuple[EntryKind, str]]:
    skip_whitespace(parser)

    if parser.char() in ["'", '"']:
        raise Exception("TODO: quoted names")

    else:
        name_start = parser.pos
        while not parser.done() and not parser.tail().startswith("->"):
            parser.advance()

        name_end = parser.pos
        if name_start == name_end:
            return None

        name = parser.text[name_start:name_end].strip()
        kind = EntryKind.FILE
        if name.endswith("/"):
            name = name[:-1]
            if not name:
                return None

            kind = EntryKind.DIRECTORY

        return (kind, name)



# DirView: printing
#-----------------------------------------------------------
NUM_DETAIL_COLUMNS = 5

def print_entries(entries: list[DirNode], max_id_width: int = 6):
    column_widths = [0] * (NUM_DETAIL_COLUMNS + 1)
    column_widths[-1] = max_id_width

    with io.StringIO() as buf:
        write_entries(buf, entries, column_widths)
        buf.seek(0, io.SEEK_SET)
        print(buf.getvalue())


def write_tree(out: TextIO, view: DirView):
    column_widths = [0] * (NUM_DETAIL_COLUMNS + 1)
    column_widths[-1] = len(str(view.tree.last_entry_id))

    details_width = 0
    root_nodes = view.tree.root.children
    if root_nodes and root_nodes[0].details is not None:
        compute_column_widths(root_nodes, column_widths)

        for i in range(NUM_DETAIL_COLUMNS):
            details_width += column_widths[i]
        details_width += (NUM_DETAIL_COLUMNS-1) # Spaces between columns
        details_width += 2 # Opening and closing brackets
        details_width += 1 # Space after closing bracket

    write_entries(out, view.tree.root.children, column_widths)

    return details_width


def write_entries(out: TextIO, entries: list[DirNode], column_widths: list[int], indent: int = 0):
    max_id_width = column_widths[-1]
    for entry in entries:
        if entry.details is not None:
            d = entry.details
            c1_width = column_widths[0]
            c2_width = column_widths[1]
            c3_width = column_widths[2]
            c4_width = column_widths[3]
            c5_width = column_widths[4]
            # ATTENTION: This must be kept in sync with the computation of `details_width` in `write_tree()`
            out.write(f"[{d.mode:<{c1_width}} {d.user:<{c2_width}} {d.group:<{c3_width}} {d.size:>{c4_width}} {d.mtime:<{c5_width}}] ")

        out.write(indent * INDENT_WIDTH * " ")

        node_state = "|"
        if entry.is_dir():
            if entry.is_expanded:
                node_state = "-"
            else:
                node_state = "+"

        out.write(f"{node_state}{entry.id[0]}:{entry.id[1]:0{max_id_width}}")
        if entry.is_executable:
            out.write("x")

        out.write(" " + escape_if_needed(str(entry.name)))
        if entry.kind == EntryKind.DIRECTORY:
            out.write("/")

        if entry.link_status != LinkStatus.NO_LINK:
            status = ""
            if entry.link_status == LinkStatus.BROKEN:
                status = "!"
            elif entry.link_status == LinkStatus.UNKNOWN:
                status = "?"

            out.write(f" ->{status}")
            if entry.link_target is not None:
                out.write(f" {entry.link_target}")

            if entry.kind == EntryKind.DIRECTORY:
                out.write("/")

        out.write("\n")

        if len(entry.children):
            write_entries(out, entry.children, column_widths, indent + 1)


def compute_column_widths(nodes: list[DirNode], widths: list[int]):
    for node in nodes:
        if node.details is None:
            return

        widths[0] = max(widths[0], len(node.details.mode))
        widths[1] = max(widths[1], len(node.details.user))
        widths[2] = max(widths[2], len(node.details.group))
        widths[3] = max(widths[3], len(node.details.size))

        if node.is_dir():
            compute_column_widths(node.children, widths)



def escape_if_needed(filename: str) -> str:
    if needs_escaping(filename):
        return escape_filename(filename)

    return filename


def needs_escaping(filename: str) -> bool:
    if filename.find("\n") != -1:
        return True

    return filename[0].isspace() or filename[-1].isspace()


def escape_filename(filename: str) -> str:
    escaped = filename.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
    return "'" + escaped + "'"


def unescape_filename(filename: str) -> str:
    if filename[0] == "'":
        if filename[-1] != "'":
            raise Exception("Missing closing quote in filename: " + filename)

        filename = filename[1:-1]
        filename = filename.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')

    return filename


# Parsing
#-----------------------------------------------------------
def compute_operations(view: DirView, lines: list[str], indent_offset: int) -> list[dict]:
    old_nodes_by_id = _nodes_by_id(view.tree.root)
    operations: list[dict] = []

    buf_tree, buf_nodes_by_id = parse_tree(view.root_dir(), lines, indent_offset)
    _compute_operations(view.root_dir(), old_nodes_by_id, buf_tree.root.children, operations)

    deleted_nodes = old_nodes_by_id.keys() - buf_nodes_by_id.keys()
    for entry_id in deleted_nodes:
        entry = old_nodes_by_id[entry_id]
        op_kind = "rm"
        if entry.is_dir():
            if is_dir_empty(view.root_dir() / entry.filepath()):
                op_kind = "rmdir"
            else:
                op_kind = "rm-rec"

        operations.append({
            "kind": op_kind,
            "filepath": view.root_dir() / entry.filepath()
        })

    return operations


def _compute_operations(
    root_dir: Path,
    old_nodes_by_id: dict[UniqueEntryID, DirNode],
    buf_nodes: list[DirNode],
    # Output:
    operations: list[dict]
):
    for buf_entry in buf_nodes:
        if buf_entry.id[0] == -1:
            # This is a new entry

            if buf_entry.link_target is not None:
                raise Exception("TODO Creating links is not supported yet")

            if buf_entry.is_file():
                operations.append({
                    "kind": "touch",
                    "filepath": root_dir / buf_entry.filepath(),
                })
            elif buf_entry.is_dir():
                operations.append({
                    "kind": "mkdir",
                    "filepath": root_dir / buf_entry.filepath(),
                })

        elif buf_entry.id in old_nodes_by_id:
            # This is an existing entry, possibly modified
            old_entry = old_nodes_by_id[buf_entry.id]
            if buf_entry.kind != old_entry.kind:
                raise Exception("TODO Changing the entry type is not supported yet")

            if buf_entry.link_target != old_entry.link_target:
                raise Exception("TODO Changing link target is not supported yet")

            if buf_entry.filepath() != old_entry.filepath():
                operations.append({
                    "kind": "mv",
                    "old_filepath": root_dir / old_entry.filepath(),
                    "new_filepath": root_dir / buf_entry.filepath(),
                })

        else:
            raise Exception("Invalid entry id: " + str(buf_entry.id))

        if buf_entry.is_dir():
            _compute_operations(root_dir, old_nodes_by_id, buf_entry.children, operations)


def parse_tree(root_dir: Path, lines: list[str], indent_offset: int) -> tuple[DirTree, dict[UniqueEntryID, DirNode]]:
    tree = DirTree(-1, root_dir)
    nodes_by_id: dict[UniqueEntryID, DirNode] = {}

    parent_stack: list[DirNode] = [tree.root]
    prev_indent = 0
    prev_node: Optional[DirNode] = None
    for line in lines:
        if not line.strip():
            continue

        # Update parent stack if indention has changed
        indent = get_indent(line, indent_offset)
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

        # Parse node
        parent = parent_stack[-1]
        node = try_parse_line(line, parent)
        if node:
            if node.id in nodes_by_id:
                raise Exception("Duplicate node id: " + str(node.id))
        else:
            name = unescape_filename(line.strip())
            kind = EntryKind.FILE
            if name.endswith("/"):
                kind = EntryKind.DIRECTORY

            node = tree.lookup_or_create(parent.filepath() / name, kind)

        nodes_by_id[node.id] = node
        prev_node = node

    return tree, nodes_by_id


def _nodes_by_id(node: DirNode) -> dict[UniqueEntryID, DirNode]:

    def _nodes_by_id_rec(
        node: DirNode,
        nodes_out: dict[UniqueEntryID, DirNode]
    ):
        for child in node.children:
            nodes_out[child.id] = child
            _nodes_by_id_rec(child, nodes_out)

    nodes_out: dict[UniqueEntryID, DirNode] = {}
    _nodes_by_id_rec(node, nodes_out)

    return nodes_out


# Externally called functions
#===============================================================================
# Initializes a new Vfx buffer
def init():
    global SESSIONS, GLOBAL_STATE

    if not GLOBAL_STATE.root_dir:
        GLOBAL_STATE.root_dir = Path.cwd()

    buf_name = make_buf_name(str(GLOBAL_STATE.root_dir))

    # The directory buffer should have no influence on the alternative file.
    # To this end, we need to update the alt file in two situations:
    # 1. When closing the directory buffer:
    #    - In this case we restore the state as if Vfx() was never called,
    #      i.e, we need to restore the alt file to the one before calling
    #      Vfx(). We do this with a autocmd on BufUnload.
    # 2. When opening a file:
    #    - In this case we want to set the alt file to the file that was
    #      active when calling Vfx(). We can achieve that by using :keepalt.
    #      (See VfxOpenEntry() below. Note that we need to disable the BufUnload
    #      autocmd so that it doesn't mess with the alt file)

    # We need to query the alt file *before* switching the buffer
    alt_buf = vim.eval('bufnr("#")')

    # Create new buffer and set options
    # Initially, I was using `bufadd()` instead of `:edit` but this leaves the
    # "[No Name]" buffer behind. `:edit` on the other hand automatically closes
    # that buffer if it is empty.
    vim.command("edit " + escape_for_vim_command(buf_name))
    vim.command('setlocal bufhidden=wipe')
    vim.command('setlocal buftype=acwrite')
    vim.command('setlocal noswapfile')
    vim.command('setlocal buflisted')
    vim.command('setlocal ft=vfx')

    vim.command("setlocal expandtab")

    buf_no = BufNo(vim.current.buffer.number)
    SESSIONS[buf_no] = Session(
        buf_no = buf_no,
        alternate_buf = alt_buf,
        view = DirView(
            next_gen_id(),
            GLOBAL_STATE.root_dir,
            GLOBAL_STATE.expanded_dirs,
            GLOBAL_STATE.show_dotfiles,
            GLOBAL_STATE.show_details,
        )
    )

    update_buffer()
    vim.command(f'normal {GLOBAL_STATE.cursor_line}G')


def quit():
    s = get_session(); assert s
    on_exit(s)

    # Since we have set bufhidden=wipe, quitting is done by simply switching
    # to another buffer (to the alternative buffer in this case)
    alt_buf_no = int(vim.eval('bufnr("#")'))
    if alt_buf_no != s.buf_no and int(vim.eval(f"buflisted({alt_buf_no})")):
        vim.command(f"buffer {alt_buf_no}")
    else:
        vim.command("enew")
        # This completely deletes the buffer once it is hidden (it won't even
        # be listed with `:ls!`)
        # This is needed to prevent the buffer list from being littered with
        # "[No Name]" buffers.
        vim.command("setlocal bufhidden=wipe")

    restore_alternate_buf(s)


def open_entry():
    if is_buf_modified():
        print("There are unsaved changes")
        return

    s = get_session(); assert s
    filepath = s.view.root_dir() / get_path_at(vim_get_line_no(), s.indent_offset)
    if filepath.is_dir():
        s.view.cd(filepath)
        update_buffer()

        new_name = make_buf_name(str(s.view.root_dir()))
        rename_buffer(new_name)
    else:
        default_app = default_app_for(filepath)
        if should_open_in_vim(default_app):
            on_exit(s)
            vim.command("keepalt edit " + escape_for_vim_command(str(filepath)))
        else:
            subprocess.Popen(["xdg-open", filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def toggle_expand():
    if is_buf_modified():
        print("There are unsaved changes")
        return

    s = get_session(); assert s
    if s.view.toggle_expand(get_path_at(vim_get_line_no(), s.indent_offset)):
        update_buffer()


def move_up():
    if is_buf_modified():
        print("There are unsaved changes")
        return

    s = get_session(); assert s
    if s.view.move_up():
        update_buffer()

        new_name = make_buf_name(str(s.view.root_dir()))
        rename_buffer(new_name)


def toggle_dotfiles():
    if is_buf_modified():
        print("There are unsaved changes")
        return

    s = get_session(); assert s
    s.view.show_dotfiles = not s.view.show_dotfiles
    s.view.refresh_from_fs()
    update_buffer()


def toggle_details():
    if is_buf_modified():
        print("There are unsaved changes")
        return

    s = get_session(); assert s
    s.view.show_details = not s.view.show_details
    s.view.refresh_from_fs()
    update_buffer()


def update_buffer():
    s = get_session(); assert s
    with io.StringIO() as buf:
        s.indent_offset = write_tree(buf, s.view)
        buf.seek(0, io.SEEK_SET)
        vim_set_buffer_contents(buf.readlines())

        if s.indent_offset:
            vim.command(f"setl varsofttabstop={s.indent_offset+2},{INDENT_WIDTH}")
            # There is also 'vartabstop' in case I want to support tabs for indentation
        else:
            vim.command(f"setl varsofttabstop=2,{INDENT_WIDTH}")

    vim.command("setl nomodified") # Don't mark the buffer as modified


# For internal use only. Called when the BufWriteCmd autocmd event is fired.
def on_buf_save():
    if is_buf_modified():
        s = get_session(); assert s

        operations = compute_operations(s.view, vim.current.buffer, s.indent_offset)

        if not operations:
            vim.command("setl nomodified") # Don't mark the buffer as modified
            return

        for op in operations:
            if op["kind"] in ["touch", "mkdir", "rm", "rm-rec", "rmdir"]:
                print(op["kind"] + " " + str(op["filepath"]))
            elif op["kind"] == "mv":
                print(op["kind"] + " " + str(op["old_filepath"]) + " -> " + str(op["new_filepath"]))
            else:
                raise Exception("Unsupported op: " + op["kind"])

        choice = int(vim.eval('confirm("Apply changes?", "yes\nno", 2)'))
        if choice == 1:
            for op in operations:
                if op["kind"] == "touch":
                    if op["filepath"].exists():
                        print("Error: destination already exists: " + str(op["filepath"]))
                        print("Aborting")
                        return
                    else:
                        op["filepath"].touch()

                elif op["kind"] == "mkdir":
                    op["filepath"].mkdir(parents=True)

                elif op["kind"] == "rmdir":
                    op["filepath"].rmdir()

                elif op["kind"] == "rm":
                    subprocess.run(["trash", "--", op["filepath"]])

                elif op["kind"] == "rm-rec":
                    subprocess.run(["trash", "--", op["filepath"]])

                elif op["kind"] == "mv":
                    if op["new_filepath"].exists():
                        print("Error: destination already exists: " + str(op["new_filepath"]))
                        print("Aborting")
                        return
                    else:
                        op["old_filepath"].rename(op["new_filepath"])

                else:
                    raise Exception("Unsupported op: " + op["kind"])

            s.view.refresh_from_fs()
            update_buffer()


# For internal use. Called when the BufUnload autocmd event is fired.
def on_buf_unload():
    s = get_session()
    # Check if we have already exited
    if not s:
        return

    on_exit(s)
    restore_alternate_buf(s)

# For internal use.
def get_indent_for_vim(line_no: int) -> int:
    s = get_session()

    if line_no <= 1:
        return 0

    prev_line = vim_get_line(line_no - 1)
    if not(prev_line):
        return 0

    idx = 0
    if prev_line[0] == "[":
        while idx < len(prev_line) and prev_line[idx] != "]":
            idx += 1
        idx += 1

    indent = idx
    tabstop = vim.eval("&tabstop")
    while idx < len(prev_line):
        if prev_line[idx] == " ":
            indent += 1
        elif prev_line[idx] == "\t":
            indent += tabstop
        else:
            break

        idx += 1

    if idx < len(prev_line) and prev_line[idx] in ["|", "+", "-"]:
        indent += 2

    return indent
