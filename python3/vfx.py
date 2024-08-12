import vim # type: ignore
import io
import re
import subprocess
from pathlib import Path

from enum import Enum
from dataclasses import (
    dataclass,
    field
)
from typing import (
    NewType,
    Optional,
    Callable,
    TextIO,
    TypeVar,
    Generic
)

from config import Config, default_config
from dir_tree import (
    DirTree,
    DirView,
    DirNode,
    NodeKind,
    LinkStatus,
    NodeID,
)
from dir_parser import (
    parse_line,
    parse_tree,
    compute_indent,
)


# Global state
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
    cursor_column: int = 1


CONFIG = default_config()
GLOBAL_TREE_STATE = GlobalTreeState()
LAST_GEN_ID = 0
SESSIONS: dict[int, Session] = {}


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
    global GLOBAL_TREE_STATE

    GLOBAL_TREE_STATE.root_dir = s.view.root_dir()
    GLOBAL_TREE_STATE.expanded_dirs = s.view.expanded_dirs
    GLOBAL_TREE_STATE.show_dotfiles = s.view.show_dotfiles
    GLOBAL_TREE_STATE.show_details = s.view.show_details
    GLOBAL_TREE_STATE.cursor_line = vim_get_line_no()
    GLOBAL_TREE_STATE.cursor_column = vim_get_column_no()

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
    line_no = vim_get_line_no()
    column = vim_get_column_no()
    buf = vim.current.buffer
    buf[:] = lines
    vim_set_line_no(line_no)
    vim_set_column_no(column)


def vim_get_line(line_no) -> str:
    return vim.eval(f'getline({line_no})')

def vim_get_line_no() -> int:
    return int(vim.eval('line(".")'))

def vim_set_line_no(line_no: int):
    vim.command(f'normal {line_no}G')

def vim_get_column_no() -> int:
    return int(vim.eval('col(".")'))

def vim_set_column_no(col: int):
    if col > 1:
        vim.command(f'normal 0{col-1}l')
    else:
        vim.command(f'normal 0')


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
    parts = [get_name_at(vim_get_line(line_no))]
    while line_no := go_to_parent(line_no, indent_offset):
        parts.append(get_name_at(vim_get_line(line_no)))

    head, *tail = reversed(parts)
    return Path(head).joinpath(*tail)


def go_to_parent(line_no: int, indent_offset: int) -> int:
    cur_indent = get_indent_at(line_no, indent_offset)
    for cur_line_no in range(line_no - 1, 0, -1):
        if get_indent_at(cur_line_no, indent_offset) < cur_indent:
            return cur_line_no

    return 0


def get_indent_at(line_no: int, indent_offset: int) -> int:
    line = vim_get_line(line_no)
    if not line:
        return 0

    _, segments = parse_line(CONFIG, line)
    return compute_indent(CONFIG, segments, indent_offset)


def get_name_at(line: str) -> str:
    node, _ = parse_line(CONFIG, line)
    return node.name


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
            # ATTENTION: This must be kept in sync with the computation of `details_width` in `write_tree()` and with `compute_column_widths()`
            out.write(f"[{d.mode:<{c1_width}} {d.user:<{c2_width}} {d.group:<{c3_width}} {d.size:>{c4_width}} {d.mtime:<{c5_width}}] ")

        out.write(indent * CONFIG.indent_width * " ")

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
        if entry.kind == NodeKind.DIRECTORY:
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

                if entry.kind == NodeKind.DIRECTORY and entry.link_target.name != "": # Root dir has no name
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
        widths[4] = max(widths[3], len(node.details.mtime))

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


# Filesystem operations
#-----------------------------------------------------------
def compute_operations(view: DirView, lines: list[str], indent_offset: int) -> list[dict]:
    old_nodes_by_id = _nodes_by_id(view.tree.root)
    operations: list[dict] = []

    buf_tree, buf_nodes_by_id = parse_tree(CONFIG, view.root_dir(), lines, indent_offset)
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
    old_nodes_by_id: dict[NodeID, DirNode],
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


def _nodes_by_id(node: DirNode) -> dict[NodeID, DirNode]:

    def _nodes_by_id_rec(
        node: DirNode,
        nodes_out: dict[NodeID, DirNode]
    ):
        for child in node.children:
            nodes_out[child.id] = child
            _nodes_by_id_rec(child, nodes_out)

    nodes_out: dict[NodeID, DirNode] = {}
    _nodes_by_id_rec(node, nodes_out)

    return nodes_out


# Externally called functions
#===============================================================================
# Initializes a new Vfx buffer
def init():
    global SESSIONS, GLOBAL_TREE_STATE

    if not GLOBAL_TREE_STATE.root_dir:
        GLOBAL_TREE_STATE.root_dir = Path.cwd()

    buf_name = make_buf_name(str(GLOBAL_TREE_STATE.root_dir))

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
            GLOBAL_TREE_STATE.root_dir,
            GLOBAL_TREE_STATE.expanded_dirs,
            GLOBAL_TREE_STATE.show_dotfiles,
            GLOBAL_TREE_STATE.show_details,
        )
    )

    update_buffer()
    vim_set_line_no(GLOBAL_TREE_STATE.cursor_line)
    vim_set_column_no(GLOBAL_TREE_STATE.cursor_column)


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
            vim.command(f"setl varsofttabstop={s.indent_offset+2},{CONFIG.indent_width}")
            # There is also 'vartabstop' in case I want to support tabs for indentation
        else:
            vim.command(f"setl varsofttabstop=2,{CONFIG.indent_width}")

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
            if op["kind"] in ["touch", "rm"]:
                print(op["kind"] + " " + str(op["filepath"]))
            elif op["kind"] in ["mkdir", "rm-rec", "rmdir"]:
                print(op["kind"] + " " + str(op["filepath"]) + "/")
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


# For internal use. Set as 'indentexpr'.
def get_indent_for_vim(line_no: int) -> int:
    s = get_session()

    if line_no <= 1:
        return 0

    prev_line = vim_get_line(line_no - 1)
    if not prev_line:
        return 0

    _, segments = parse_line(CONFIG, prev_line)

    # We want the indentation to be where the filename on the previous line starts.
    # However, the visual position depends on the value of 'conceallevel'
    # because if the node ID is concealed then this changes where the filename
    # is displayed.
    #
    # Here, we assume 'conceallevel=2' (i.e., the node ID and state are
    # replaced by a single character, which by default is either '|', '+', or
    # '-').
    #
    # Note: We assume no tabs are used. If we want to support tabs we need to
    # count the number of '\t' that occur in
    # `prev_line[:segments.node_state.start]` and multiply that with
    # `vim.eval("&tabstop")`.
    if segments.node_state is not None:
        return segments.node_state.start + 2 # +1 for the conceal-character and +1 for the following space
    else:
        return segments.name.start
