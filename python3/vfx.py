import vim # type: ignore
import io
import re
import subprocess
import shutil
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
    NODE_FLAGS_LEN,
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


def change_dir(s: Session, new_dir: Path):
    s.view.cd(new_dir)
    new_name = make_buf_name(str(s.view.root_dir()))
    rename_nonfile_buffer(vim.current.buffer, new_name)
    vim.eval(f'chdir( {escape_for_vim_expr(str(s.view.root_dir()))} )')


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

def escape_fn_for_vim_cmd(text: str):
    #return re.sub(r'(\n| |\t|\r|\\)', r'\\\1', text)

    esc = vim.eval(f'fnameescape( {escape_for_vim_expr(text)} )')
    if esc == "" and text != "":
        raise Exception("fnameescape() failed")

    return esc


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


def rename_nonfile_buffer(vim_buffer, new_name: str):
    last_buf_nr = int(vim.eval('bufnr("$")'))
    old_name = vim_buffer.name
    vim_buffer.name = new_name

    # Renaming a buffer creates a new, unlisted buffer with the previous buffer name.
    # Delete this new buffer.
    new_last_buf_nr = int(vim.eval('bufnr("$")'))
    if new_last_buf_nr > last_buf_nr:
        last_buf = vim.buffers[new_last_buf_nr]
        # This condition should always be true, but check anyway so we don't
        # accidentally wipe the wrong buffer
        if last_buf.name == old_name:
            vim.command(f'bwipeout {last_buf.number}')


def move_file(vim_buffer, new_filepath: Path):
    old_filepath = Path(vim_buffer.name)
    rename_nonfile_buffer(vim_buffer, str(new_filepath))

    # After having renamed the buffer with the above call to
    # `rename_nonfile_buffer()` we still need to do the following:
    #
    # 1. Moving the actual file
    # 2. Write the renamed buffer to disk, overwriting the moved file from the first step.
    #
    # The second step may seem redundant, but here are the reasons:
    # - There does not seem a way around the fact that we need to execute
    #   something like :saveas or :write. Otherwise, Vim will complain that the
    #   file on disk has changed (either when activating the buffer again or
    #   when trying to save).
    # - However, when writing a new file with :saveas or :write, file
    #   permissions are not preserved. Thus, we move the original file to its
    #   intended destination -- ensuring that all file attributes are preserved
    #   -- even though we then immediatly overwrite it with the contents of then
    #   buffer.
    #
    # This means that moving a file that is connected to a Vim buffer will
    # always save the buffer to disk. I don't quite like it, but this seems to
    # be the least annoying solution to me.
    #
    # (Isn't it kind of ridiculous that Vim doesn't have a builtin command to
    # properly move a file together with its buffer?)

    cur_buf_no_bkp = vim.current.buffer.number
    cur_bufhidden_bkp = vim.eval("&bufhidden")
    vim.command(f"setl bufhidden=hide") # Don't wipe buffer when switching to another one
    try:
        old_filepath.replace(new_filepath)
        vim.command(f"keepalt buffer {vim_buffer.number}")
        vim.command("silent write!") # ! because the file already exists
    finally:
        vim.command(f"keepalt buffer {cur_buf_no_bkp}")
        vim.command(f"setl bufhidden={cur_bufhidden_bkp}")


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


# Printing
#-----------------------------------------------------------
NUM_DETAIL_COLUMNS = 5

def print_entries(entries: list[DirNode], max_id_width: int = 6):
    column_widths = [0] * (NUM_DETAIL_COLUMNS + 1)
    column_widths[-1] = max_id_width

    with io.StringIO() as buf:
        write_entries(buf, entries, column_widths)
        buf.seek(0, io.SEEK_SET)
        print(buf.getvalue())


def write_tree(out: TextIO, view: DirView) -> int:
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

        out.write(f"{node_state}{entry.id[0]}:{entry.id[1]:0{max_id_width}}") # TODO Also need a max_width for gen id
        if entry.is_executable:
            out.write("x")
        else:
            out.write("_")

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


# Compute node changes
#-----------------------------------------------------------
@dataclass
class NodeChangeInfo:
    old_node: Optional[DirNode] = None
    new_occurrences: list[DirNode] = field(default_factory=list)
    explicitly_removed: bool = False
    # So we know that we don't need to generate explicit REMOVE operations if the parent is already being removed
    implicitly_removed_by: Optional[NodeID] = None
    removed_line_no: Optional[int] = None


class NodeChanges:
    old_nodes_by_id: dict[NodeID, DirNode]
    changes: dict[NodeID, NodeChangeInfo]

    def __init__(self, old_nodes_by_id: dict[NodeID, DirNode]):
        self.old_nodes_by_id = old_nodes_by_id
        self.changes = {}

    def node_new_occurrence(self, new_node: DirNode):
        # Is this is a new entry?
        if new_node.id[0] == -1:
            change_info = self._get_changes_for_new_node(new_node)
            change_info.new_occurrences.append(new_node)
        # Or is this an existing entry, possibly modified?
        elif old_node := self.old_nodes_by_id.get(new_node.id):
            change_info = self._get_changes_for_existing_node(old_node)
            change_info.new_occurrences.append(new_node)
        else:
            raise Exception(f"Invalid node id: {new_node.id}")

    def node_removed(self, old_node: DirNode, line_no: Optional[int] = None):
        change_info = self._get_changes_for_existing_node(old_node)
        change_info.explicitly_removed = True
        if line_no is not None:
            change_info.removed_line_no = line_no
        self._mark_nodes_implicitly_removed(old_node.children, old_node.id)

    def _mark_nodes_implicitly_removed(self, old_nodes: list[DirNode], implicitly_removed_by: NodeID):
        for old_node in old_nodes:
            change_info = self._get_changes_for_existing_node(old_node)
            change_info.implicitly_removed_by = implicitly_removed_by
            if old_node.is_dir():
                self._mark_nodes_implicitly_removed(old_node.children, implicitly_removed_by)

    def _get_changes_for_existing_node(self, old_node: DirNode) -> NodeChangeInfo:
        change_info = self.changes.setdefault(old_node.id, NodeChangeInfo(old_node=old_node))
        assert change_info.old_node is old_node
        return change_info

    def _get_changes_for_new_node(self, new_node: DirNode) -> NodeChangeInfo:
        change_info = self.changes.setdefault(new_node.id, NodeChangeInfo())
        assert change_info.old_node is None
        assert not change_info.explicitly_removed
        return change_info


def compute_node_changes(view: DirView, lines: list[str], indent_offset: int) -> NodeChanges:
    node_changes = NodeChanges(_nodes_by_id(view.tree.root))
    buf_tree = parse_tree(CONFIG, view.root_dir(), lines, indent_offset)
    _compute_node_changes(view.tree.root.children, buf_tree.root.children, node_changes)
    return node_changes


def _compute_node_changes(
    old_nodes: list[DirNode],
    new_nodes: list[DirNode],
    # Output:
    node_changes: NodeChanges
):
    # For the old nodes, there is a one-to-one mapping between node IDs and nodes
    old_nodes_by_id: dict[NodeID, DirNode] = {node.id:node for node in old_nodes}
    # However, there may be multiple new nodes with the same ID (because of COPYing)
    new_nodes_by_id: dict[NodeID, list[DirNode]] = {}
    for new_node in new_nodes:
        new_nodes_by_id.setdefault(new_node.id, []).append(new_node)

    # Handle new and modified nodes
    for (new_node_id, new_nodes_with_same_id) in new_nodes_by_id.items():
        # If the new node ID existed previously we need to detemine if anything has changed
        if old_node := old_nodes_by_id.get(new_node_id):
            old_node_removed = True
            old_node_line_no = None
            for new_node in new_nodes_with_same_id:
                if new_node.kind != old_node.kind:
                    raise Exception("TODO Changing node kind not supported yet")

                if new_node.link_target != old_node.link_target:
                    raise Exception("TODO Changing link target is not supported yet")

                if new_node.name == old_node.name:
                    old_node_removed = False
                else:
                    node_changes.node_new_occurrence(new_node)

                if _have_equivalent_parents(new_node, old_node):
                    old_node_line_no = new_node.line_no

                if new_node.is_dir():
                    _compute_node_changes(old_node.children, new_node.children, node_changes)

            if old_node_removed:
                node_changes.node_removed(old_node, old_node_line_no)
        else:
            _add_new_occurrences_recursively(new_nodes_with_same_id, node_changes)

    # Handle removed nodes
    removed_node_ids = old_nodes_by_id.keys() - new_nodes_by_id.keys()
    for removed_node_id in removed_node_ids:
        node_changes.node_removed(old_nodes_by_id[removed_node_id])


def _add_new_occurrences_recursively(new_nodes: list[DirNode], node_changes: NodeChanges) -> None:
    for new_entry in new_nodes:
        if new_entry.is_dir():
            _add_new_occurrences_recursively(new_entry.children, node_changes)

        node_changes.node_new_occurrence(new_entry)


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


def _have_equivalent_parents(a: DirNode, b: DirNode) -> bool:
    if a.parent is None or b.parent is None:
        return a.parent == b.parent

    # Do they both have root as their parent?
    # (Root node IDs are not parsed and so are always different)
    if a.parent.parent is None:
        return b.parent.parent is None

    return a.parent.id == b.parent.id


# Filesystem operations
#-----------------------------------------------------------
def compute_operations(node_changes: NodeChanges) -> list[dict]:
    operations = []
    for change in node_changes.changes.values():
        if change.old_node is None:
            for new_node in change.new_occurrences:
                operations.append({"kind": "NEW", "node": new_node, "line_no": new_node.line_no})
        else:
            if change.explicitly_removed:
                if len(change.new_occurrences) == 1:
                    operations.append({"kind": "MOVE", "old_node": change.old_node, "new_node": change.new_occurrences[0], "line_no": change.new_occurrences[0].line_no})
                else:
                    for new_node in change.new_occurrences:
                        operations.append({"kind": "COPY", "old_node": change.old_node, "new_node": new_node, "line_no": new_node.line_no})
                    operations.append({"kind": "REMOVE", "node": change.old_node, "line_no": change.removed_line_no})
            else:
                for new_node in change.new_occurrences:
                    operations.append({"kind": "COPY", "old_node": change.old_node, "new_node": new_node, "line_no": new_node.line_no})

    return operations



# Functions that are called from Vim
#===============================================================================
# Initializes a new Vfx buffer
def init():
    global SESSIONS, GLOBAL_TREE_STATE

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
    vim.command("edit " + escape_fn_for_vim_cmd(buf_name))
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

    vim.eval(f'prop_type_add("vfx_change", {{"bufnr": {buf_no}, "highlight": "Changed"}})')
    vim.eval(f'prop_type_add("vfx_remove", {{"bufnr": {buf_no}, "highlight": "Removed"}})')

    update_buffer()
    vim_set_line_no(GLOBAL_TREE_STATE.cursor_line)
    vim_set_column_no(GLOBAL_TREE_STATE.cursor_column)


def quit():
    s = get_session(); assert s

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
        change_dir(s, filepath)
        update_buffer()
    else:
        default_app = default_app_for(filepath)
        if should_open_in_vim(default_app):
            vim.command("keepalt edit " + escape_fn_for_vim_cmd(str(filepath)))
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
    root_dir = s.view.root_dir()
    if root_dir.parent != root_dir:
        change_dir(s, root_dir.parent)
        update_buffer()


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

        first_tab_stop = s.indent_offset + CONFIG.node_state_symbol_width + 1 # +1 for the space following the node state symbol
        vim.command(f"setl varsofttabstop={first_tab_stop},{CONFIG.indent_width}")
        # There is also 'vartabstop' in case I want to support tabs for indentation

    vim.command("setl nomodified")


# For internal use only. Called when the BufWriteCmd autocmd event is fired.
def on_buf_save():
    if is_buf_modified():
        s = get_session(); assert s

        node_changes = compute_node_changes(s.view, vim.current.buffer, s.indent_offset)
        operations = compute_operations(node_changes)
        if not operations:
            vim.command("setl nomodified") # Don't mark the buffer as modified
            return

        for op in operations:
            if op["kind"] == "NEW":
                cmd = "MKDIR" if op['node'].is_dir() else "TOUCH"
                print(f"{cmd} {op['node'].filepath_str()}")
            elif op["kind"] == "REMOVE":
                cmd = "REMOVE-REC" if op['node'].is_dir() else "REMOVE"
                print(f"{cmd} {op['node'].filepath_str()}")
            elif op["kind"] == "COPY":
                cmd = "COPY-REC" if op['old_node'].is_dir() else "COPY"
                print(f"COPY {op['old_node'].filepath_str()} -> {op['new_node'].filepath_str()}")
            elif op["kind"] == "MOVE":
                print(f"MOVE {op['old_node'].filepath_str()} -> {op['new_node'].filepath_str()}")
            else:
                raise Exception("Unsupported op: " + op["kind"])

        choice = int(vim.eval('confirm("Apply changes?", "yes\nno", 2)'))
        if choice == 1:
            buffers_by_name = {buf.name: buf for buf in vim.buffers if buf.valid}
            for op in operations:
                if op["kind"] == "NEW":
                    filepath = s.view.root_dir() / op["node"].filepath()
                    if op["node"].is_file():
                        if filepath.exists():
                            print("Error: destination already exists: " + str(filepath))
                            print("Aborting")
                            return
                        else:
                            filepath.touch()
                    else:
                        filepath.mkdir(parents=True)

                elif op["kind"] == "REMOVE":
                    filepath = s.view.root_dir() / op["node"].filepath()
                    subprocess.run(["trash", "--", filepath])

                elif op["kind"] == "COPY":
                    old_filepath = s.view.root_dir() / op["old_node"].filepath()
                    new_filepath = s.view.root_dir() / op["new_node"].filepath()
                    if new_filepath.exists():
                        print("Error: destination already exists: " + str(new_filepath))
                        print("Aborting")
                        return
                    else:
                        if op["old_node"].is_file():
                            shutil.copyfile(old_filepath, new_filepath, follow_symlinks=False)
                        else:
                            shutil.copytree(old_filepath, new_filepath, symlinks=True)

                elif op["kind"] == "MOVE":
                    old_filepath = s.view.root_dir() / op["old_node"].filepath()
                    new_filepath = s.view.root_dir() / op["new_node"].filepath()
                    if new_filepath.exists():
                        print("Error: destination already exists: " + str(new_filepath))
                        print("Aborting")
                        return
                    else:
                        # For the lookup to work old_filepath needs to denote an absolute path
                        buffer = buffers_by_name.get(str(old_filepath))
                        if buffer is None:
                            old_filepath.rename(new_filepath)
                        else:
                            move_file(buffer, new_filepath)

                else:
                    raise Exception("Unsupported op: " + op["kind"])


            s.view.refresh_from_fs()
            update_buffer()


def display_changes():
    def add_text_prop(line_no, text, style, align = "after"):
        if align == "after":
            text = "    " + text
        else:
            text += "\n"
        vim.eval(f'prop_add({line_no}, 0, {{"type": "{style}", "text": {escape_for_vim_expr(text)}, "text_align": "{align}"}})')

    # TODO Use b:changetick to only recompute the changes if the buffer has changed

    s = get_session(); assert s
    node_changes = compute_node_changes(s.view, vim.current.buffer, s.indent_offset)
    operations = compute_operations(node_changes)
    vim.eval('prop_clear(1, line("$"))')
    for op in operations:
        if op["kind"] == "NEW":
            cmd = "MKDIR" if op['node'].is_dir() else "TOUCH"
            add_text_prop(op["node"].line_no, f"{cmd} {op['node'].filepath_str()}", style="vfx_change")
        elif op["kind"] == "REMOVE":
            cmd = "REMOVE-REC" if op['node'].is_dir() else "REMOVE"
            text = f"{cmd} {op['node'].filepath_str()}"
            if op["line_no"] is None:
                add_text_prop(1, text, align="above", style="vfx_remove")
            else:
                add_text_prop(op["line_no"], text, style="vfx_remove")
        elif op["kind"] == "COPY":
            cmd = "COPY-REC" if op['old_node'].is_dir() else "COPY"
            add_text_prop(op["line_no"], f"COPY {op['old_node'].filepath_str()} -> {op['new_node'].filepath_str()}", style="vfx_change")
        elif op["kind"] == "MOVE":
            add_text_prop(op["line_no"], f"MOVE {op['old_node'].filepath_str()} -> {op['new_node'].filepath_str()}", style="vfx_change")
        else:
            raise Exception("Unsupported op: " + op["kind"])



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
