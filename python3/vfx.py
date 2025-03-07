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
from dir_tree import *
from dir_parser import (
    parse_line,
    parse_buffer,
    NODE_FLAGS_LEN,
    EntryOffset,
)


# TODO Test edge case: Changing a directory to a file with the same name (and vice versa)

# TODO Use listener_add() to listen for changes to the buffer so we don't need to reparse everything


# Global state
#===============================================================================
@dataclass
class Session:
    buf_no: int
    alternate_buf: int
    replaced_buf: int
    base_tree: DirTree
    cur_path: Path
    changedtick: int
    path_info: PathInfoNode
    mods_by_id: dict[NodeIDKnown, Modification] = field(default_factory=dict)


def vim_changetick() -> int:
    return int(vim.eval("b:changedtick"))


def update_session_from_buf(s: Session) -> Optional[DirTree]:
    changedtick = vim_changetick()
    if s.changedtick != changedtick:
        buf_tree = parse_buffer(CONFIG, vim.current.buffer)
        s.path_info.set_node(buf_tree.root_dir(), get_expanded_dirs_for_node(buf_tree.root))
        update_base_tree_from_buf(s.base_tree, buf_tree, s.path_info, error_callback=print_error)
        if buf_tree.root_dir() != s.cur_path:
            change_dir(s, buf_tree.root_dir())

        s.changedtick = changedtick
        return buf_tree

    return None


def update_session_from_buf__return_tree(s: Session) -> DirTree:
    if buf_tree := update_session_from_buf(s):
        return buf_tree

    return parse_buffer(CONFIG, vim.current.buffer)


@dataclass
class GlobalTreeState:
    root_dir: Optional[Path] = None
    path_info: PathInfoNode = field(default_factory=PathInfoNode)
    show_dotfiles: bool = False
    show_details: bool = True
    cursor_line: int = 1
    cursor_column: int = 1


CONFIG = default_config()
GLOBAL_TREE_STATE = GlobalTreeState()
SESSIONS: dict[int, Session] = {}


# Utils
#===============================================================================
def get_session() -> Optional[Session]:
    buf_no = int(vim.current.buffer.number)
    return SESSIONS.get(buf_no)


# Should be called right before the buffer is closed
def on_exit(s: Session):
    global GLOBAL_TREE_STATE

    GLOBAL_TREE_STATE.root_dir = s.cur_path
    GLOBAL_TREE_STATE.path_info = s.path_info.clone()
    GLOBAL_TREE_STATE.show_dotfiles = s.base_tree.has_dotfiles
    GLOBAL_TREE_STATE.show_details = s.base_tree.has_details
    GLOBAL_TREE_STATE.cursor_line = vim_get_line_no()
    GLOBAL_TREE_STATE.cursor_column = vim_get_column_no()

    del SESSIONS[s.buf_no]


def change_dir(s: Session, new_dir: Path, update_pwd: bool = False):
    assert new_dir.is_absolute()

    # Add new_dir to base_tree
    new_dir = new_dir.resolve()
    s.base_tree.add_path(new_dir)
    s.cur_path = new_dir
    s.path_info.set_expanded(new_dir, True)

    # Refresh base_tree
    refresh_node(
        s.base_tree,
        s.base_tree.lookup(s.cur_path),
        path_info = s.path_info.lookup_or_create(s.cur_path),
        refresh = False,
        error_callback = print_error,
    )

    new_name = make_buf_name(str(s.cur_path))
    rename_nonfile_buffer(vim.current.buffer, new_name)
    if update_pwd:
        vim_cd(s.cur_path)


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
    # Maybe just open all MIME types matching text/* in vim
    return default_app in ["", "vim.desktop", "gvim.desktop", "emacs.desktop"]


def is_dir_empty(path: Path) -> bool:
    for c in path.iterdir():
        return False

    return True


def update_buffer(tree: DirTree):
    s = get_session(); assert s
    was_modified = is_buf_modified()

    with io.StringIO() as buf:
        indent_offset = write_tree(CONFIG, buf, tree, tree.lookup(s.cur_path), s.path_info.try_lookup(s.cur_path))
        buf.seek(0, io.SEEK_SET)
        vim_set_buffer_contents(buf.readlines())

        s.changedtick = vim_changetick()

        first_tab_stop = indent_offset + CONFIG.node_state_symbol_width + 1 # +1 for the space following the node state symbol
        vim.command(f"setl varsofttabstop={first_tab_stop},{CONFIG.indent_width}")
        # There is also 'vartabstop' in case I want to support tabs for indentation

    if not was_modified:
        vim.command("setl nomodified")

def print_error(msg: str):
    vim.command(f"echohl ErrorMsg | echo { escape_for_vim_expr('Error: ' + msg) } | echohl None")


# Vim helpers
#-----------------------------------------------------------
def escape_for_vim_expr(text: str):
    return "'" + text.replace("'", "''") + "'"

def escape_fn_for_vim_cmd(text: str):
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


def vim_cd(path: Path):
    vim.eval(f'chdir( {escape_for_vim_expr(str(path))} )')


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
    # 1. Move the actual file
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
    #   -- and then immediatly overwrite it with the contents of the new
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


def add_text_prop(line_no: int, text: str, style: str, align: str = "after"):
    if align == "after":
        text = "    " + text
    else:
        text += "\n"
    vim.eval(f'prop_add({line_no}, 0, {{"type": "{style}", "text": {escape_for_vim_expr(text)}, "text_align": "{align}"}})')


# Working with filepaths in the buffer
#-----------------------------------------------------------
def get_path_at(line_no: int) -> Path:
    parts = [get_name_at(vim_get_line(line_no))]
    while line_no := go_to_parent(line_no):
        parts.append(get_name_at(vim_get_line(line_no)))

    head, *tail = reversed(parts)
    return Path(head).joinpath(*tail)


def go_to_parent(line_no: int) -> int:
    cur_offset = get_indent_at(line_no)
    for cur_line_no in range(line_no - 1, NUM_LINES_BEFORE_TREE, -1):
        next_offset = get_indent_at(cur_line_no)
        indent_diff = (next_offset.diff(cur_offset) / CONFIG.indent_width)
        if indent_diff < 0:
            return cur_line_no

    return 0


def get_indent_at(line_no: int) -> EntryOffset:
    line = vim_get_line(line_no)
    if not line:
        return EntryOffset()

    _, segments = parse_line(CONFIG, line)
    return EntryOffset.from_segments(segments, CONFIG)


def get_name_at(line: str) -> str:
    node, _ = parse_line(CONFIG, line)
    return node.name


# Functions that are called from Vim
#===============================================================================
# Initializes a new Vfx buffer
def init():
    global SESSIONS, GLOBAL_TREE_STATE

    if GLOBAL_TREE_STATE.root_dir is None:
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
    alt_buf = int(vim.eval('bufnr("#")'))
    replaced_buf = int(vim.eval('bufnr("%")'))

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

    buf_no = int(vim.current.buffer.number)
    path_info = GLOBAL_TREE_STATE.path_info.clone()
    path_info.set_expanded(GLOBAL_TREE_STATE.root_dir, True)

    SESSIONS[buf_no] = Session(
        buf_no = buf_no,
        alternate_buf = alt_buf,
        replaced_buf = replaced_buf,
        base_tree = DirTree(
            GLOBAL_TREE_STATE.root_dir,
            GLOBAL_TREE_STATE.show_details,
            GLOBAL_TREE_STATE.show_dotfiles
        ),
        cur_path = GLOBAL_TREE_STATE.root_dir,
        path_info = path_info,
        changedtick = 0,
    )


    vim.eval(f'prop_type_add("vfx_change", {{"bufnr": {buf_no}, "highlight": "Changed"}})')
    vim.eval(f'prop_type_add("vfx_remove", {{"bufnr": {buf_no}, "highlight": "Removed"}})')
    vim.eval(f'prop_type_add("vfx_error", {{"bufnr": {buf_no}, "highlight": "Error"}})')

    s = SESSIONS[buf_no]
    refresh_node(s.base_tree, s.base_tree.root, s.path_info.lookup_or_create(s.base_tree.root_dir()), refresh=True, error_callback=print_error)
    update_buffer(s.base_tree)

    vim_set_line_no(GLOBAL_TREE_STATE.cursor_line)
    vim_set_column_no(GLOBAL_TREE_STATE.cursor_column)

    s.changedtick = vim_changetick()


def quit():
    s = get_session(); assert s

    if is_buf_modified():
        print("There are unsaved changes")
        return

    # Since we have set bufhidden=wipe, quitting is done by simply switching
    # to another buffer (to the alternative buffer in this case).
    # Note that we need to use the value of `bufnr("#")` at the time init() was called, which we
    # store in s.replaced_buf. This is because renaming a buffer (like we do when changing the
    # directory) updates the alternative buffer.
    if s.replaced_buf != s.buf_no and int(vim.eval(f"buflisted({s.replaced_buf})")):
        vim.command(f"buffer {s.replaced_buf}")
    else:
        vim.command("enew")
        # This completely deletes the buffer once it is hidden (it won't even
        # be listed with `:ls!`)
        # This is needed to prevent the buffer list from being littered with
        # "[No Name]" buffers.
        vim.command("setlocal bufhidden=wipe")

    restore_alternate_buf(s)



def open_entry(split_window: bool = False):
    s = get_session(); assert s
    buf_tree = update_session_from_buf__return_tree(s)

    filepath = s.cur_path / get_path_at(vim_get_line_no())
    if filepath.is_dir():
        change_dir(s, filepath)
        update_buf_tree_from_base(buf_tree, s.cur_path, s.base_tree, s.path_info, s.mods_by_id)
        update_buffer(buf_tree)
    elif filepath.is_file():
        default_app = default_app_for(filepath)
        if should_open_in_vim(default_app):
            open_filepath = filepath
            cwd = vim.eval("getcwd()")
            if filepath.is_relative_to(cwd):
                open_filepath = filepath.relative_to(cwd)
            if split_window:
                cur_win = int(vim.eval("winnr()"))
                prev_win = int(vim.eval("winnr('#')"))
                if prev_win == 0 or prev_win == cur_win:
                    vim.command("vsplit")
                else:
                    vim.command(f"{prev_win}wincmd w")

            vim.command("keepalt edit " + escape_fn_for_vim_cmd(str(open_filepath)))
        else:
            xdg_open_bin = shutil.which("xdg-open")
            if xdg_open_bin is None:
                raise Exception("xdg-open could not be found")

            # Redirecting stdin, stdout and stderr ensures that the terminal can be closed even if
            # the spawned process is still running
            subprocess.Popen([xdg_open_bin, filepath], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def open_vim_cwd():
    s = get_session(); assert s
    vim_cd(s.base_tree.root_dir())


def change_vim_cwd():
    s = get_session(); assert s
    buf_tree = update_session_from_buf__return_tree(s)
    change_dir(s, Path(vim.eval("getcwd()")))
    update_buf_tree_from_base(buf_tree, s.cur_path, s.base_tree, s.path_info, s.mods_by_id)
    update_buffer(buf_tree)


def move_up():
    s = get_session(); assert s
    buf_tree = update_session_from_buf__return_tree(s)
    change_dir(s, s.cur_path.parent)
    update_buf_tree_from_base(buf_tree, s.cur_path, s.base_tree, s.path_info, s.mods_by_id)
    update_buffer(buf_tree)


def toggle_expand():
    s = get_session(); assert s
    buf_tree = update_session_from_buf__return_tree(s)
    node = buf_tree.try_lookup(get_path_at(vim_get_line_no()))
    if node is None or not node.is_dir() or node.is_new():
        return

    s.path_info.set_expanded(node.abs_filepath(), not node.is_expanded)

    update_base_tree_from_buf(s.base_tree, buf_tree, s.path_info, error_callback=print_error)
    update_buf_tree_from_base(buf_tree, s.cur_path, s.base_tree, s.path_info, s.mods_by_id)
    update_buffer(buf_tree)


def toggle_dotfiles():
    s = get_session(); assert s
    buf_tree = update_session_from_buf__return_tree(s)
    buf_tree.has_dotfiles = not buf_tree.has_dotfiles

    update_base_tree_from_buf(s.base_tree, buf_tree, s.path_info, error_callback=print_error)
    update_buf_tree_from_base(buf_tree, s.cur_path, s.base_tree, s.path_info, s.mods_by_id)
    update_buffer(buf_tree)


def toggle_details():
    s = get_session(); assert s
    buf_tree = update_session_from_buf__return_tree(s)
    buf_tree.has_details = not buf_tree.has_details

    update_base_tree_from_buf(s.base_tree, buf_tree, s.path_info, error_callback=print_error)
    update_buf_tree_from_base(buf_tree, s.cur_path, s.base_tree, s.path_info, s.mods_by_id)
    update_buffer(buf_tree)


# For internal use only. Called when the BufWriteCmd autocmd event is fired.
#def on_buf_save():
#    if is_buf_modified():
#        s = get_session(); assert s
#
#        node_changes = compute_node_changes(s.view, vim.current.buffer, s.indent_offset)
#        operations = compute_operations(node_changes)
#        if not operations:
#            vim.command("setl nomodified") # Don't mark the buffer as modified
#            return
#
#        apply_changes = True
#        if CONFIG.confirm_changes:
#            for op in operations:
#                print(op_to_str(op))
#
#            choice = int(vim.eval('confirm("Apply changes?", "yes\nno", 2)'))
#            apply_changes = choice == 1
#
#        if apply_changes:
#            buffers_by_name = {buf.name: buf for buf in vim.buffers if buf.valid}
#            for op in operations:
#                if op["kind"] == "NEW":
#                    filepath = s.view.root_dir() / op["node"].filepath()
#                    if op["node"].is_file():
#                        if filepath.exists():
#                            print("Error: destination already exists: " + str(filepath))
#                            print("Aborting")
#                            return
#                        else:
#                            filepath.touch()
#                    else:
#                        filepath.mkdir(parents=True)
#
#                elif op["kind"] == "REMOVE":
#                    filepath = s.view.root_dir() / op["node"].filepath()
#                    subprocess.run(["trash", "--", filepath])
#
#                elif op["kind"] == "COPY":
#                    old_filepath = s.view.root_dir() / op["old_node"].filepath()
#                    new_filepath = s.view.root_dir() / op["new_node"].filepath()
#                    if new_filepath.exists():
#                        print("Error: destination already exists: " + str(new_filepath))
#                        print("Aborting")
#                        return
#                    else:
#                        if op["old_node"].is_file():
#                            shutil.copyfile(old_filepath, new_filepath, follow_symlinks=False)
#                        else:
#                            shutil.copytree(old_filepath, new_filepath, symlinks=True)
#
#                elif op["kind"] == "MOVE":
#                    old_filepath = s.view.root_dir() / op["old_node"].filepath()
#                    new_filepath = s.view.root_dir() / op["new_node"].filepath()
#                    if new_filepath.exists():
#                        print("Error: destination already exists: " + str(new_filepath))
#                        print("Aborting")
#                        return
#                    else:
#                        # TODO When renaming a directory, we need to rename all open child buffers
#
#                        # For the lookup to work old_filepath needs to denote an absolute path
#                        buffer = buffers_by_name.get(str(old_filepath))
#                        if buffer is None:
#                            old_filepath.rename(new_filepath)
#                        else:
#                            move_file(buffer, new_filepath)
#
#                else:
#                    raise Exception("Unsupported op: " + op["kind"])
#
#
#            s.view.refresh_from_fs()
#            update_buffer()


def display_changes():
    # TODO Use b:changetick to only recompute the changes if the buffer has changed

    s = get_session(); assert s
    buf_tree = update_session_from_buf__return_tree(s)
    mods = compute_changes(s.base_tree, buf_tree)
    s.mods_by_id = get_mods_by_id(mods)
    vim.eval('prop_clear(1, line("$"))')
    for mod in mods:
        for op in mod.get_ops():
            if op["kind"] == "delete":
                add_text_prop(1, op_to_str(op), align="above", style="vfx_remove")
            elif op["kind"] == "create":
                add_text_prop(op['node'].line_no, op_to_str(op), style="vfx_change")
            elif op["kind"] in ["move", "copy"]:
                add_text_prop(op['dest_node'].line_no, op_to_str(op), style="vfx_change")


    def display_errors(node: DirNode):
        if node.error:
            if node.line_no is None:
                add_text_prop(1, node.error, align="above", style="vfx_error")
            else:
                add_text_prop(node.line_no, node.error, style="vfx_error")

        if node.is_dir() and node.is_expanded:
            for child in node.children:
                display_errors(child)

    display_errors(buf_tree.root)


# For internal use only. Called when the BufUnload autocmd event is fired.
def on_buf_unload():
    s = get_session()
    # Check if we have already exited
    if not s:
        return

    on_exit(s)
    restore_alternate_buf(s)


# For internal use only. Set as 'indentexpr'.
def get_indent_for_vim(line_no: int) -> int:
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


class Logger:
    buf_no: int

    def info(self, msg: str):
        vim.buffers[self.buf_no].append(msg)


def open_logger():
    buf_name = "Vfx Log"
    if not is_buf_name_available(buf_name):
        print("Log already open")
        return

    vim.command("vsplit " + escape_fn_for_vim_cmd("Vfx Log"))
    vim.command('setlocal bufhidden=wipe')
    vim.command('setlocal buftype=acwrite')
    vim.command('setlocal noswapfile')
    vim.command('setlocal buflisted')
