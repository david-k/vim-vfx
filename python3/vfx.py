import vim
import io
from pathlib import Path

from dataclasses import dataclass, field
from typing import NewType, Optional

import dir_tree


# Globals
#===============================================================================
BufNo = NewType("BufNo", int)

@dataclass
class Session:
    buf_no: BufNo
    dtree: dir_tree.DirTree
    gen_id: int
    alternate_buf: BufNo


@dataclass
class GlobalTreeState:
    root_dir: Optional[Path] = None
    expanded_dirs: set[Path] = field(default_factory=set)
    show_dotfiles: bool = False
    cursor_line: int = 1


GLOBAL_STATE = GlobalTreeState()
NEXT_GEN_ID = 1
SESSIONS: dict[int, Session] = {}


# Utils
#===============================================================================
def next_gen_id() -> int:
    global NEXT_GEN_ID

    gid = NEXT_GEN_ID
    NEXT_GEN_ID += 1
    return gid


def get_session() -> Session:
    buf_no = BufNo(vim.current.buffer.number)
    return SESSIONS.get(buf_no, None)


# Should be called right before the buffer is closed
def on_exit(s: Session):
    global GLOBAL_STATE

    GLOBAL_STATE.root_dir = s.dtree.root_dir
    GLOBAL_STATE.expanded_dirs = s.dtree.expanded_dirs
    GLOBAL_STATE.show_dotfiles = s.dtree.show_dotfiles
    GLOBAL_STATE.cursor_line = vim.eval('line(".")')

    del SESSIONS[s.buf_no]


def restore_alternate_buf(s: Session):
    if int(vim.eval(f"buflisted({s.alternate_buf})")):
        vim.command(f"let @# = {s.alternate_buf}")


# Vim helpers
#-----------------------------------------------------------
def escape_for_vim_expr(text: str):
    return "'" + text.replace("'", "''") + "'"

def escape_for_vim_command(text: str):
    return text.replace("\n", "\\\n")


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


# Working with filepaths in the buffer
#-----------------------------------------------------------
def is_expanded(line_no: int) -> bool:
    return get_indent(line_no+1) > get_indent(line_no)


def get_path_at(line_no: int) -> Path:
    parts = [get_filename_at_line(line_no)]
    while line_no := go_to_parent(line_no):
        parts.append(get_filename_at_line(line_no))

    head, *tail = reversed(parts)
    return Path(head).joinpath(*tail)


def go_to_parent(line_no: int) -> int:
    cur_indent = get_indent(line_no)
    for cur_line_no in range(line_no - 1, 0, -1):
        if get_indent(cur_line_no) < cur_indent:
            return cur_line_no

    return 0


def get_indent(line_no: int) -> int:
    line = vim_get_line(line_no)
    indent = 0
    for c in line:
        if c == "\t":
            indent += 1
        else:
            break

    return indent


def get_filename_at_line(line_no: int) -> str:
    filename = vim_get_line(line_no).strip()
    return dir_tree.unescape(filename)


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
    vim.command('setlocal buftype=nofile')
    vim.command('setlocal noswapfile')
    vim.command('setlocal buflisted')
    vim.command('setlocal ft=vfx')

    buf_no = BufNo(vim.current.buffer.number)
    SESSIONS[buf_no] = Session(
        buf_no = buf_no,
        dtree = dir_tree.DirTree(
            GLOBAL_STATE.root_dir,
            GLOBAL_STATE.expanded_dirs,
            GLOBAL_STATE.show_dotfiles
        ),
        gen_id = next_gen_id(),
        alternate_buf = alt_buf
    )

    update_buffer()
    vim.command(f'normal {GLOBAL_STATE.cursor_line}G')


def quit():
    s = get_session()
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
    s = get_session()
    filepath = s.dtree.root_dir / get_path_at(vim_get_line_no())
    if filepath.is_dir():
        s.dtree.cd(filepath)
        update_buffer()

        new_name = make_buf_name(str(s.dtree.root_dir))
        rename_buffer(new_name)
    else:
        on_exit(s)
        vim.command("keepalt edit " + escape_for_vim_command(str(filepath)))

def toggle_expand():
    s = get_session()
    if s.dtree.toggle_expand(get_path_at(vim_get_line_no())):
        update_buffer()


def move_up():
    s = get_session()
    if s.dtree.move_up():
        update_buffer()

        new_name = make_buf_name(str(s.dtree.root_dir))
        rename_buffer(new_name)


def toggle_dotfiles():
    s = get_session()
    s.dtree.show_dotfiles = not s.dtree.show_dotfiles
    s.dtree.refresh_from_fs()
    update_buffer()


def update_buffer():
    s = get_session()
    with io.StringIO() as buf:
        dir_tree.print_tree(buf, s.dtree)
        buf.seek(0, io.SEEK_SET)
        vim_set_buffer_contents(buf.readlines())


# For internal use only. Called when the BufUnload autocmd event is fired.
def on_buf_unload():
    s = get_session()
    # Check if we have already exited
    if not s:
        return

    on_exit(s)
    restore_alternate_buf(s)
