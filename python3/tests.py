import sys
import io
import inspect
import shutil
from config import Config, default_config
from pathlib import Path
from dataclasses import field

from dir_tree import *
from dir_parser import (
    parse_buffer,
    unwrap,
)


SCRIPT_DIR = Path(__file__).parent.resolve()
TEST_SCENARIOS_DIR = (SCRIPT_DIR / "../test-scenarios").resolve()
TEST_REALM_DIR = (SCRIPT_DIR / "../test-realm").resolve()


# Utils
#===============================================================================
def equiv_nodes(a: DirNode, b: DirNode) -> bool:
    if a.id != b.id or a.kind != b.kind or a.name != b.name:
        return False

    if a.is_dir():
        if a.is_expanded != b.is_expanded:
            return False

        if a.is_expanded:
            a_children_by_name = {c.name: c for c in a.children}
            for b_child in b.children:
                if a_child := a_children_by_name.get(b_child.name):
                    if not equiv_nodes(a_child, b_child):
                        return False
                else:
                    return False
    return True

def equiv_trees(a: DirTree, b: DirTree) -> bool:
    if a.has_details != b.has_details:
        return False

    if a.has_dotfiles != b.has_dotfiles:
        return False

    return equiv_nodes(a.root, b.root)


def equiv_mods(actual_mods: list[Modification], expected_mod_strings: list[str]) -> bool:
    actual_mods_set = set()
    for mod in actual_mods:
        for op in mod.get_ops():
            actual_mods_set.add(op_to_str(op))

    expected_mods_set = set(expected_mod_strings)
    return actual_mods_set == expected_mods_set


def mk_tree(buffer: str) -> DirTree:
    return parse_buffer(default_config(), buffer.strip().splitlines())


def mk_path(p: Path|str) -> Path:
    if isinstance(p, Path):
        return p

    return Path(p)


def create_files(node: DirNode):
    if node.is_file():
        node.abs_filepath().touch()
    else:
        node.abs_filepath().mkdir()
        for child in node.children:
            create_files(child)


def write_and_parse_tree(tree: DirTree, node: DirNode, path_info: Optional[PathInfoNode], config: Config) -> DirTree:
    with io.StringIO() as buf:
        write_tree(config, buf, tree, node, path_info)
        buf.seek(0, io.SEEK_SET)
        return parse_buffer(config, buf.readlines())


def tree_to_string(tree: DirTree, node: DirNode, path_info: Optional[PathInfoNode], config: Config) -> str:
    with io.StringIO() as buf:
        write_tree(config, buf, tree, node, path_info)
        buf.seek(0, io.SEEK_SET)
        return buf.read()

def print_error(msg: str):
    print("Error: " + msg)


def to_set(root: PathInfoNode) -> set[Path]:
    def to_set_rec(node: PathInfoNode, cur_path: Path, expanded_dirs: set[Path]):
        if node.is_expanded:
            expanded_dirs.add(cur_path)

        if node.num_expanded_children:
            for child_name, child_node in node.children.items():
                to_set_rec(child_node, cur_path / child_name, expanded_dirs)

    expanded_dirs: set[Path] = set()
    to_set_rec(root, Path("/"), expanded_dirs)
    return expanded_dirs


class TestSession:
    test_dir: Path
    base_tree: DirTree
    buf_tree: DirTree
    path_info: PathInfoNode
    cur_path: Path
    config: Config
    modifications: list[Modification]


    @staticmethod
    def init_fs(tree: DirTree, init_dir: Path|str) -> "TestSession":
        create_files(tree.root)

        init_dir = mk_path(init_dir)
        assert not init_dir.is_absolute()

        s = TestSession()
        s.test_dir = tree.root_dir()
        s.config = default_config()
        s.cur_path = tree.root_dir() / init_dir

        s.path_info = PathInfoNode()
        s.path_info.set_expanded(s.cur_path, True)

        s.base_tree = DirTree(s.cur_path, has_dotfiles=tree.has_dotfiles, has_details=tree.has_details)
        refresh_node(s.base_tree, s.base_tree.root, path_info=s.path_info.lookup(s.base_tree.root_dir()), refresh=True, error_callback=print_error)

        s.buf_tree = write_and_parse_tree(s.base_tree, s.base_tree.root, s.path_info, s.config)
        assert len(s.buf_tree.root.gather_errors()) == 0

        s.modifications = []

        return s


    def print_mods(self):
        for mod in self.modifications:
            mod.print()


    # Editor commands
    #------------------------------
    def set_expanded(self, path: Path|str, expanded: bool):
        abs_path = self.buf_tree.root_dir() / mk_path(path)
        is_expanded = self.path_info.is_expanded_at(abs_path)
        self.path_info.set_expanded(abs_path, not is_expanded)

        update_base_tree_from_buf(self.base_tree, self.buf_tree, self.path_info, error_callback=print_error)
        update_buf_tree_from_base(self.buf_tree, self.buf_tree.root_dir(), self.base_tree, self.path_info, get_mods_by_id(self.modifications))
        self.modifications = compute_changes(self.base_tree, self.buf_tree)


    def change_dir(self, new_buf_dir_rel: Path|str):
        new_buf_dir_rel = mk_path(new_buf_dir_rel)
        assert not new_buf_dir_rel.is_absolute()

        new_buf_dir = self.test_dir / new_buf_dir_rel
        self.base_tree.add_path(new_buf_dir)
        self.path_info.set_expanded(new_buf_dir, True)
        self.cur_path = new_buf_dir

        # Refresh base_tree
        refresh_node(self.base_tree, self.base_tree.lookup(self.cur_path), self.path_info.lookup_or_create(self.cur_path), refresh=False, error_callback=print_error)

        update_buf_tree_from_base(self.buf_tree, new_buf_dir, self.base_tree, self.path_info, get_mods_by_id(self.modifications))
        assert len(self.buf_tree.root.gather_errors()) == 0

        self.modifications = compute_changes(self.base_tree, self.buf_tree)


    def update_buffer(self, new_buffer: str):
        self.buf_tree = parse_buffer(self.config, new_buffer.strip().splitlines())
        self.path_info.set_node(self.buf_tree.root_dir(), get_expanded_dirs_for_node(self.buf_tree.root))
        update_base_tree_from_buf(self.base_tree, self.buf_tree, self.path_info, error_callback=print_error)

        if self.buf_tree.root_dir() != self.cur_path:
            self.change_dir(self.buf_tree.root_dir().relative_to(self.test_dir))

        self.modifications = compute_changes(self.base_tree, self.buf_tree)


    def get_buffer(self) -> str:
        return tree_to_string(self.buf_tree, self.buf_tree.root, None, self.config)


# Tests
#===============================================================================

# When expanding a parent directory it is remembered which sub-directories where expanded before
def test_preserve_subdirectory_expansion():
    # Prepare test directory
    test_dir = TEST_REALM_DIR / inspect.getframeinfo(unwrap(inspect.currentframe())).function
    shutil.rmtree(test_dir, ignore_errors=True)

    scenario = (
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}
-2_ baz/
    -12_ sub/
        |13_ subfile
    |9_ ciao
    |10_ hello
    |8_ subfile
|6_ bar
|1_ foo
|5_ helo
|4_ hmm
"""
).strip()

    s = TestSession.init_fs(mk_tree(scenario), init_dir="./")
    assert to_set(s.path_info) == {test_dir}
    assert to_set(get_expanded_dirs(s.buf_tree)) == to_set(s.path_info)

    s.set_expanded("baz", True)
    s.set_expanded("baz/sub", True)
    assert to_set(s.path_info) == to_set(get_expanded_dirs(s.buf_tree)) == {test_dir, test_dir / "baz", test_dir / "baz/sub"}

    # If a sub-directory is expanded, then this remains the case after collapsing and expanding a
    # parent directory 
    s.set_expanded("baz", False)
    assert to_set(get_expanded_dirs(s.buf_tree)) == {test_dir}

    s.set_expanded("baz", True)
    assert to_set(get_expanded_dirs(s.buf_tree)) == {test_dir, test_dir / "baz", test_dir / "baz/sub"}

    # If a sub-directory is collapsed, then this remains the case after collapsing and expanding a
    # parent directory 
    s.set_expanded("baz/sub", False)
    assert to_set(get_expanded_dirs(s.buf_tree)) == {test_dir, test_dir / "baz"}

    s.set_expanded("baz", False)
    assert to_set(get_expanded_dirs(s.buf_tree)) == {test_dir}

    s.set_expanded("baz", True)
    assert to_set(get_expanded_dirs(s.buf_tree)) == {test_dir, test_dir / "baz"}


def test_remember_unsaved_changes_after_refresh():
    # Prepare test directory
    test_dir = TEST_REALM_DIR / inspect.getframeinfo(unwrap(inspect.currentframe())).function
    shutil.rmtree(test_dir, ignore_errors=True)

    scenario = (
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}
-1_ baz/
    |3_ hello
    |4_ subfile
|2_ foo
"""
)

    s = TestSession.init_fs(mk_tree(scenario), init_dir="./")
    s.update_buffer(
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}
-1_ baz/
    |3_ hello
|4_ subfile
|2_ foo
"""
)
    assert equiv_mods(s.modifications, ["MOVE baz/subfile -> subfile"])

    s.set_expanded("baz", False)
    assert equiv_trees(s.buf_tree, mk_tree(
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}
+1_ baz/
|4_ subfile
|2_ foo
"""
))
    assert equiv_mods(s.modifications, ["MOVE baz/subfile -> subfile"])

    s.set_expanded("baz", True)
    assert equiv_mods(s.modifications, ["MOVE baz/subfile -> subfile"])
    assert equiv_trees(s.buf_tree, mk_tree(
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}
-1_ baz/
    |3_ hello
|4_ subfile
|2_ foo
"""
))


def test_create_nested_file():
    # Prepare test directory
    test_dir = TEST_REALM_DIR / inspect.getframeinfo(unwrap(inspect.currentframe())).function
    shutil.rmtree(test_dir, ignore_errors=True)

    scenario = (
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}
-1_ baz/
    |3_ hello
    |4_ subfile
|2_ foo
"""
)

    s = TestSession.init_fs(mk_tree(scenario), init_dir="./")

    s.update_buffer(
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}
-1_ baz/
    |3_ hello
    |4_ subfile
|2_ foo
dubi/du
"""
)
    assert equiv_mods(s.modifications, [
        "CREATE dubi/",
        "CREATE dubi/du",
    ])


    s.update_buffer(
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}
-1_ baz/
    |3_ hello
    |4_ subfile
|2_ foo
  dubi/du
  hi/hi/
      ho
  haha/
      hihi/
"""
)
    assert equiv_mods(s.modifications, [
        "CREATE dubi/",
        "CREATE dubi/du",
        "CREATE hi/",
        "CREATE hi/hi/",
        "CREATE hi/hi/ho",
        "CREATE haha/",
        "CREATE haha/hihi/",
    ])


def test_cd_down():
    # Prepare test directory
    test_dir = TEST_REALM_DIR / inspect.getframeinfo(unwrap(inspect.currentframe())).function
    shutil.rmtree(test_dir, ignore_errors=True)

    scenario = (
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}
-1_ baz/
    |3_ hello
    |4_ subfile
|2_ foo
"""
)

    s = TestSession.init_fs(mk_tree(scenario), init_dir="./")
    s.change_dir("baz")
    expected_buffer = (
f"""
> root_id=1 show_details=False show_dotfiles=False | {test_dir}/baz
|3_ hello
|4_ subfile
"""
).lstrip()
    assert s.get_buffer() == expected_buffer


def test_cd_up():
    # Prepare test directory
    test_dir = TEST_REALM_DIR / inspect.getframeinfo(unwrap(inspect.currentframe())).function
    shutil.rmtree(test_dir, ignore_errors=True)

    scenario = (
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}
-2_ baz/
    -12_ sub/
        |13_ subfile
    |9_ ciao
    |10_ hello
    |8_ subfile
|6_ bar
|1_ foo
|5_ helo
|4_ hmm
"""
)
    s = TestSession.init_fs(mk_tree(scenario), init_dir="./baz")

    expected_buffer = (
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}/baz
+1_ sub/
|2_ ciao
|3_ hello
|4_ subfile
"""
).lstrip()
    assert s.get_buffer() == expected_buffer

    s.change_dir("./")
    expected_buffer = (
f"""
> root_id=5 show_details=False show_dotfiles=False | {test_dir}
-0_ baz/
    +1_ sub/
    |2_ ciao
    |3_ hello
    |4_ subfile
|6_ bar
|7_ foo
|8_ helo
|9_ hmm
"""
).lstrip()
    assert s.get_buffer() == expected_buffer


def test_cd_down_two_levels():
    # Prepare test directory
    test_dir = TEST_REALM_DIR / inspect.getframeinfo(unwrap(inspect.currentframe())).function
    shutil.rmtree(test_dir, ignore_errors=True)

    scenario = (
f"""
> root_id=0 show_details=False show_dotfiles=False | {test_dir}
-2_ baz/
    -12_ sub/
        |13_ subfile
        -14_ deepdir/
            |15_ deepfile
    |9_ ciao
    |10_ hello
|1_ foo
"""
)
    s = TestSession.init_fs(mk_tree(scenario), init_dir="./")

    s.change_dir("./baz/sub")
    s.set_expanded("deepdir", True)

    assert len(s.modifications) == 0


def test_up_twice():
    # Prepare test directory
    test_dir = TEST_REALM_DIR / inspect.getframeinfo(unwrap(inspect.currentframe())).function
    shutil.rmtree(test_dir, ignore_errors=True)

    scenario = (
f"""
> root_id=0 show_details=True show_dotfiles=False | {test_dir}
[drwxr-xr-x david david 4.0K 2025-03-04 22:09] -2_ baz/
[drwxr-xr-x david david 4.0K 2025-02-27 14:39]     -9_ sub/
[drwxr-xr-x david david 4.0K 2025-02-27 14:39]         +14_ deepdir/
[-rw-r--r-- david david  0.0 2025-01-28 09:25]         |15_ subfile
[-rw-r--r-- david david  0.0 2025-01-26 14:11]     |11_ ciao
[-rw-r--r-- david david  8.0 2025-01-26 14:58] |4_ bar
[-rw-r--r-- david david  0.0 2025-01-26 14:11] |5_ foo
"""
)
    s = TestSession.init_fs(mk_tree(scenario), init_dir="./baz/sub")
    s.change_dir("./baz")
    s.change_dir("./")


def test_up_up_expand():
    # Prepare test directory
    test_dir = TEST_REALM_DIR / inspect.getframeinfo(unwrap(inspect.currentframe())).function
    shutil.rmtree(test_dir, ignore_errors=True)

    scenario = (
f"""
> root_id=0 show_details=True show_dotfiles=False | {test_dir}
[drwxr-xr-x david david 4.0K 2025-03-04 22:09] -2_ baz/
[drwxr-xr-x david david 4.0K 2025-02-27 14:39]     -9_ sub/
[drwxr-xr-x david david 4.0K 2025-02-27 14:39]         +14_ deepdir/
[-rw-r--r-- david david  0.0 2025-01-28 09:25]         |15_ subfile
[-rw-r--r-- david david  0.0 2025-01-26 14:11]     |11_ ciao
[-rw-r--r-- david david  8.0 2025-01-26 14:58] |4_ bar
[-rw-r--r-- david david  0.0 2025-01-26 14:11] |5_ foo
"""
)
    s = TestSession.init_fs(mk_tree(scenario), init_dir="./baz/sub")
    s.change_dir("./")
    s.set_expanded("baz", True)


def test_parsing():
    scenario1 = (
f"""
> root_id=7 show_details=True show_dotfiles=False | /home/david/vfx-test
[drwxr-xr-x david david 4.0K 2025-02-03 10:31] -0_ baz/
[drwxr-xr-x david david 4.0K 2025-01-28 09:25]     -1_ sub/
[-rw-r--r-- david david  0.0 2025-01-28 09:25]         |6_ subfile
[-rw-r--r-- david david  0.0 2025-01-26 14:11]     |3_ ciao
[-rw-r--r-- david david  0.0 2025-01-26 14:11]     -4_ hello/
[-rw-r--r-- david david  0.0 2025-01-26 14:11]         -5_ so/
[-rw-r--r-- david david  0.0 2025-01-26 14:11]             |6_ yo
[-rw-r--r-- david david  0.0 2025-01-26 14:11]     |7_ asdf
[-rw-r--r-- david david  8.0 2025-01-26 14:58] |10_ bar
[-rw-r--r-- david david  0.0 2025-01-26 14:11] |11_ foo
"""
)

    #parse_buffer(default_config(), scenario1.strip().splitlines()).print()

    scenario2 = (
f"""
> root_id=0 show_details=True show_dotfiles=False | /home/david/vfx-test/baz
                                                 new_dir/
                                                     new_sub
[drwxr-xr-x david david 4.0K 2025-01-28 09:25]     +1_ sub/
[-rw-r--r-- david david  0.0 2025-01-26 14:11] |3_ ciao
[-rw-r--r-- david david  0.0 2025-01-26 14:11] |4_ hello
[-rw-r--r-- david david  0.0 2025-01-28 10:31] |5_ subfile
                                                 so/
                                                     ho
"""
)
    #parse_buffer(default_config(), scenario2.strip().splitlines()).print()


def test_expand_cd_sub():
    # Prepare test directory
    test_dir = TEST_REALM_DIR / inspect.getframeinfo(unwrap(inspect.currentframe())).function
    shutil.rmtree(test_dir, ignore_errors=True)

    scenario = (
f"""
> root_id=0 show_details=True show_dotfiles=False | {test_dir}
[drwxr-xr-x david david 4.0K 2025-03-04 22:09] -2_ baz/
[drwxr-xr-x david david 4.0K 2025-02-27 14:39]     -9_ sub/
[drwxr-xr-x david david 4.0K 2025-02-27 14:39]         +14_ deepdir/
[-rw-r--r-- david david  0.0 2025-01-28 09:25]         |15_ subfile
[-rw-r--r-- david david  0.0 2025-01-26 14:11]     |11_ ciao
[-rw-r--r-- david david  8.0 2025-01-26 14:58] |4_ bar
[-rw-r--r-- david david  0.0 2025-01-26 14:11] |5_ foo
"""
)
    s = TestSession.init_fs(mk_tree(scenario), init_dir="./")
    s.set_expanded("baz", True)
    s.set_expanded("baz/sub", True)
    s.set_expanded("baz/sub", False)
    s.change_dir("./baz")

    assert s.buf_tree.lookup(Path("sub")).is_expanded == False


# Main
#===============================================================================
TEST_REALM_DIR.mkdir(exist_ok=True)

test_preserve_subdirectory_expansion()
test_remember_unsaved_changes_after_refresh()
test_create_nested_file()
test_cd_down()
test_cd_down_two_levels()
test_cd_up()
test_up_twice()
test_up_up_expand()
test_parsing()
test_expand_cd_sub()
