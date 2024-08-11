import os
import pwd
import grp
import stat
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


#===============================================================================
UniqueEntryID = tuple[int, int] # Generation ID, Entry ID


class EntryKind(Enum):
    DIRECTORY = 1
    FILE = 2


class LinkStatus(Enum):
    NO_LINK = 1
    GOOD = 2
    BROKEN = 3
    UNKNOWN = 4


@dataclass
class NodeDetails:
    user: str
    group: str
    mode: str
    size: str
    mtime: str


@dataclass
class DirNode:
    id: UniqueEntryID
    name: str # For all non-root nodes, the name does not contain any path separators
    kind: EntryKind
    parent: Optional["DirNode"] = None

    link_status: LinkStatus = LinkStatus.NO_LINK
    link_target: Optional[Path] = None

    # For files
    is_executable: bool = False

    # For directories
    is_expanded: bool = False
    children: list["DirNode"] = field(default_factory=list)

    details: Optional[NodeDetails] = None

    def is_dir(self) -> bool:
        return self.kind == EntryKind.DIRECTORY

    def is_file(self) -> bool:
        return self.kind == EntryKind.FILE

    def is_root(self) -> bool:
        return bool(self.parent)

    # Returns path relative to root
    def filepath(self) -> Path:
        if self.parent:
            return self.parent.filepath() / self.name

        return Path()


class DirTree:
    root: DirNode
    last_entry_id: int = 0

    def __init__(self, gen_id: int, root_dir: Path):
        self.root = DirNode(
            id = (gen_id, self._next_id()),
            name = str(root_dir),
            kind = EntryKind.DIRECTORY
        )

    def root_dir(self) -> Path:
        return Path(self.root.name)

    def cd(self, new_root_dir: Path):
        self.root = DirNode(
            id = (self.root.id[0], self._next_id()),
            name = str(new_root_dir),
            kind = EntryKind.DIRECTORY
        )

    def make_node(self,
        name: str,
        kind: EntryKind,
        parent: DirNode,
        is_executable: bool = False,
        link_target: Optional[Path] = None,
        link_status: LinkStatus = LinkStatus.NO_LINK,
        details: Optional[NodeDetails] = None
    ) -> DirNode:
        if not name:
            raise Exception("Nodes cannot have empty names")

        if len(Path(name).parts) != 1:
            raise Exception("Non-root nodes must not contain path separators")

        node = DirNode(
            id = (self.root.id[0], self._next_id()),
            parent = parent,
            name = name,
            kind = kind,
            is_executable = is_executable,
            link_target = link_target,
            link_status = link_status,
            details = details,
        )
        parent.children.append(node)
        return node

    def lookup(self, p: Path) -> Optional[DirNode]:
        return self._lookup(self.root, list(p.parts))

    def lookup_or_create(self, p: Path, kind: EntryKind) -> DirNode:
        node = self._lookup(self.root, list(p.parts), kind); assert node is not None
        return node

    def _lookup(self, parent: DirNode, path_parts: list[str], create_kind: Optional[EntryKind] = None) -> Optional[DirNode]:
        if not parent.is_dir():
            raise Exception("Node is not a directory")

        head, *tail = path_parts
        for node in parent.children:
            if node.name == head:
                if len(tail):
                    return self._lookup(node, tail, create_kind)
                else:
                    return node

        if create_kind:
            for (idx, name) in enumerate(path_parts):
                parent = self.make_node(
                    name,
                    kind = create_kind if idx == len(path_parts) - 1 else EntryKind.DIRECTORY,
                    parent = parent
                )

            return parent

        return None


    def _next_id(self) -> int:
         self.last_entry_id += 1
         return self.last_entry_id


class DirView:
    tree: DirTree
    expanded_dirs: set[Path]
    show_dotfiles: bool
    show_details: bool


    def __init__(self, gen_id: int, root_dir: Path, expanded_dirs: set[Path], show_dotfiles: bool, show_details: bool):
        self.tree = DirTree(gen_id, root_dir)
        self.expanded_dirs = expanded_dirs.copy()
        self.show_dotfiles = show_dotfiles
        self.show_details = show_details

        self.refresh_from_fs()


    def root_dir(self) -> Path:
        return self.tree.root_dir()

    def move_up(self):
        root_dir = self.tree.root_dir()
        if root_dir.parent == root_dir:
            return False

        self.cd(root_dir.parent)
        return True


    def cd(self, new_root: Path):
        self.tree.cd(new_root)
        self.refresh_from_fs()


    def toggle_expand(self, path: Path) -> bool:
        node = self.tree.lookup(path)
        if not node or not node.is_dir():
            return False

        node_abspath = self.tree.root_dir() / path
        if node.is_expanded:
            node.is_expanded = False
            node.children.clear()
            self.expanded_dirs.discard(node_abspath)
        else:
            self._refresh_node_children(node)
            self.expanded_dirs.add(node_abspath)

        return True


    def refresh_from_fs(self):
        self._refresh_node_children(self.tree.root)


    def _refresh_node_children(self, node: DirNode):
        if not node.is_dir():
            raise Exception("Expected directory node")

        def entry_sort_key(entry: os.DirEntry) -> tuple[bool, str]:
            # Sorting by `not is_dir()` puts directories first and everything else
            # second (including broken links, for which both is_dir() and is_file()
            # return false)
            return (not entry_is_dir(entry), entry.name)

        path = self.root_dir() / node.filepath()
        child_entries: list[os.DirEntry] = [entry for entry in os.scandir(path) if self.show_dotfiles or not entry.name.startswith(".")]
        child_entries.sort(key = entry_sort_key)

        node.children.clear()
        for entry in child_entries:
            child_node = self._make_node(entry, parent=node)
            node_path = path / child_node.name
            if child_node.is_dir() and node_path in self.expanded_dirs:
                self._refresh_node_children(child_node)

        node.is_expanded = True


    def _make_node(self, entry: os.DirEntry, parent: DirNode) -> DirNode:
        entry_path = Path(entry.path)
        link_target = None
        link_status = LinkStatus.NO_LINK
        is_executable = False

        stat_itself = entry.stat(follow_symlinks=False)
        stat_target = stat_itself
        if entry.is_symlink():
            link_status = LinkStatus.UNKNOWN
            try:
                link_target = entry_path.readlink()
                if (entry_path.parent / link_target).exists():
                    link_status = LinkStatus.GOOD
                    stat_target = entry.stat(follow_symlinks=True)
                else:
                    link_status = LinkStatus.BROKEN
            except PermissionError:
                pass

        if entry.is_file():
            # Symlinks are always executable (in fact, st_mode has all bits set) so we need to use `stat_target`
            is_executable = bool(stat_target.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))

        details = None
        if self.show_details:
            details = NodeDetails(
                user = pwd.getpwuid(stat_itself.st_uid).pw_name,
                group = grp.getgrgid(stat_itself.st_gid).gr_name,
                mode = stat.filemode(stat_itself.st_mode),
                size = size_to_human(stat_target.st_size),
                mtime = datetime.fromtimestamp(stat_itself.st_mtime).isoformat(" ", "minutes"),
            )

        return self.tree.make_node(
            name = entry.name,
            kind = EntryKind.DIRECTORY if entry_is_dir(entry) else EntryKind.FILE,
            parent = parent,
            is_executable = is_executable,
            link_target = link_target,
            link_status = link_status,
            details = details,
        )


    def _try_get_entry(self, entries: list[DirNode], path_parts: list[str]) -> Optional[DirNode]:
        head, *tail = path_parts
        for entry in entries:
            if entry.name == head:
                if len(tail):
                    if entry.is_dir():
                        return self._try_get_entry(entry.children, tail)
                    else:
                        return None
                else:
                    return entry

        return None


def entry_is_dir(entry: os.DirEntry) -> bool:
    # We may get a permission error if `entry` is a symlink and points
    # to a directory that we don't have access to. In this case, we treat
    # `entry` as a file
    try:
        return entry.is_dir()
    except PermissionError:
        pass

    return False


def size_to_human(size: int) -> str:
    cur_size = float(size)
    units = ["", "K", "M", "G", "T"]
    for i in range(len(units)):
        if cur_size < 1024:
            break

        cur_size /= 1024

    return f"{cur_size:.1f}{units[i]}"
