import os
import pwd
import grp
import stat
import bisect
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, NewType, TextIO, Callable, Any

from config import Config
import log


# Directory tree
#===============================================================================
@dataclass
class NodeIDNew:
    pass

@dataclass(frozen=True)
class NodeIDKnown:
    seq_no: int

NodeID = NodeIDNew | NodeIDKnown


class NodeKind(Enum):
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

    def copy(self):
        return NodeDetails(
            user = self.user,
            group = self.group,
            mode = self.mode,
            size = self.size,
            mtime = self.mtime,
        )


@dataclass
class DirNode:
    id: NodeID
    name: str # Only the root node is allowed to contain path separators
    kind: NodeKind
    parent: Optional["DirNode"] = None
    details: Optional[NodeDetails] = None
    error: Optional[str] = None
    fs_error: bool = False

    link_status: LinkStatus = LinkStatus.NO_LINK
    link_target: Optional[Path] = None

    # For files:
    is_executable: bool = False

    # For directories:
    is_expanded: bool = False # Even if False the DirNode may still contain some children (added via DirTree.lookup_or_create_existing())
    children: list["DirNode"] = field(default_factory=list)

    line_no: Optional[int] = None

    def copy_without_children(self, parent: "DirNode") -> "DirNode":
        return DirNode(
            id = self.id,
            name = self.name,
            kind = self.kind,
            parent = parent,
            details = None if self.details is None else self.details.copy(),
            error = self.error,
            link_status = self.link_status,
            link_target = self.link_target,
            is_executable = self.is_executable,
            is_expanded = self.is_expanded,
            line_no = self.line_no,
        )

    def is_new(self) -> bool:
        return isinstance(self.id, NodeIDNew)

    def is_known(self) -> bool:
        return isinstance(self.id, NodeIDKnown)

    def is_dir(self) -> bool:
        return self.kind == NodeKind.DIRECTORY

    def is_file(self) -> bool:
        return self.kind == NodeKind.FILE

    def is_root(self) -> bool:
        return not bool(self.parent)

    # Returns path relative to root
    def filepath(self) -> Path:
        if self.parent:
            return self.parent.filepath() / self.name

        return Path()

    def filepath_str(self) -> str:
        filepath = str(self.filepath())
        return filepath + "/" if self.is_dir() else filepath

    def abs_filepath(self) -> Path:
        if self.parent:
            return self.parent.abs_filepath() / self.name

        return Path(self.name)

    def add_error(self, msg: str):
        if self.error is None:
            self.error = msg
        else:
            self.error += " | " + msg

    def get_child_idx(self, id: NodeID) -> int:
        for idx, node in enumerate(self.children):
            if node.id == id:
                return idx

        raise Exception("No child found that has given ID")


    def remove_child(self, id: NodeID) -> "DirNode":
        node = self.children.pop(self.get_child_idx(id))
        node.parent = None
        return node


    def seq_no(self) -> int:
        match self.id:
            case NodeIDKnown(seq_no):
                return seq_no

            case _:
                raise Exception("Node has no seq no")


    def print(self, indent: int):
        match self.id:
            case NodeIDNew():
                print(f"{'':<{indent}}NewNode(\"{self.name}\")")
            case NodeIDKnown(seq_no):
                print(f"{'':<{indent}}KnownNode({seq_no}, \"{self.name}\", expanded={self.is_expanded}, details={self.details is not None})")

        for child in self.children:
            child.print(indent + 4)


    def gather_errors(self) -> list[tuple["DirNode", str]]:
        errors: list[tuple["DirNode", str]] = []

        def gather_errors_rec(node: DirNode):
            if node.error is not None:
                errors.append((self, node.error))

            for child in node.children:
                gather_errors_rec(child)

        gather_errors_rec(self)
        return errors


def node_sort_key(node: DirNode) -> tuple[bool, str]:
    return (not node.is_dir(), node.name)


def next_id_default(tree: "DirTree", node_abs_path: Path) -> NodeIDKnown:
    return  tree.next_node_id()


# While there are pending modifications, the NodeID for a file in the filesystem must never change.
# For this reason:
# - We always load all nodes in a directory to ensure that e.g. toggling dotfiles does not change
#   any IDs
# - When collapsing a directory, we never unload its child nodes
# - Navigating the directory tree (i.e., changing the root directory,
#   expanding/collapsing a node) while there are pending modification is only allowed if in the new
#   tree all the modifications are still visible.
class DirTree:
    root: DirNode
    has_details: bool
    has_dotfiles: bool
    next_seq_no: int = 0

    def __init__(self, root_dir: Path, has_details: bool, has_dotfiles: bool):
        self.has_details = has_details
        self.has_dotfiles = has_dotfiles
        self.root = DirNode(
            id = self.next_node_id(),
            name = str(root_dir),
            kind = NodeKind.DIRECTORY,
            is_expanded = False,
        )


    def root_dir(self) -> Path:
        return Path(self.root.name)


    def add_path(self, new_root: Path):
        old_root = self.root_dir()

        assert new_root.is_absolute()
        # REVISIT: Use new_root.resolve() to ensure symlinks don't mess up comparison with old_root?
        if new_root == old_root:
            return

        if new_root.is_relative_to(old_root):
            relative_path = new_root.relative_to(old_root)
            self.lookup_or_create_existing(relative_path, NodeKind.DIRECTORY)
        else:
            old_root_node = self.root
            old_root_node.name = Path(old_root_node.name).name

            self.root = DirNode(
                id = self.next_node_id(),
                name = str(os.path.commonpath([old_root, new_root])),
                kind = NodeKind.DIRECTORY,
                is_expanded = False,
            )

            # Insert old_root_node below self.root
            old_root_parent = self.lookup_or_create_existing(old_root.parent, NodeKind.DIRECTORY)
            old_root_parent.children.append(old_root_node)
            old_root_node.parent = old_root_parent

            # Create new node for new_root
            self.lookup_or_create_existing(new_root.relative_to(self.root.abs_filepath()), NodeKind.DIRECTORY)


    def cd(self, new_root: Path, next_id_cb: Callable[["DirTree", Path], NodeIDKnown] = next_id_default):
        old_root = self.root_dir()

        assert new_root.is_absolute()
        # REVISIT: Use new_root.resolve() to ensure symlinks don't mess up comparison with old_root?
        if new_root == old_root:
            return

        if new_root.is_relative_to(old_root):
            relative_path = new_root.relative_to(old_root)
            self.root = self.lookup_or_create_existing(relative_path, NodeKind.DIRECTORY, next_id_cb)
            self.root.name = str(new_root)
            self.root.parent = None

        elif old_root.is_relative_to(new_root):
            old_root_node = self.root
            old_root_node.name = Path(old_root_node.name).name

            self.root = DirNode(
                id = next_id_cb(self, new_root),
                name = str(new_root),
                kind = NodeKind.DIRECTORY,
                is_expanded = False,
            )

            # Insert old_root_node below self.root
            old_root_parent = self.lookup_or_create_existing(old_root.parent, NodeKind.DIRECTORY, next_id_cb)
            old_root_parent.children.append(old_root_node)
            old_root_node.parent = old_root_parent

        else:
            self.root = DirNode(
                id = next_id_cb(self, new_root),
                name = str(new_root),
                kind = NodeKind.DIRECTORY,
                is_expanded = False,
            )


    def is_expanded_at(self, p: Path) -> bool:
        node = self.try_lookup(p)
        if node is None:
            return False

        return node.is_expanded


    def lookup(self, p: Path) -> DirNode:
        node = self.try_lookup(p)
        if not node:
            raise Exception("Node not found")

        return node


    def try_lookup(self, p: Path) -> Optional[DirNode]:
        parts = list(self._make_relative(p).parts)
        if len(parts) == 0:
            return self.root

        return self._lookup(self.root, parts)


    def lookup_or_create_new(self, p: Path, kind: NodeKind) -> DirNode:
        parts = list(self._make_relative(p).parts)
        if len(parts) == 0:
            assert kind == NodeKind.DIRECTORY
            return self.root

        node = self._lookup(self.root, parts, "create_new", kind); assert node is not None
        return node


    def lookup_or_create_existing(self, p: Path, kind: NodeKind, next_id_cb: Callable[["DirTree", Path], NodeIDKnown] = next_id_default) -> DirNode:
        parts = list(self._make_relative(p).parts)
        if len(parts) == 0:
            assert kind == NodeKind.DIRECTORY
            return self.root

        node = self._lookup(self.root, parts, "create_existing", kind, next_id_cb); assert node is not None
        return node


    def next_node_id(self) -> NodeIDKnown:
        self.next_seq_no += 1
        return NodeIDKnown(self.next_seq_no - 1)


    def print(self):
        for child in self.root.children:
            child.print(0)


    def _lookup(
        self,
        parent: DirNode,
        path_parts: list[str],
        if_missing: str = "stop", # "create_new", "create_existing"
        create_kind: Optional[NodeKind] = None,
        next_id_cb: Callable[["DirTree", Path], NodeIDKnown] = next_id_default,
    ) -> Optional[DirNode]:

        if not parent.is_dir():
            return None

        head, *tail = path_parts
        for node in parent.children:
            if node.name == head:
                if len(tail):
                    return self._lookup(node, tail, if_missing, create_kind, next_id_cb)
                else:
                    return node

        if if_missing in ["create_new", "create_existing"]:
            assert create_kind is not None
            line_no = None
            if len(parent.children) and parent.children[-1].line_no is not None:
                line_no = parent.children[-1].line_no + 1
            elif parent.line_no is not None:
                line_no = parent.line_no + 1

            create_new = if_missing == "create_new"
            for (idx, name) in enumerate(path_parts):
                new_node = DirNode(
                    NodeIDNew() if create_new else next_id_cb(self, parent.abs_filepath() / name),
                    name,
                    kind = create_kind if idx == len(path_parts) - 1 else NodeKind.DIRECTORY,
                    parent = parent,
                    line_no = line_no,
                    is_expanded = create_new,
                )
                parent.children.append(new_node)
                parent = new_node

            return parent

        return None


    def _make_relative(self, p: Path) -> Path:
        if p.is_absolute() and p.is_relative_to(self.root_dir()):
            return p.relative_to(self.root_dir())

        return p


@dataclass
class PathInfoNode:
    is_expanded: bool = False
    num_expanded_children: int = 0 # Number of (transitive) children that are expanded
    children: dict[str, "PathInfoNode"] = field(default_factory=dict)

    def get_expanded_dir_count(self):
        if self.is_expanded:
            return self.num_expanded_children + 1

        return self.num_expanded_children


    def clone(self) -> "PathInfoNode":
        return PathInfoNode(
            self.is_expanded,
            self.num_expanded_children,
            {child_name: child_node.clone() for child_name, child_node in self.children.items()},
        )


    def is_expanded_at(self, path: Path) -> bool:
        if node := self.try_lookup(path):
            return node.is_expanded

        return False


    def lookup_or_create(self, path: Path) -> "PathInfoNode":
        assert path.is_absolute()
        return self._lookup_or_create(list(path.parts)[1:])


    def lookup(self, path: Path) -> "PathInfoNode":
        node = self.try_lookup(path)
        if node is None:
            raise Exception("PathInfoNode not found")

        return node

    def try_lookup(self, path: Path) -> Optional["PathInfoNode"]:
        assert path.is_absolute()
        return self._try_lookup(list(path.parts)[1:])


    def _lookup_or_create(self, path_parts: list[str]) -> "PathInfoNode":
        if len(path_parts) == 0:
            return self

        head, *tail = path_parts
        child = self.children.setdefault(head, PathInfoNode())
        return child._lookup_or_create(tail)

    def _try_lookup(self, path_parts: list[str]) -> Optional["PathInfoNode"]:
        if len(path_parts) == 0:
            return self

        head, *tail = path_parts
        if child := self.children.get(head):
            return child._try_lookup(tail)

        return None


    def set_expanded(self, path: Path, expanded: bool):
        assert path.is_absolute()
        self._set_expanded(list(path.parts)[1:], expanded)


    def _set_expanded(self, path_parts: list[str], expanded: bool) -> bool:
        if len(path_parts) == 0:
            if self.is_expanded != expanded:
                self.is_expanded = expanded
                return True
        else:
            head, *tail = path_parts
            child = self.children.setdefault(head, PathInfoNode())
            if child._set_expanded(tail, expanded):
                if expanded:
                    self.num_expanded_children += 1
                else:
                    assert self.num_expanded_children > 0
                    self.num_expanded_children -= 1

                return True

        return False


    def set_node(self, path: Path, new_node: "PathInfoNode"):
        self._set_node(list(path.parts)[1:], new_node)


    def _set_node(self, path_parts: list[str], new_node: "PathInfoNode"):
        if len(path_parts) == 0:
            self.is_expanded = new_node.is_expanded
            self.num_expanded_children = new_node.num_expanded_children
            self.children = new_node.children
        else:
            head, *tail = path_parts
            self.children.setdefault(head, PathInfoNode())._set_node(tail, new_node)
            self.num_expanded_children += new_node.get_expanded_dir_count()


    def print(self):
        self._print()

    def _print(self, level: int = 0):
        for child_name, child_node in self.children.items():
            expanded_char = "+" if child_node.is_expanded else "-"
            print(" "*2*level + expanded_char + " " + child_name + "(" + str(child_node.num_expanded_children) + ")")
            child_node._print(level + 1)



def get_expanded_dirs(tree: DirTree) -> PathInfoNode:
    assert tree.root_dir().is_absolute()

    root_node = PathInfoNode()
    root_node.set_node(tree.root_dir(), get_expanded_dirs_for_node(tree.root))

    return root_node


def get_expanded_dirs_for_node(node: DirNode) -> PathInfoNode:
    assert node.is_dir() and not node.is_new()

    path_node = PathInfoNode(
        is_expanded = node.is_expanded,
        num_expanded_children = 0,
        children = {},
    )
    if path_node.is_expanded:
        for child in node.children:
            if not child.is_new() and child.is_dir() and node.is_expanded:
                child_path_node = get_expanded_dirs_for_node(child)
                path_node.num_expanded_children += child_path_node.get_expanded_dir_count()
                path_node.children[child.name] = child_path_node

    return path_node


# def get_expanded_dirs(tree: DirTree) -> set[Path]:
#     def get_expanded_dirs_rec(node: DirNode, expanded_dirs: set[Path]):
#         if not node.is_new() and node.is_dir() and node.is_expanded:
#             expanded_dirs.add(tree.root_dir() / node.filepath())
#             for child in node.children:
#                 get_expanded_dirs_rec(child, expanded_dirs)

#     expanded_dirs: set[Path] = set()
#     get_expanded_dirs_rec(tree.root, expanded_dirs)
#     return expanded_dirs


def get_nodes_by_id(tree: DirTree) -> tuple[dict[NodeIDKnown, list[DirNode]], list[DirNode]]:
    def get_nodes_by_id_rec(
            node: DirNode,
            nodes_by_id: dict[NodeIDKnown, list[DirNode]],
            new_nodes: list[DirNode]
        ):
        match node.id:
            case NodeIDNew():
                new_nodes.append(node)
            case NodeIDKnown():
                if not node.is_root():
                    nodes_by_id.setdefault(node.id, []).append(node)

        if node.is_dir() and node.is_expanded:
            for child in node.children:
                get_nodes_by_id_rec(child, nodes_by_id, new_nodes)

    nodes_by_id: dict[NodeIDKnown, list[DirNode]] = {}
    new_nodes: list[DirNode] = []
    get_nodes_by_id_rec(tree.root, nodes_by_id, new_nodes)
    return nodes_by_id, new_nodes


def get_nodes_in_expanded_dirs(root: DirNode, path_node: Optional[PathInfoNode], include_dotfiles: bool) -> dict[NodeIDKnown, DirNode]:
    def get_nodes_in_expanded_dirs_rec(
            node: DirNode,
            path_node: Optional[PathInfoNode],
            nodes_by_id: dict[NodeIDKnown, DirNode],
        ):
        if node.is_dir():
            include_children = node.is_expanded and (path_node is None or path_node.is_expanded)
            traverse_children = path_node is None or (path_node is not None and path_node.is_expanded)

            if include_children or traverse_children:
                for child in node.children:
                    if child.name.startswith(".") and not include_dotfiles:
                        continue

                    if include_children:
                        match child.id:
                            case NodeIDNew():
                                pass
                            case NodeIDKnown():
                                nodes_by_id[child.id] = child

                    if traverse_children:
                        child_path_info = get_child(path_node, child.name)
                        get_nodes_in_expanded_dirs_rec(child, child_path_info, nodes_by_id)

    nodes_by_id: dict[NodeIDKnown, DirNode] = {}
    get_nodes_in_expanded_dirs_rec(root, path_node, nodes_by_id)
    return nodes_by_id


# CREATE: orig_node=None, new_nodes=[NEW_NODE], delete_orig=...
# DELETE: orig_node=ORIG_NODE, new_nodes=[], delete_orig=True
# MOVE:   orig_node=ORIG_NODE, new_nodes=[NEW_NODE1, NEW_NODE2, ...], delete_orig=True
#         (if there are multiple new nodes then this is a combined move/copy)
# COPY:   orig_node=ORIG_NODE, new_nodes=[NEW_NODE1, NEW_NODE2, ...], delete_orig=False
@dataclass
class Modification:
    orig_node: Optional[DirNode]
    new_nodes: list[DirNode]
    delete_orig: bool

    def is_create(self) -> bool:
        return self.orig_node is None

    def has_delete(self) -> bool:
        return self.delete_orig and len(self.new_nodes) != 1

    def get_ops(self) -> list[dict]:
        ops = []
        if self.has_delete():
            assert self.orig_node is not None
            ops.append({"kind": "delete", "node": self.orig_node})

        if self.is_create():
            new_node = self.new_nodes[0]
            ops.append({"kind": "create", "node": new_node})
        elif self.orig_node and len(self.new_nodes):
            kind = "move" if self.delete_orig and len(self.new_nodes) == 1 else "copy"
            for new_node in self.new_nodes:
                ops.append({"kind": kind, "src_node": self.orig_node, "dest_node": new_node})

        return ops

    def print(self):
        for op in self.get_ops():
            print(op_to_str(op))


def op_to_str(op: dict) -> str:
    if op["kind"] == "delete":
        return f"DELETE {op['node'].filepath_str()}"
    elif op["kind"] == "create":
        return f"CREATE {op['node'].filepath_str()}"
    elif op["kind"] in ["move", "copy"]:
        return f"{op['kind'].upper()} {op['src_node'].filepath_str()} -> {op['dest_node'].filepath_str()}"

    assert False


def get_ids_from(tree: DirTree) -> Callable[[DirTree, Path], NodeIDKnown]:
    def get_id_from_cb(_: DirTree, node_path: Path) -> NodeIDKnown:
        node_id = tree.lookup(node_path).id
        assert isinstance(node_id, NodeIDKnown)
        return node_id

    return get_id_from_cb


def update_buf_tree_from_base(
    buf_tree: DirTree,
    buf_root_dir: Path,
    base_tree: DirTree,
    path_info: PathInfoNode,
    mods_by_id: dict[NodeIDKnown, Modification]
):
    fs_nodes_by_id = get_nodes_in_expanded_dirs(
        base_tree.root,
        path_node=None,
        include_dotfiles=True
    )

    buf_tree.cd(buf_root_dir, next_id_cb = get_ids_from(base_tree))
    _update_buf_node_from_base(
        buf_tree.root,
        base_tree.lookup(buf_tree.root_dir()),
        buf_tree.has_dotfiles,
        path_info.try_lookup(buf_tree.root_dir()),
        fs_nodes_by_id,
        mods_by_id
    )


# Refresh `buf_node` with new information from the filesystem as represented by `fs_node`.
#
# Q: Why do we need to pass `path_info` to check whether the directory is expanded? Why can't we
#    just use `buf_node.is_expanded`?
# A: This is needed when expanding a directory that contains subdirectories. In this case we need
#    `is_expanded` to determine which of these subdirectories to expand. (We can't use
#    `buf_node.is_expanded` in this case because these `buf_node`s don't exist yet.)
def _update_buf_node_from_base(
    buf_node: DirNode,
    fs_node: DirNode,
    show_dotfiles: bool,
    path_info: Optional[PathInfoNode],
    fs_nodes_by_id: dict[NodeIDKnown, DirNode],
    mods_by_id: dict[NodeIDKnown, Modification],
):
    assert buf_node.id == fs_node.id
    assert Path(buf_node.name).name == Path(fs_node.name).name

    buf_node.kind = fs_node.kind
    buf_node.details = fs_node.details.copy() if fs_node.details else None
    buf_node.link_status = fs_node.link_status
    buf_node.link_target = fs_node.link_target
    buf_node.is_executable = fs_node.is_executable

    if buf_node.is_dir():
        buf_node.is_expanded = bool(path_info and path_info.is_expanded)
        if buf_node.is_expanded:
            assert fs_node.is_expanded
            # Remove child nodes form buf_node that do not exist in the filesystem anymore
            # (We search in `fs_nodes_by_id` and not only in `fs_node.children` in case the file has
            # been copied/moved in the buf tree.)
            idx = 0
            while idx < len(buf_node.children):
                buf_child_ = buf_node.children[idx]
                if buf_child_.is_new():
                    idx += 1
                    continue

                assert isinstance(buf_child_.id, NodeIDKnown)
                fs_child_ = fs_nodes_by_id.get(buf_child_.id)

                if fs_child_ is None or (buf_child_.name.startswith(".") and not show_dotfiles):
                    buf_node.children.pop(idx)
                    continue

                if fs_child_:
                    _update_buf_node_from_base(buf_child_, fs_child_, show_dotfiles, get_child(path_info, buf_child_.name), fs_nodes_by_id, mods_by_id)

                idx += 1


            # Insert the child nodes from fs_node into buf_node in sorted
            # order without changing the order of children of buf_node
            buf_children_by_id = {child.id: child for child in buf_node.children if not child.is_new()}
            for fs_child in fs_node.children:
                if fs_child.name.startswith(".") and not show_dotfiles:
                    continue

                if fs_child.id not in buf_children_by_id:
                    assert isinstance(fs_child.id, NodeIDKnown)
                    mod = mods_by_id.get(fs_child.id)
                    # Only insert the child node if it has not been deleted
                    if not mod or not mod.delete_orig:
                        buf_child = fs_child.copy_without_children(parent=buf_node)
                        _update_buf_node_from_base(buf_child, fs_child, show_dotfiles, get_child(path_info, buf_child.name), fs_nodes_by_id, mods_by_id)

                        buf_child_idx = bisect.bisect(buf_node.children, node_sort_key(fs_child), key = node_sort_key)
                        buf_node.children.insert(buf_child_idx, buf_child)

        else:
            buf_node.children.clear()


# Compute node changes
#===============================================================================
def compute_changes(base_tree: DirTree, buf_tree: DirTree) -> list[Modification]:
    base_root = base_tree.lookup(buf_tree.root_dir())

    buf_nodes_by_id, new_nodes = get_nodes_by_id(buf_tree)
    visible_base_nodes_by_id = get_nodes_in_expanded_dirs(
        base_root,
        get_expanded_dirs_for_node(buf_tree.root),
        base_tree.has_dotfiles
    )
    deleted_nodes = [visible_base_nodes_by_id[deleted_id]
                     for deleted_id in (visible_base_nodes_by_id.keys() - buf_nodes_by_id.keys())]

    modifications: list[Modification] = []

    # CREATE modifications
    for new_node in new_nodes:
        modifications.append(Modification(
            orig_node=None,
            new_nodes=[new_node],
            delete_orig=False
        ))

    # DELETE modifications
    for deleted_node in deleted_nodes:
        modifications.append(Modification(
            orig_node=deleted_node,
            new_nodes=[],
            delete_orig=True
        ))

    # MOVE/COPY modifications
    all_base_nodes_by_id = get_nodes_in_expanded_dirs(base_root, path_node=None, include_dotfiles=True)
    for buf_nodes in buf_nodes_by_id.values():
        node_id = buf_nodes[0].id
        assert isinstance(node_id, NodeIDKnown)
        base_node = all_base_nodes_by_id.get(node_id)
        if base_node is None:
            raise Exception(f"compute_changes: Invalid node ID: {node_id}")

        modification = Modification(
            orig_node=base_node,
            new_nodes=[],
            delete_orig=True,
        )
        for buf_node in buf_nodes:
            if buf_node.kind != base_node.kind:
                raise Exception("Changing node type not supported yet")

            if buf_node.link_target != base_node.link_target:
                raise Exception("Changing link target not supported yet")

            if buf_node.name == base_node.name and _have_same_parents(buf_node, base_node):
                modification.delete_orig = False
            else:
                modification.new_nodes.append(buf_node)

        if len(modification.new_nodes):
            modifications.append(modification)

    return modifications


def _have_same_parents(a: DirNode, b: DirNode) -> bool:
    if a.parent is None or b.parent is None:
        return a.parent == b.parent

    return a.parent.id == b.parent.id

def get_mods_by_id(mods: list[Modification]) -> dict[NodeIDKnown, Modification]:
    return {mod.orig_node.id: mod for mod in mods if mod.orig_node and isinstance(mod.orig_node.id, NodeIDKnown)}


# Filesystem operations
#===============================================================================
def get_child(path_node: Optional[PathInfoNode], child: str) -> Optional[PathInfoNode]:
    if path_node is None:
        return None

    return path_node.children.get(child)


class RefreshStats:
    num_file_queries: int = 0
    num_scandir_queries: int = 0
    num_nodes_visited: int = 0


def refresh_node(
    tree: DirTree,
    node: DirNode,
    path_info: Optional[PathInfoNode],
    refresh: bool,
    error_callback: Callable[[str], None],
):
    with log.scope("Refresh node") as scope:
        stats = RefreshStats()
        stats.num_nodes_visited += 1
        _refresh_node(tree, node, path_info, refresh, error_callback, stats)
        scope.info_add(f" (nodes_visited={stats.num_nodes_visited}, fs_queries={stats.num_file_queries+stats.num_scandir_queries}, refresh={refresh})")


def _refresh_node(
    tree: DirTree,
    node: DirNode,
    path_info: Optional[PathInfoNode],
    refresh: bool,
    error_callback: Callable[[str], None],
    stats: RefreshStats,
):
    if node.is_new():
        return

    node_abspath = tree.root_dir() / node.filepath()

    # Newly inserted root nodes don't have their details set so we always load them if they are
    # missing
    if refresh or (node.details is None and tree.has_details):
        stats.num_file_queries += 1
        info = _query_entry_info(node_abspath, load_details = tree.has_details)
        node.is_executable = info["is_executable"]
        node.link_target = info["link_target"]
        node.link_status = info["link_status"]
        node.details = info["details"]
        node.kind = info["kind"]

        if node.is_file():
            node.children.clear()
            node.is_expanded = False

    if node.is_dir():
        refresh_direct_children = (
            (refresh and node.is_expanded) or
            (not node.is_expanded and path_info and path_info.is_expanded)
        )
        if refresh_direct_children:
            nodes_by_name: dict[str, DirNode] = {node.name: node for node in node.children}
            node.children.clear()
            node.is_expanded = True
            try:
                stats.num_scandir_queries += 1
                with os.scandir(node_abspath) as entries:
                    for entry in entries:
                        log.debug(f"ScandirNext {entry.path}")
                        stats.num_scandir_queries += 1
                        child_node = nodes_by_name.get(entry.name)
                        node_is_fresh = child_node is None
                        if child_node is None:
                            stats.num_file_queries += 1
                            info = _query_entry_info(Path(entry.path), tree.has_details)
                            child_node = DirNode(
                                # We use a dummy ID here and only set the real ID after the children have
                                # been sorted. This ensures deterministic assignment of IDs, which makes
                                # testing easier.
                                id = NodeIDKnown(-1),
                                name = info["name"],
                                kind = info["kind"],
                                parent = node,
                                is_executable = info["is_executable"],
                                link_target = info["link_target"],
                                link_status = info["link_status"],
                                details = info["details"],
                            )

                        refresh_child = refresh
                        if node_is_fresh:
                            refresh_child = False

                        _refresh_node(tree, child_node, get_child(path_info, child_node.name), refresh_child, error_callback, stats)
                        node.children.append(child_node)

                    node.children.sort(key=node_sort_key)
                    stats.num_nodes_visited += len(node.children)

                    for child in node.children:
                        if child.seq_no() == -1:
                            child.id = tree.next_node_id()

            except PermissionError as e:
                # Ensure that we don't repeatedly report the same error whenever we refresh the tree
                if not node.fs_error:
                    error_callback(f"No permission: {e.filename}")
                    node.fs_error = True

        elif path_info:
            # Even if we don't need to refresh the direct children of `node` we may still need to
            # refresh its indirect children if `node` contains any (newly) expanded sub-directories
            for child in node.children:
                if not child.is_dir():
                    continue

                child_path_info = get_child(path_info, child.name)
                if child_path_info and child_path_info.get_expanded_dir_count():
                    _refresh_node(tree, child, child_path_info, refresh, error_callback, stats)

            stats.num_nodes_visited += len(node.children)


# def refresh_node(tree: DirTree, node: DirNode, expanded_dirs: Optional[set[Path]], error_callback: Callable[[str], None]):
#     if node.is_new():
#         return

#     node_abspath = tree.root_dir() / node.filepath()
#     info = _query_entry_info(node_abspath, load_details = tree.has_details)
#     node.is_executable = info["is_executable"]
#     node.link_target = info["link_target"]
#     node.link_status = info["link_status"]
#     node.details = info["details"]
#     node.kind = info["kind"]
#     if node.is_file():
#         node.children.clear()
#         node.is_expanded = False

#     if node.is_dir():
#         should_be_expanded = node.is_expanded or (expanded_dirs is not None and node_abspath in expanded_dirs)
#         if should_be_expanded:
#             nodes_by_name: dict[str, DirNode] = {node.name: node for node in node.children}
#             node.children.clear()
#             node.is_expanded = True

#             try:
#                 with os.scandir(node_abspath) as entries:
#                     for entry in entries:
#                         child_node = nodes_by_name.get(entry.name)
#                         if child_node:
#                             refresh_node(tree, child_node, expanded_dirs, error_callback)
#                             node.children.append(child_node)
#                         else:
#                             info = _query_entry_info(Path(entry.path), tree.has_details)
#                             child_node = DirNode(
#                                 # We use a dummy ID here and only set the real ID after the children have
#                                 # been sorted. This ensures deterministic assignment of IDs, which makes
#                                 # testing easier.
#                                 id = NodeIDKnown(-1),
#                                 name = info["name"],
#                                 kind = info["kind"],
#                                 parent = node,
#                                 is_executable = info["is_executable"],
#                                 link_target = info["link_target"],
#                                 link_status = info["link_status"],
#                                 details = info["details"],
#                             )
#                             node.children.append(child_node)

#                     node.children.sort(key=node_sort_key)

#                     for child in node.children:
#                         if child.seq_no() == -1:
#                             child.id = tree.next_node_id()

#             except PermissionError as e:
#                 # Ensure that we don't repeat reporting the same error whenever we refresh the tree
#                 if not node.fs_error:
#                     error_callback(f"No permission: {e.filename}")
#                     node.fs_error = True
#         # else:
#         #     for child in node.children:
#         #         refresh_node(tree, child, expanded_dirs, error_callback)



def _query_entry_info(path: Path, load_details: bool) -> dict:
    log.debug(f"Query {path}")

    link_target = None
    link_status = LinkStatus.NO_LINK
    is_executable = False

    stat_itself = path.stat(follow_symlinks=False) # TODO This can throw
    stat_target = stat_itself
    if stat.S_ISLNK(stat_itself.st_mode): # is symbolic link?
        link_status = LinkStatus.UNKNOWN
        try:
            link_target = path.readlink()
            if (path.parent / link_target).exists():
                link_status = LinkStatus.GOOD
                stat_target = path.stat(follow_symlinks=True)
            else:
                link_status = LinkStatus.BROKEN
        except PermissionError:
            pass

    if stat.S_ISREG(stat_target.st_mode):
        # Symlinks are always executable (in fact, st_mode has all bits set) so we need to use `stat_target`
        is_executable = bool(stat_target.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))

    details = None
    if load_details:
        details = NodeDetails(
            user = pwd.getpwuid(stat_itself.st_uid).pw_name, # TODO This can probably also throw
            group = grp.getgrgid(stat_itself.st_gid).gr_name, # TODO dito
            mode = stat.filemode(stat_itself.st_mode),
            size = size_to_human(stat_target.st_size),
            mtime = datetime.fromtimestamp(stat_itself.st_mtime).isoformat(" ", "minutes"),
        )

    return {
        "name": path.name,
        "kind": NodeKind.DIRECTORY if stat.S_ISDIR(stat_target.st_mode) else NodeKind.FILE,
        "is_executable": is_executable,
        "link_target": link_target,
        "link_status": link_status,
        "details": details,
    }


# Printing
#-----------------------------------------------------------
NUM_DETAIL_COLUMNS = 5

# Number of lines before the directory tree is displayed
NUM_LINES_BEFORE_TREE = 1

def write_tree(config: Config, out: TextIO, view: DirTree, node: DirNode, path_info: Optional[PathInfoNode]) -> int:
    assert node.is_dir()

    out.write(f"> root_id={node.seq_no()} show_details={view.has_details} show_dotfiles={view.has_dotfiles} | {node.abs_filepath()}\n")
    assert NUM_LINES_BEFORE_TREE == 1

    column_widths = [0] * (NUM_DETAIL_COLUMNS + 1)
    column_widths[-1] = len(str(min(view.next_seq_no-1, 0)))

    details_width = 0
    if view.has_details:
        compute_column_widths(node.children, column_widths)
        for i in range(NUM_DETAIL_COLUMNS):
            details_width += column_widths[i]
        details_width += (NUM_DETAIL_COLUMNS-1) # Spaces between columns
        details_width += 2 # Opening and closing brackets
        details_width += 1 # Space after closing bracket

    path_info_children = None if path_info is None else path_info.children
    write_entries(config, out, node.children, column_widths, view.has_details, view.has_dotfiles, path_info_children)
    return details_width


NUM_SPACES_IGNORED_AFTER_DETAILS = 1

def write_entries(
    config: Config,
    out: TextIO,
    entries: list[DirNode],
    column_widths: list[int],
    show_details: bool,
    show_dotfiles: bool,
    path_infos: Optional[dict[str, PathInfoNode]],
    indent: int = 0
):
    max_id_width = column_widths[-1]
    for entry in entries:
        if entry.name.startswith(".") and not show_dotfiles:
            continue

        path_info = None if path_infos is None else path_infos.get(entry.name)
        is_expanded = entry.is_expanded and (path_info is None or path_info.is_expanded)

        # We put one space after the details, which is ignored when computing indentation
        assert NUM_SPACES_IGNORED_AFTER_DETAILS == 1
        if show_details:
            c1_width = column_widths[0]
            c2_width = column_widths[1]
            c3_width = column_widths[2]
            c4_width = column_widths[3]
            c5_width = column_widths[4]
            if entry.details is None:
                out.write(f" {' ':<{c1_width}} {' ':<{c2_width}} {' ':<{c3_width}} {' ':>{c4_width}} {' ':<{c5_width}}  ")
            else:
                d = entry.details
                # ATTENTION: This must be kept in sync with the computation of `details_width` in `write_tree()` and with `compute_column_widths()`
                out.write(f"[{d.mode:<{c1_width}} {d.user:<{c2_width}} {d.group:<{c3_width}} {d.size:>{c4_width}} {d.mtime:<{c5_width}}] ")

        out.write(indent * config.indent_width * " ")

        match entry.id:
            case NodeIDNew():
                out.write(" ")

            case NodeIDKnown(seq_no):
                node_state = "|"
                if entry.is_dir():
                    if is_expanded:
                        node_state = "-"
                    else:
                        node_state = "+"

                out.write(f"{node_state}{seq_no:0{max_id_width}}") # TODO Also need a max_width for gen id
                if entry.is_executable:
                    out.write("x")
                else:
                    out.write("_")


        out.write(" ")
        while True:
            out.write(escape_if_needed(str(entry.name)))
            if next_inline_entry := continue_inline_path(entry):
                out.write("/")
                entry = next_inline_entry
            else:
                break

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

        if is_expanded:
            path_info_children = None if path_info is None else path_info.children
            write_entries(config, out, entry.children, column_widths, show_details, show_dotfiles, path_info_children, indent + 1)


def continue_inline_path(node: DirNode) -> Optional[DirNode]:
    if (
        node.is_new() and
        node.is_dir() and
        node.link_status == LinkStatus.NO_LINK and
        len(node.children) == 1
    ):
        child = node.children[0]
        if node.line_no is not None and child.line_no == node.line_no:
            return child

    return None


def compute_column_widths(nodes: list[DirNode], widths: list[int]):
    for node in nodes:
        if node.details:
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


def size_to_human(size: int) -> str:
    cur_size = float(size)
    units = ["", "K", "M", "G", "T"]
    for i in range(len(units)):
        if cur_size < 1024:
            break

        cur_size /= 1024

    return f"{cur_size:.1f}{units[i]}"
