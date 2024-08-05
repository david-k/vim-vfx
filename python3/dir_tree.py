from enum import Enum
from pathlib import Path
from pprint import pprint

from dataclasses import dataclass, field
from typing import TextIO, Optional


#===============================================================================
class EntryKind(Enum):
    # Order important because used for sorting
    DIRECTORY = 1
    FILE = 2


@dataclass
class DirEntry:
    id: int
    name: str
    kind: EntryKind
    is_link: bool

    # For directories
    is_expanded: bool = False
    children: list["DirEntry"] = field(default_factory=list)

    def is_dir(self):
        return self.kind == EntryKind.DIRECTORY

    def is_file(self):
        return self.kind == EntryKind.FILE


class DirTree:
    root_dir: Path
    entries: list[DirEntry]
    expanded_dirs: set[Path]
    show_dotfiles: bool
    next_entry_id: int

    def __init__(self, root_dir: Path, expanded_dirs: set[Path], show_dotfiles: bool):
        self.entries = []
        self.expanded_dirs = expanded_dirs.copy()
        self.show_dotfiles = show_dotfiles
        self.next_entry_id = 1

        self.cd(root_dir)
        self.refresh_from_fs()


    def move_up(self):
        if self.root_dir.parent == self.root_dir:
            return False

        self.cd(self.root_dir.parent)
        return True


    def cd(self, new_root: Path):
        self.root_dir = new_root.absolute()
        self.refresh_from_fs()


    def toggle_expand(self, path: Path) -> bool:
        entry = self._try_get_entry(self.entries, list(path.parts))
        if not entry or not entry.is_dir():
            return False

        entry_abspath = self.root_dir / path
        if entry.is_expanded:
            entry.is_expanded = False
            entry.children.clear()
            self.expanded_dirs.discard(entry_abspath)
        else:
            entry.is_expanded = True
            entry.children = self._load_entries(entry_abspath)
            self.expanded_dirs.add(entry_abspath)

        return True


    def refresh_from_fs(self):
        self.entries = self._load_entries(self.root_dir)


    def _load_entries(self, path: Path):
        entries = [self._entry_from_path(p) for p in path.iterdir() if self.show_dotfiles or not p.name.startswith(".")]
        entries.sort(key = lambda entry: (entry.kind.value, entry.name))
        for entry in entries:
            entry_path = path / entry.name
            if entry.is_dir() and entry_path in self.expanded_dirs:
                entry.is_expanded = True
                entry.children = self._load_entries(entry_path)

        return entries


    def _next_id(self) -> int:
         id = self.next_entry_id
         self.next_entry_id += 1
         return id


    def _entry_from_path(self, p: Path) -> DirEntry:
        return DirEntry(
            id = self._next_id(),
            name = p.name,
            kind = EntryKind.DIRECTORY if p.is_dir() else EntryKind.FILE,
            is_link = p.is_symlink()
        )


    def _try_get_entry(self, entries: list[DirEntry], path_parts: list[str]) -> Optional[DirEntry]:
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


def print_tree(out: TextIO, tree: DirTree):
    print_entries(out, tree.entries, 0)


def print_entries(out: TextIO, entries: list[DirEntry], indent: int):
    for entry in entries:
        out.write(indent * "\t" + escape_if_needed(entry.name))
        if entry.kind == EntryKind.DIRECTORY:
            out.write("/")
        out.write("\n")

        if len(entry.children):
            print_entries(out, entry.children, indent + 1)



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


def unescape(filename: str) -> str:
    if filename[0] == "'":
        if filename[-1] != "'":
            raise Exception("Missing closing quote in filename: " + filename)

        filename = filename[1:-1]
        return filename.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')

    return filename



#===============================================================================
def test():
    #dtree = DirTree(Path.cwd(), {Path.cwd() / "python3"})
    dtree = DirTree(Path("/home/david"), {})

    import io
    with io.StringIO() as out:
        print_tree(out, dtree)
        out.seek(0, io.SEEK_SET)
        print(out.readlines())

if __name__ == "__main__":
    test()
