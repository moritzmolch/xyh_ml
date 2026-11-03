from dataclasses import dataclass


@dataclass
class Variable:
    name: str
    column: str


@dataclass
class Shift:
    name: str
    variables: list[Variable]


@dataclass
class Sample:
    era: str
    sample: str
    channel: str
    main_files: list[str]
    friend_files: dict[str, list[str]]
    shifts: list[Shift]


@dataclass
class FileSlice:
    """Entry range ``[entry_start, entry_stop)`` of one main file and the
    corresponding friend files."""

    main_file: str
    friend_files: dict[str, str]
    entry_start: int
    entry_stop: int

    @property
    def num_entries(self) -> int:
        return self.entry_stop - self.entry_start


@dataclass
class Chunk:
    """Contiguous range of events of a sample, possibly spanning several
    files."""

    index: int
    slices: list[FileSlice]

    @property
    def num_entries(self) -> int:
        return sum(s.num_entries for s in self.slices)
