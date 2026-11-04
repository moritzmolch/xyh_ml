from dataclasses import dataclass
from typing import Any


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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sample":
        """Create a sample from its dictionary representation (e.g. from
        ``dataclasses.asdict`` or the metadata JSON file), converting the
        nested shifts and variables to dataclass objects."""
        return cls(
            era=d["era"],
            sample=d["sample"],
            channel=d["channel"],
            main_files=list(d["main_files"]),
            friend_files={k: list(v) for k, v in d["friend_files"].items()},
            shifts=[
                Shift(
                    name=s["name"],
                    variables=[Variable(**v) for v in s["variables"]],
                )
                for s in d["shifts"]
            ],
        )


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
