from dataclasses import dataclass


@dataclass
class Variable:
    variable: str
    column: str


@dataclass
class Shift:
    shift: str
    variables: list[Variable]


@dataclass
class Sample:
    era: str
    sample: str
    channel: str
    main_files: list[str]
    friend_files: dict[str, list[str]]
    shifts: list[Shift]
