from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass
class IntentCsvData:
    utterance: str
    intent: str
    label: int


class SuperEnum(Enum):
    @classmethod
    def to_dict(cls):
        """Returns a dictionary representation of the enum."""
        return {e.name: e.value for e in cls}

    @classmethod
    def keys(cls):
        """Returns a list of all the enum keys."""
        return cls._member_names_

    @classmethod
    def values(cls):
        """Returns a list of all the enum values."""
        return list(cls._value2member_map_.keys())

    def __str__(self):
        return self.name


class Steps(SuperEnum):
    train = auto()
    dev = auto()
    test = auto()
