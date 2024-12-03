
from dataclasses import dataclass, field
from .environment_parent import Metadata

@dataclass
class MetadataSynthesis(Metadata):
    num_cnot: int = field(default_factory=lambda: "num_cnot")
    depth: int = field(default_factory=lambda: "depth")