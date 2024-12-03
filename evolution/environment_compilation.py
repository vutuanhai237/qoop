
from dataclasses import dataclass, field
from .environment_parent import Metadata

@dataclass
class MetadataCompilation(Metadata):
    depth: int = field(default_factory=lambda: "depth")