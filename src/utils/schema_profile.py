from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SchemaProfile:
    field_names: List[str] = field(default_factory=list)
    field_types: Dict[str, str] = field(default_factory=dict)
    nesting_depth: int = 0
    record_count: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)
