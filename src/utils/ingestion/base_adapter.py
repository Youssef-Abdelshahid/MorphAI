from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class UnsupportedFormatError(NotImplementedError):
    pass


@dataclass
class AdapterResult:
    ok: bool
    message: str = ""
    data: Optional[Any] = None
    schema: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class BaseFormatAdapter:
    modality: str = ""
    input_format: str = ""
    is_implemented: bool = False

    def validate_input(self, path: Path) -> AdapterResult:
        return AdapterResult(
            ok=False,
            message=self._not_implemented_message(),
            errors=[self._not_implemented_message()],
        )

    def parse(self, path: Path) -> AdapterResult:
        raise UnsupportedFormatError(self._not_implemented_message())

    def profile_structure(self, path: Path) -> AdapterResult:
        raise UnsupportedFormatError(self._not_implemented_message())

    def to_internal_dataset(self, path: Path) -> AdapterResult:
        raise UnsupportedFormatError(self._not_implemented_message())

    def _not_implemented_message(self) -> str:
        return (
            f"The '{self.input_format}' input format for {self.modality} "
            "is planned but not implemented yet."
        )
