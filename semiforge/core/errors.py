"""Error taxonomy E100~E500."""

from __future__ import annotations


class SemiForgeError(Exception):
    """Base error. code encodes the failing subsystem."""

    code = "E000"

    def __init__(self, msg: str) -> None:
        super().__init__(f"[{self.code}] {msg}")


class DataError(SemiForgeError):
    code = "E100"


class ModelError(SemiForgeError):
    code = "E200"


class EvalError(SemiForgeError):
    code = "E300"


class RegistryError(SemiForgeError):
    code = "E400"


class PipelineError(SemiForgeError):
    code = "E500"
