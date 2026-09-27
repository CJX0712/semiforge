"""Strategy protocol - all SSL strategies implement this contract.

Unified semantics: fit(dataset) then predict returns labels for ALL n points.
Evaluation is performed on the unlabeled subset by the caller.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from semiforge.core.types import SSLDataset


@runtime_checkable
class SSLStrategy(Protocol):
    name: str

    def fit(self, dataset: SSLDataset) -> SSLStrategy: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...
