"""The five baseline implementations reported by the ICRA paper."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    module: str
    description: str

    def load(self) -> Any:
        return import_module(self.module)

    def main(self) -> Callable[..., Any]:
        return self.load().main


BASELINES = {
    "pdr": BaselineSpec("PDR", "cowear.baselines.pdr", "peak-trough Weinberg PDR"),
    "ridi": BaselineSpec("RIDI", "cowear.baselines.ridi", "RIDI SVR velocity correction"),
    "ronin": BaselineSpec("RoNIN", "cowear.baselines.ronin", "RoNIN 1-D ResNet velocity"),
    "tlio": BaselineSpec("TLIO", "cowear.baselines.tlio", "TLIO displacement and uncertainty"),
    "cowear_lstm": BaselineSpec("CoWear LSTM", "cowear.models.cowear_lstm", "CoWear common-target LSTM"),
}


def get_baseline(name: str) -> BaselineSpec:
    try:
        return BASELINES[name]
    except KeyError as exc:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}") from exc


__all__ = ["BASELINES", "BaselineSpec", "get_baseline"]
