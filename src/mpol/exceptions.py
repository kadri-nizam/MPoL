from __future__ import annotations


class CellSizeError(Exception):
    ...


class TorchIncompatibleFunctionError(Exception):
    ...


class WrongDimensionError(Exception):
    ...


class DataError(Exception):
    ...


class ThresholdExceededError(Exception):
    ...
