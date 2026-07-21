from __future__ import annotations

import math
import keyword
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from power_forecasting.data import parse_timestamps


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    transform: str
    inputs: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1"
    rationale: str = ""

    def __post_init__(self) -> None:
        _validate_feature_name(self.name)
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if not all(isinstance(column, str) and column for column in self.inputs):
            raise ValueError(f"feature {self.name}: inputs must be non-empty strings")
        if not isinstance(self.parameters, Mapping):
            raise TypeError(f"feature {self.name}: parameters must be a mapping")
        if not all(isinstance(key, str) for key in self.parameters):
            raise ValueError(f"feature {self.name}: parameter names must be strings")
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(
                {key: _freeze_parameter(value) for key, value in self.parameters.items()}
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transform": self.transform,
            "inputs": list(self.inputs),
            "parameters": {
                key: _json_compatible(value)
                for key, value in sorted(self.parameters.items())
            },
            "version": self.version,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureSpec":
        return cls(
            name=data["name"],
            transform=data["transform"],
            inputs=tuple(data["inputs"]),
            parameters=data.get("parameters", {}),
            version=data.get("version", "1"),
            rationale=data.get("rationale", ""),
        )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    return value


def _freeze_parameter(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_parameter(nested) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_parameter(item) for item in value)
    return value


def _validate_feature_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or not name.isidentifier()
        or keyword.iskeyword(name)
    ):
        raise ValueError(f"invalid feature name: {name!r}")


def _datetime_input(frame: pd.DataFrame, spec: FeatureSpec, index: int) -> pd.Series:
    input_name = spec.inputs[index]
    return parse_timestamps(
        frame[input_name],
        error_message=f"feature {spec.name}: invalid datetime input {input_name}",
    )


def _cyclic_hour(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    timestamps = _datetime_input(frame, spec, 0)
    hours = timestamps.dt.hour.to_numpy(dtype=float)
    if spec.name.endswith("_sin"):
        return np.sin(2 * math.pi * hours / 24)
    if spec.name.endswith("_cos"):
        return np.cos(2 * math.pi * hours / 24)
    raise ValueError(f"feature {spec.name}: cyclic_hour output must end with _sin or _cos")


def _cyclic_day_of_year(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    timestamps = _datetime_input(frame, spec, 0)
    day_of_year = timestamps.dt.dayofyear.to_numpy(dtype=float)
    if spec.name.endswith("_sin"):
        return np.sin(2 * math.pi * day_of_year / 365.25)
    if spec.name.endswith("_cos"):
        return np.cos(2 * math.pi * day_of_year / 365.25)
    raise ValueError(
        f"feature {spec.name}: cyclic_day_of_year output must end with _sin or _cos"
    )


def _numeric_input(frame: pd.DataFrame, spec: FeatureSpec, index: int) -> np.ndarray:
    values = pd.to_numeric(frame[spec.inputs[index]], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"feature {spec.name}: non-finite numeric input {spec.inputs[index]}")
    return values


def _effective_irradiance(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    irradiance = _numeric_input(frame, spec, 0)
    cloud_cover = _numeric_input(frame, spec, 1)
    return irradiance * np.clip(1 - cloud_cover, 0, 1)


def _temperature_derating(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    irradiance = _numeric_input(frame, spec, 0)
    temperature = _numeric_input(frame, spec, 1)
    reference = _parameter_float(spec, "reference", 25.0)
    coefficient = _parameter_float(spec, "coefficient", 0.004)
    factor = np.clip(1 - coefficient * np.maximum(temperature - reference, 0), 0, 1)
    return irradiance * factor


def _cloud_attenuation(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    cloud_cover = _numeric_input(frame, spec, 0)
    return np.clip(1 - cloud_cover, 0, 1)


def _interaction(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    return _numeric_input(frame, spec, 0) * _numeric_input(frame, spec, 1)


def _ratio(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    numerator = _numeric_input(frame, spec, 0)
    denominator = _numeric_input(frame, spec, 1)
    epsilon = _parameter_float(spec, "epsilon", 1e-6)
    if (np.abs(denominator) <= epsilon).any():
        raise ValueError(f"feature {spec.name}: denominator near zero")
    return numerator / denominator


def _lag(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    periods = _parameter_int_member(spec, "periods", {1, 2, 3, 6, 12, 24})
    values, insufficient = _history_values(
        frame,
        spec,
        window=periods,
        reducer=lambda history: history[-periods],
    )
    return values


def _rolling_mean(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    window = _parameter_int_member(spec, "window", {3, 6, 12, 24})
    values, insufficient = _history_values(
        frame,
        spec,
        window=window,
        reducer=lambda history: float(np.mean(history[-window:])),
    )
    return values


def _history_values(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    *,
    window: int,
    reducer: Callable[[np.ndarray], float],
    allow_partial: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if "plant_id" not in frame.columns:
        raise ValueError(f"feature {spec.name}: plant_id column is required")
    if "timestamp" not in frame.columns:
        raise ValueError(f"feature {spec.name}: timestamp column is required")

    source = _numeric_input(frame, spec, 0)
    timestamps = parse_timestamps(
        frame["timestamp"],
        error_message=f"feature {spec.name}: invalid timestamp column",
    )
    working = pd.DataFrame(
        {
            "_position": np.arange(len(frame), dtype=int),
            "_plant_id": frame["plant_id"].to_numpy(),
            "_timestamp": timestamps.to_numpy(),
            "_source": source,
        }
    ).sort_values(["_plant_id", "_timestamp", "_position"], kind="mergesort")

    values = np.full(len(frame), np.nan, dtype=float)
    insufficient = np.zeros(len(frame), dtype=bool)
    for _, plant_rows in working.groupby("_plant_id", sort=False):
        prior_by_timestamp: list[float] = []
        for _, timestamp_rows in plant_rows.groupby("_timestamp", sort=False):
            positions = timestamp_rows["_position"].to_numpy(dtype=int)
            if len(prior_by_timestamp) < window:
                insufficient[positions] = True
            if prior_by_timestamp and (allow_partial or len(prior_by_timestamp) >= window):
                history = np.asarray(prior_by_timestamp, dtype=float)
                values[positions] = reducer(history)
            prior_by_timestamp.append(float(timestamp_rows["_source"].iloc[-1]))
    return values, insufficient


TRANSFORMS: dict[str, Callable[[pd.DataFrame, FeatureSpec], np.ndarray]] = {
    "cyclic_hour": _cyclic_hour,
    "cyclic_day_of_year": _cyclic_day_of_year,
    "effective_irradiance": _effective_irradiance,
    "temperature_derating": _temperature_derating,
    "cloud_attenuation": _cloud_attenuation,
    "interaction": _interaction,
    "ratio": _ratio,
    "lag": _lag,
    "rolling_mean": _rolling_mean,
}


_TRANSFORM_ARITY = {
    "cyclic_hour": 1,
    "cyclic_day_of_year": 1,
    "effective_irradiance": 2,
    "temperature_derating": 2,
    "cloud_attenuation": 1,
    "interaction": 2,
    "ratio": 2,
    "lag": 1,
    "rolling_mean": 1,
}

_TRANSFORM_PARAMETERS = {
    "cyclic_hour": frozenset(),
    "cyclic_day_of_year": frozenset(),
    "effective_irradiance": frozenset(),
    "temperature_derating": frozenset({"reference", "coefficient"}),
    "cloud_attenuation": frozenset(),
    "interaction": frozenset(),
    "ratio": frozenset({"epsilon"}),
    "lag": frozenset({"periods"}),
    "rolling_mean": frozenset({"window"}),
}


def _parameter_float(spec: FeatureSpec, key: str, default: float) -> float:
    raw_value = spec.parameters.get(key, default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature {spec.name}: parameter {key} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"feature {spec.name}: parameter {key} must be finite")
    return value


def _parameter_int_member(spec: FeatureSpec, key: str, allowed: set[int]) -> int:
    if key not in spec.parameters:
        raise ValueError(f"feature {spec.name}: missing parameter {key}")
    raw_value = spec.parameters[key]
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"feature {spec.name}: parameter {key} must be an integer")
    value = int(raw_value)
    if value not in allowed:
        raise ValueError(f"feature {spec.name}: parameter {key} outside allowed set")
    return value


def _validate_parameters(spec: FeatureSpec) -> None:
    allowed = _TRANSFORM_PARAMETERS[spec.transform]
    unexpected = sorted(set(spec.parameters) - allowed)
    if unexpected:
        raise ValueError(f"feature {spec.name}: unexpected parameters: {unexpected}")

    if spec.transform == "temperature_derating":
        _parameter_float(spec, "reference", 25.0)
        coefficient = _parameter_float(spec, "coefficient", 0.004)
        if coefficient < 0:
            raise ValueError(f"feature {spec.name}: coefficient must be non-negative")
    elif spec.transform == "ratio":
        epsilon = _parameter_float(spec, "epsilon", 1e-6)
        if epsilon <= 0:
            raise ValueError(f"feature {spec.name}: epsilon must be > 0")
    elif spec.transform == "lag":
        _parameter_int_member(spec, "periods", {1, 2, 3, 6, 12, 24})
    elif spec.transform == "rolling_mean":
        _parameter_int_member(spec, "window", {3, 6, 12, 24})


def _validate_spec(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    _validate_feature_name(spec.name)
    if spec.transform not in TRANSFORMS:
        raise ValueError(f"feature {spec.name}: unknown transform {spec.transform}")
    expected_arity = _TRANSFORM_ARITY[spec.transform]
    if len(spec.inputs) != expected_arity:
        raise ValueError(
            f"feature {spec.name}: {spec.transform} expects {expected_arity} inputs"
        )
    if "generation_mw" in spec.inputs:
        raise ValueError(f"feature {spec.name}: target leakage input generation_mw")
    missing = [column for column in spec.inputs if column not in frame.columns]
    if missing:
        raise ValueError(f"feature {spec.name}: missing source columns: {missing}")
    _validate_parameters(spec)


def apply_feature_specs(frame: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    seen_names: set[str] = set()
    for spec in specs:
        if spec.name in seen_names:
            raise ValueError(f"duplicate feature name: {spec.name}")
        seen_names.add(spec.name)
        _validate_spec(frame, spec)

    output = pd.DataFrame(index=frame.index)
    for spec in specs:
        values = TRANSFORMS[spec.transform](frame, spec)
        if spec.transform in {"lag", "rolling_mean"}:
            invalid = np.isinf(values).any()
        else:
            invalid = not np.isfinite(values).all()
        if invalid:
            raise ValueError(f"feature {spec.name}: non-finite output")
        output[spec.name] = values
    return output
