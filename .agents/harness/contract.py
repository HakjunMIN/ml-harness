"""Compatibility shim for the original ``harness.contract`` import path."""

from legacy_adapter.contract import (
    AdapterConfig,
    AdapterContractError,
    HARNESS_ENV_KEYS,
    MANIFEST_FIELDS,
    SCHEMA_VERSION,
    load_adapter,
    main,
    run_adapter,
    sha256_file,
)

__all__ = [
    "AdapterConfig",
    "AdapterContractError",
    "load_adapter",
    "main",
    "run_adapter",
    "sha256_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
