from __future__ import annotations

import re


_PROFILE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def is_profile_name(value: object) -> bool:
    """Return whether *value* is a canonical external profile identifier."""

    return type(value) is str and _PROFILE_NAME_PATTERN.fullmatch(value) is not None


__all__ = ["is_profile_name"]
