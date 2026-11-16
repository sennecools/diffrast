"""Log level for diffrast's own messages, numbered as in nvdiffrast."""

from __future__ import annotations

import operator
import warnings

INFO, WARNING, ERROR, FATAL = range(4)

_level = WARNING


def get_log_level() -> int:
    """Return the current log level: 0 is info, 1 warning (the default), 2 error and 3 fatal."""
    return _level


def set_log_level(level: int) -> None:
    """Set the lowest level of message that diffrast reports.

    Args:
        level: 0 for info, 1 for warning, 2 for error or 3 for fatal. Warnings are only issued
            at levels 0 and 1; errors are always raised as exceptions, whatever the level.
    """
    global _level
    try:
        value = operator.index(level)
    except TypeError:
        raise TypeError(f"level must be an int from 0 to 3, got {level!r}.") from None
    if not INFO <= value <= FATAL:
        raise ValueError(f"level must be an int from 0 to 3, got {level!r}.")
    _level = value


def warn(message: str, stacklevel: int = 1) -> None:
    """Issue a ``UserWarning`` attributed to the caller, unless the log level silences warnings."""
    if _level <= WARNING:
        warnings.warn(message, stacklevel=stacklevel + 1)
