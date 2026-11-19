"""Scope LongMedia runtime state to the work that owns it."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import MethodType
from typing import Any

_LOG = logging.getLogger(__name__)
_MISSING = object()


def snapshot_attributes(owner: object, *names: str) -> dict[str, Any]:
    return {name: vars(owner).get(name, _MISSING) for name in names}


def restore_attributes(owner: object, saved: Mapping[str, Any]) -> None:
    for name, value in saved.items():
        if value is not _MISSING:
            setattr(owner, name, value)
        elif name in vars(owner):
            delattr(owner, name)


@contextmanager
def temporary_method_override(
    owner: object, name: str, method: Callable[..., Any],
    *, attributes: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    attributes = attributes or {}
    saved = snapshot_attributes(owner, name, *attributes)
    try:
        for key, value in attributes.items():
            setattr(owner, key, value)
        setattr(owner, name, MethodType(method, owner))
        yield
    finally:
        restore_attributes(owner, saved)


class SamplerCleanup:
    """Attempt independent cleanup stages even when a previous stage fails."""

    def __init__(self) -> None:
        self.first_error: BaseException | None = None

    def attempt(self, label: str, action: Callable[[], Any]) -> Any:
        try:
            return action()
        except BaseException as error:
            if self.first_error is None:
                self.first_error = error
            _LOG.exception("LongMedia sampler cleanup failed during %s", label)
            return None

    def finish(self, active_error: BaseException | None) -> None:
        if self.first_error is not None and active_error is None:
            raise self.first_error
