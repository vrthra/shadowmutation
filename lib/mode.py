# Current execution mode

from __future__ import annotations
import os
from enum import Enum
from typing import Union

import logging
logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    NOT_SPECIFIED = 0
    SPLIT_STREAM = 1 # split stream execution
    MODULO_EQV = 2 # split stream + modulo equivalence pruning execution
    SHADOW = 3 # shadow types and no forking
    SHADOW_FORK_CHILD = 4 # shadow and forking, child first
    SHADOW_FORK_PARENT = 5 # shadow and forking, parent first
    SHADOW_CACHE = 6 # shadow types and no forking with caching
    SHADOW_FORK_CACHE = 7 # shadow and forking with caching (always parent first)

    @staticmethod
    def get_mode(mode: Union[str, None]) -> ExecutionMode:
        if mode is None:
            return ExecutionMode.NOT_SPECIFIED
        elif mode == 'split':
            return ExecutionMode.SPLIT_STREAM
        elif mode == 'modulo':
            return ExecutionMode.MODULO_EQV
        elif mode == 'shadow':
            return ExecutionMode.SHADOW
        elif mode == 'shadow_fork_child':
            return ExecutionMode.SHADOW_FORK_CHILD
        elif mode == 'shadow_fork_parent':
            return ExecutionMode.SHADOW_FORK_PARENT
        elif mode == 'shadow_cache':
            return ExecutionMode.SHADOW_CACHE
        elif mode == 'shadow_fork_cache':
            return ExecutionMode.SHADOW_FORK_CACHE
        else:
            raise ValueError("Unknown Execution Mode", mode)

    def is_not_specified(self) -> bool:
        return self == ExecutionMode.NOT_SPECIFIED

    def should_start_forker(self) -> bool:
        if self in _SHOULD_START_FORKER:
            return True
        else:
            return False

    def is_shadow_variant(self) -> bool:
        return self in _IS_SHADOW_VARIANT

    def is_shadow_fork_variant(self) -> bool:
        return self in _IS_SHADOW_FORK_VARIANT

    def is_shadow_fork_child(self) -> bool:
        return self == ExecutionMode.SHADOW_FORK_CHILD

    def is_shadow_fork_parent(self) -> bool:
        return self == ExecutionMode.SHADOW_FORK_PARENT

    def is_shadow_fork_cache(self) -> bool:
        return self == ExecutionMode.SHADOW_FORK_CACHE

    def is_split_stream_variant(self) -> bool:
        return self in _IS_SPLIT_STREAM_VARIANT

    def is_split_stream(self) -> bool:
        return self == ExecutionMode.SPLIT_STREAM

    def is_modulo_eqv(self) -> bool:
        return self == ExecutionMode.MODULO_EQV

    def uses_cache(self) -> bool:
        return self in _USES_CACHE


_SHOULD_START_FORKER = set([
    ExecutionMode.SPLIT_STREAM,
    ExecutionMode.MODULO_EQV,
    ExecutionMode.SHADOW_FORK_CHILD,
    ExecutionMode.SHADOW_FORK_PARENT,
    ExecutionMode.SHADOW_FORK_CACHE,
])

_IS_SHADOW_VARIANT = set([
    ExecutionMode.SHADOW,
    ExecutionMode.SHADOW_CACHE,
    ExecutionMode.SHADOW_FORK_CHILD,
    ExecutionMode.SHADOW_FORK_PARENT,
    ExecutionMode.SHADOW_FORK_CACHE
])

_IS_SHADOW_FORK_VARIANT = set([
    ExecutionMode.SHADOW_FORK_CHILD,
    ExecutionMode.SHADOW_FORK_PARENT,
    ExecutionMode.SHADOW_FORK_CACHE
])

_IS_SPLIT_STREAM_VARIANT = set([ExecutionMode.SPLIT_STREAM, ExecutionMode.MODULO_EQV])

_USES_CACHE = set([ExecutionMode.SHADOW_CACHE, ExecutionMode.SHADOW_FORK_CACHE])


_EXECUTION_MODE: ExecutionMode = ExecutionMode.NOT_SPECIFIED


def reinit_execution_mode(execution_mode: Union[str, None]) -> None:
    global _EXECUTION_MODE
    if execution_mode is not None:
        _EXECUTION_MODE = ExecutionMode.get_mode(execution_mode)
    else:
        _EXECUTION_MODE = ExecutionMode.get_mode(os.environ.get('EXECUTION_MODE'))


def get_execution_mode() -> ExecutionMode:
    return _EXECUTION_MODE