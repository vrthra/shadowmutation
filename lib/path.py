# Manage the current logical path and killed paths / mutations.

import os
from typing import Any, Union

from .constants import MAINLINE, ShadowExceptionStop
from .mode import get_execution_mode

_LOGICAL_PATH = MAINLINE
_SELECTED_MUTANT: Union[int, None] = None
_STRONGLY_KILLED: set[int] = set()
_SEEN_MUTANTS: set[int] = set()
_MASKED_MUTANTS: set[int] = set()
_NS_ACTIVE_MUTANTS: Union[set[int], None] = None # NS = Non Shadow

def reinit_path(logical_path: Union[int, None]) -> None:
    global _LOGICAL_PATH
    global _SELECTED_MUTANT
    global _STRONGLY_KILLED
    global _NS_ACTIVE_MUTANTS
    global _SEEN_MUTANTS
    global _MASKED_MUTANTS

    if logical_path is not None:
        _LOGICAL_PATH = logical_path
    else:
        _LOGICAL_PATH = int(os.environ.get('LOGICAL_PATH', MAINLINE))

    _SELECTED_MUTANT = None
    _STRONGLY_KILLED = set()

    mode = get_execution_mode()

    if mode.is_shadow_variant():
        _SEEN_MUTANTS = set()
        _MASKED_MUTANTS = set()
    elif mode.is_split_stream_variant():
        _NS_ACTIVE_MUTANTS = None
        _MASKED_MUTANTS = set()
    elif mode.is_not_specified():
        pass
    else:
        raise ValueError(f"Unknown execution mode: {mode}")


def get_logical_path() -> int:
    return _LOGICAL_PATH


def set_logical_path(path: int) -> None:
    global _LOGICAL_PATH
    _LOGICAL_PATH = path


def active_mutants() -> set[int]:
    global _SEEN_MUTANTS
    global _MASKED_MUTANTS
    return _SEEN_MUTANTS - _MASKED_MUTANTS


def add_strongly_killed(mut: int) -> None:
    global _MASKED_MUTANTS

    if mut in _STRONGLY_KILLED:
        # TODO reduce multiply killed mutants
        # logger.debug(f"redundant strongly killed: {mut}")
        pass

    _MASKED_MUTANTS.add(mut)
    _STRONGLY_KILLED.add(mut)

    if get_logical_path() == mut:
        if get_execution_mode().is_shadow_variant():
            cur_active_mutants = active_mutants()
            if len(cur_active_mutants) > 0:
                set_logical_path(cur_active_mutants.pop())
            else:
                from .fork import get_forking_context
                context = get_forking_context()
                if context is not None:
                    context.child_end() # child fork ends here
                else:
                    raise ShadowExceptionStop()


def get_selected_mutant() -> Union[int, None]:
    return _SELECTED_MUTANT


def set_selected_mutant(mut: Union[int, None]):
    global _SELECTED_MUTANT
    _SELECTED_MUTANT = mut


def get_strongly_killed() -> set[int]:
    return _STRONGLY_KILLED


def merge_strongly_killed(killed: set[int]):
    global _STRONGLY_KILLED
    _STRONGLY_KILLED |= killed


def get_seen_mutants() -> set[int]:
    return _SEEN_MUTANTS


def add_seen_mutants(muts: set[int]):
    global _SEEN_MUTANTS
    _SEEN_MUTANTS |= muts


def get_masked_mutants() -> set[int]:
    return _MASKED_MUTANTS


def set_masked_mutants(masked: set[int]) -> None:
    global _MASKED_MUTANTS
    _MASKED_MUTANTS = masked


def add_masked_mutants(masked: set[int]):
    global _MASKED_MUTANTS
    _MASKED_MUTANTS |= masked


def get_ns_active_mutants() -> set[int]:
    return _NS_ACTIVE_MUTANTS


def set_ns_active_mutants(muts: set[int]) -> None:
    global _NS_ACTIVE_MUTANTS
    _NS_ACTIVE_MUTANTS = muts


def add_ns_active_mutants(muts: set[int]):
    global _NS_ACTIVE_MUTANTS
    _NS_ACTIVE_MUTANTS |= muts


def remove_ns_active_mutants(muts: set[int]):
    global _NS_ACTIVE_MUTANTS
    _NS_ACTIVE_MUTANTS -= muts


def t_get_killed() -> dict[str, Any]:
    res = {
        'strong': _STRONGLY_KILLED,
        'masked': _MASKED_MUTANTS,
    }
    mode = get_execution_mode()
    if mode.is_shadow_variant():
        res['seen'] = _SEEN_MUTANTS
    elif mode.is_split_stream_variant():
        res['active'] = _NS_ACTIVE_MUTANTS
    return res
