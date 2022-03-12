# Manage the current logical path and killed paths / mutations.

import json
import os
from tempfile import TemporaryFile
from typing import Any, Optional, Union

from .utils import MAINLINE, ShadowExceptionStop
from .mode import get_execution_mode

import logging
logger = logging.getLogger(__name__)

_LOGICAL_PATH = MAINLINE
_SELECTED_MUTANT: Union[int, None] = None
_STRONGLY_KILLED: set[int] = set()
_SEEN_MUTANTS: set[int] = set()
_FUNCTION_SEEN_MUTANTS: set[int] = set()
_FUNCTION_MASKED_MUTANTS: set[int] = set()
_MASKED_MUTANTS: set[int] = set()
_NS_ACTIVE_MUTANTS: Union[set[int], None] = None # NS = Non Shadow
_SHARED_RESULTS_PATH: Optional[Any] = None

def reinit_path(logical_path: Union[int, None]) -> None:
    global _LOGICAL_PATH
    global _SELECTED_MUTANT
    global _STRONGLY_KILLED
    global _NS_ACTIVE_MUTANTS
    global _SEEN_MUTANTS
    global _FUNCTION_SEEN_MUTANTS
    global _FUNCTION_MASKED_MUTANTS
    global _MASKED_MUTANTS
    global _SHARED_RESULTS_PATH

    if logical_path is not None:
        _LOGICAL_PATH = logical_path
    else:
        _LOGICAL_PATH = int(os.environ.get('LOGICAL_PATH', MAINLINE))

    _SELECTED_MUTANT = None
    _STRONGLY_KILLED = set()

    mode = get_execution_mode()

    if mode.is_shadow_variant():
        _SEEN_MUTANTS = set()
        _FUNCTION_SEEN_MUTANTS = set()
        _FUNCTION_MASKED_MUTANTS = set()
        _MASKED_MUTANTS = set()
    elif mode.is_split_stream_variant():
        _NS_ACTIVE_MUTANTS = None
        _MASKED_MUTANTS = set()
    elif mode.is_not_specified():
        pass
    else:
        raise ValueError(f"Unknown execution mode: {mode}")

    if mode.is_shadow_fork_variant():
        _SHARED_RESULTS_PATH = TemporaryFile(mode='w+t')
        save_shared()


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

    if get_execution_mode().is_shadow_variant():
        if mut in _STRONGLY_KILLED:
            logger.warning(f"redundant strongly killed: {mut}")
            assert False

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


def get_function_seen_masked() -> set[int]:
    return _FUNCTION_SEEN_MUTANTS, _FUNCTION_MASKED_MUTANTS


def add_function_seen_mutants(muts: set[int]):
    global _FUNCTION_SEEN_MUTANTS
    _FUNCTION_SEEN_MUTANTS |= muts


def reset_function_seen_masked() -> set[int]:
    global _FUNCTION_SEEN_MUTANTS
    global _FUNCTION_MASKED_MUTANTS
    _FUNCTION_SEEN_MUTANTS.clear()
    _FUNCTION_MASKED_MUTANTS.clear()


def add_seen_mutants(muts: set[int]):
    global _SEEN_MUTANTS
    global _FUNCTION_SEEN_MUTANTS
    _SEEN_MUTANTS |= muts
    _FUNCTION_SEEN_MUTANTS |= muts


def get_masked_mutants() -> set[int]:
    return _MASKED_MUTANTS


def set_masked_mutants(masked: set[int]) -> None:
    global _MASKED_MUTANTS
    _MASKED_MUTANTS = masked


def add_masked_mutants(masked: set[int]):
    global _MASKED_MUTANTS
    global _FUNCTION_MASKED_MUTANTS
    _MASKED_MUTANTS |= masked
    _FUNCTION_MASKED_MUTANTS |= masked


def remove_masked_mutants(masked: set[int]):
    global _MASKED_MUTANTS
    _MASKED_MUTANTS -= masked


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


def load_shared() -> None:
    if get_execution_mode().is_shadow_fork_variant():
        global _STRONGLY_KILLED
        _SHARED_RESULTS_PATH.seek(0)
        try:
            data = set(json.load(_SHARED_RESULTS_PATH))
        except Exception as e:
            logger.warning(f"{e}")
            raise e

        add_seen_mutants(data)
        for mut in data - _STRONGLY_KILLED:
            try:
                add_strongly_killed(mut)
            except ShadowExceptionStop as e:
                logger.debug(f"load shared ended run")
                raise e



def save_shared() -> None:
    if get_execution_mode().is_shadow_fork_variant():
        _SHARED_RESULTS_PATH.truncate(0)
        _SHARED_RESULTS_PATH.seek(0)
        try:
            json.dump(list(_STRONGLY_KILLED), _SHARED_RESULTS_PATH)
        except Exception as e:
            logger.warning(f"{e}")
            raise e

        _SHARED_RESULTS_PATH.flush()
