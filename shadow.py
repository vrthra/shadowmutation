
from multiprocessing import get_logger
import pickle
import os
import sys
import json
import tempfile
import time
import atexit
import traceback
import types
import signal
from typing import Any, Callable, TypeVar, Tuple, Union
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from copy import deepcopy
from functools import wraps

from lib.constants import MAINLINE
from lib.mode import reinit_execution_mode, get_execution_mode
from lib.path import (
    reinit_path,
    get_logical_path, set_logical_path,
    add_strongly_killed, get_strongly_killed, t_get_killed,
    get_seen_mutants,
    active_mutants,
    get_masked_mutants, set_masked_mutants, add_masked_mutants,
    get_ns_active_mutants, set_ns_active_mutants, add_ns_active_mutants, remove_ns_active_mutants,
)
from lib.fork import Forker, get_forking_context, new_forking_context, reinit_forking_context, set_forking_context
from lib.line_counter import get_counter_results, reinit_trace, disable_line_counting
from lib.func_wrap import t_wrap
from lib.cache import maybe_clean_cache, reinit_cache
from lib.shadow_variable import (
    ShadowVariable, set_new_no_init, t_class, shadow_assert, t_combine_shadow, unset_new_no_init, untaint_args
)

from typing import Union
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(process)d %(filename)s:%(lineno)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class SetEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


RESULT_FILE: Union[str, None] = None
# FORKING_CONTEXT: Union[None, Forker] = None


def reinit(logical_path: Union[int, None]=None, execution_mode: Union[str, None]=None, no_atexit: bool=False) -> None:
    # logger.info("Reinit global shadow state")
    # initializing shadow
    global RESULT_FILE

    RESULT_FILE = os.environ.get('RESULT_FILE')

    reinit_execution_mode(execution_mode)
    reinit_path(logical_path)
    reinit_forking_context()
    reinit_cache()

    if os.environ.get('GATHER_ATEXIT', '0') == '1':
        atexit.register(t_gather_results)
    else:
        atexit.unregister(t_gather_results)

    reinit_trace()


def t_wait_for_forks() -> None:
    forking_context = get_forking_context()
    if forking_context is not None:
        forking_context.wait_for_forks()


def t_counter_results():
    return get_counter_results()


def t_gather_results() -> Any:
    disable_line_counting()
    t_wait_for_forks()
    maybe_clean_cache()

    results = t_get_killed()
    results['execution_mode'] = get_execution_mode().name
    results = {**results, **t_counter_results()}
    if RESULT_FILE is not None:
        with open(RESULT_FILE, 'wt') as f:
            json.dump(results, f, cls=SetEncoder)
    return results


def t_final_exception() -> None:
    # Program is crashing, mark all active mutants as strongly killed
    for mut in active_mutants():
        add_strongly_killed(mut)
    t_gather_results()


def t_final_exception_test() -> None:
    for mut in active_mutants():
        add_strongly_killed(mut)
    t_wait_for_forks()


def t_cond(cond: Any) -> bool:

    if type(cond) == ShadowVariable:
        diverging_mutants = []
        companion_mutants = []

        # get the logical path result, this is used to decide which mutations follow the logical path and which do not
        logical_result = cond._get_logical_res(get_logical_path())
        assert type(logical_result) == bool, f"{cond}"

        for path, val in cond._all_path_results(get_seen_mutants(), get_masked_mutants()):
            if path == MAINLINE or path == get_logical_path():
                continue
            assert type(val) == bool, f"{cond}"
            if val == logical_result:
                companion_mutants.append(path)
            else:
                diverging_mutants.append(path)

        forking_context = get_forking_context() 
        if forking_context is not None:
            original_path = get_logical_path()
            # Fork if enabled
            if diverging_mutants:
                # logger.debug(f"path: {get_logical_path()} masked: {get_masked_mutants()} seen: {get_seen_mutants()} companion: {companion_mutants} diverging: {diverging_mutants}")
                # select the path to follow, just pick first
                path = diverging_mutants[0]
                if forking_context.maybe_fork(path):
                    # we are now in the forked child
                    add_masked_mutants(set(companion_mutants + [original_path]))
                    return t_cond(cond)
                else:
                    add_masked_mutants(set(diverging_mutants))
        else:
            # Follow the logical path, if that is not the same as mainline mark other mutations as inactive
            if diverging_mutants:
                add_masked_mutants(set(diverging_mutants))

        return logical_result

    else:
        return cond
        # logger.debug("vanilla")
        # raise NotImplementedError()


def split_assert(cmp_result):
    # logger.debug(f"t_assert {cmp_result}")
    assert type(cmp_result) == bool, f"{type(cmp_result)}"
    if cmp_result:
        return
    else:
        if get_ns_active_mutants() is None:
            logger.warning("NS_ACTIVE_MUTANTS is None")
            raise ValueError("NS_ACTIVE_MUTANTS is None")
        for mut in get_ns_active_mutants():
            add_strongly_killed(mut)
            # logger.info(f"t_assert strongly killed: {mut}")


def t_assert(cmp_result):
    mode = get_execution_mode()
    if mode.is_shadow_variant():
        shadow_assert(cmp_result)
    elif mode.is_split_stream_variant():
        split_assert(cmp_result)
    else:
        raise ValueError("Unknown execution mode: {mode}")


def t_logical_path():
    return get_logical_path()


def t_seen_mutants():
    return get_seen_mutants()


def t_masked_mutants():
    return get_masked_mutants()


def t_active_mutants():
    return active_mutants()


def untaint(obj):
    if hasattr(obj, '_shadow'):
        return obj._shadow[MAINLINE]
    return obj


def get_ns_active(mutations, active, masked):
    if active is not None:
        filtered_mutations = { path: val for path, val in mutations.items() if path in active }
    else:
        filtered_mutations = { path: val for path, val in mutations.items() if path not in masked }

    # logger.debug(f"log_path: {get_logical_path()}")
    filtered_mutations[MAINLINE] = mutations[MAINLINE]
    if get_logical_path() in mutations:
        filtered_mutations[get_logical_path()] = mutations[get_logical_path()]

    return filtered_mutations


def combine_split_stream(mutations):
    forking_context = get_forking_context()

    if get_logical_path() == MAINLINE:
        all_muts = set(mutations.keys())
        for mut_id, val in mutations.items():
            if mut_id in [MAINLINE, get_logical_path()]:
                continue

            if isinstance(val, Exception):
                logger.debug(f"val is exception: {val}")

            if forking_context.maybe_fork(mut_id):
                set_ns_active_mutants(set([mut_id]))
                set_masked_mutants(all_muts - get_ns_active_mutants())
                return val

    try:
        return_val = mutations[get_logical_path()]
    except KeyError:
        return_val = mutations[MAINLINE]

    if isinstance(return_val, Exception):
        if get_logical_path() == MAINLINE:
            logger.debug(f"mainline return_val is exception: {return_val}")
            raise NotImplementedError()
        add_strongly_killed(get_logical_path())
        forking_context.child_end()

    return return_val


def combine_modulo_eqv(mutations):
    forking_context = get_forking_context()

    mutations = get_ns_active(mutations, get_ns_active_mutants(), get_masked_mutants())

    if get_logical_path() in mutations:
        log_res = mutations[get_logical_path()]
    else:
        log_res = mutations[MAINLINE]

    # if get_logical_path() == MAINLINE:
    combined = defaultdict(list)
    for mut_id in set(mutations.keys()) | (get_ns_active_mutants() or set()):
        if mut_id in [MAINLINE, get_logical_path()]:
            continue

        if mut_id in mutations:
            val = mutations[mut_id]
        else:
            val = mutations[MAINLINE]

        combined[val].append(mut_id)

    for val, mut_ids in combined.items():
        if isinstance(val, Exception):
            for mut_id in mut_ids:
                add_strongly_killed(mut_id)
            continue

        if val != log_res:
            main_mut_id = mut_ids[0]
            if get_ns_active_mutants() is not None:
                remove_ns_active_mutants(set(mut_ids))
            add_masked_mutants(set(mut_ids))
            # logger.debug(f"masked: {get_masked_mutants()}")
            if forking_context.maybe_fork(main_mut_id):
                set_masked_mutants(set())
                set_ns_active_mutants(set(mut_ids))
                return val

    if isinstance(log_res, Exception):
        if get_logical_path() != MAINLINE:
            assert get_ns_active_mutants() is not None
            for mut_id in get_ns_active_mutants():
                add_strongly_killed(mut_id)
            forking_context.child_end()
        else:
            msg = f"Mainline value has exception, this indicates a not green test suite: {log_res}"
            logger.error(msg)
            raise ValueError(msg)

    try:
        return mutations[get_logical_path()]
    except KeyError:
        return mutations[MAINLINE]


def t_combine_split_stream(mutations: dict[int, Any]) -> Any:

    evaluated_mutations = {}
    for mut, res in mutations.items():
        if get_ns_active_mutants() is not None and mut != MAINLINE and mut not in get_ns_active_mutants():
            continue

        if type(res) != ShadowVariable and callable(res):
            try:
                res = res()
            except Exception as e:
                res = e

        evaluated_mutations[mut] = res

    mode = get_execution_mode()

    if mode.is_split_stream():
        res = combine_split_stream(evaluated_mutations)
    elif mode.is_modulo_eqv():
        res = combine_modulo_eqv(evaluated_mutations)
    else:
        raise NotImplementedError()
    return res


def t_combine(mutations: dict[int, Any]) -> Any:
    mode = get_execution_mode()
    if mode.is_shadow_variant():
        return t_combine_shadow(mutations)
    elif mode.is_split_stream_variant():
        return t_combine_split_stream(mutations)


# Init when importing shadow
reinit()
