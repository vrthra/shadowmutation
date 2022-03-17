# Wrapping functions.

from copy import deepcopy
from functools import wraps
from itertools import chain
from lib.utils import MAINLINE, ShadowException, ShadowExceptionStop
from lib.fork import get_forking_context, new_forking_context, set_forking_context
from lib.path import active_mutants, get_logical_path, get_masked_mutants, get_seen_mutants, get_strongly_killed, try_next_logical_path, set_logical_path, set_masked_mutants
from lib.shadow_variable import ShadowVariable, copy_args, set_new_no_init, unset_new_no_init
from lib.cache import push_cache_stack, pop_cache_stack, call_maybe_cache

import logging
logger = logging.getLogger(__name__)

def fork_wrap(f, *args, **kwargs):

    old_forking_context = get_forking_context()
    old_masked_mutants = deepcopy(get_masked_mutants())

    forking_context = new_forking_context()

    push_cache_stack()
    try:
        res = call_maybe_cache(f, *args, **kwargs)
    except ShadowExceptionStop:
        # Mainline should stop, there are not results. Reset forking context and stop path.
        set_forking_context(old_forking_context)
        try_next_logical_path()
        raise ValueError("This should never be reached.")

    mainline_result, fork_results = forking_context.wait_for_forks(fork_res=(res, args, kwargs))
    pop_cache_stack()

    # Filter args and kwargs for currently available, they will be updated with the fork values.
    for arg in args:
        if type(arg) == ShadowVariable:
            arg._keep_active(get_seen_mutants(), get_masked_mutants())

    for arg in kwargs.values():
        if type(arg) == ShadowVariable:
            arg._keep_active(get_seen_mutants(), get_masked_mutants())

    set_forking_context(old_forking_context)
    set_masked_mutants(old_masked_mutants | get_strongly_killed())

    res = ShadowVariable(mainline_result, from_mapping=True)

    for child_res in fork_results:
        seen = child_res['seen']
        masked = child_res['masked']
        fork_res = child_res['fork_res']
        if fork_res is not None:
            child_fork_res, child_fork_args, child_fork_kwargs = fork_res
            res._merge(child_fork_res, seen, masked)

            # Update the args with the fork values, this is for functions that mutate the arguments.
            for ii, val in child_fork_args.items():
                args[ii]._merge(val, seen, masked)

            for key, val in child_fork_kwargs.items():
                kwargs[key]._merge(val, seen, masked)


    # If only mainline in return value untaint it
    res._keep_active(get_seen_mutants(), get_masked_mutants())
    return res._maybe_untaint()


def no_fork_wrap(f, *args, **kwargs):
    initial_args, initial_kwargs = copy_args(args, kwargs)
    before_logical_path = get_logical_path()
    before_masked = deepcopy(get_masked_mutants())

    remaining_paths = set([get_logical_path()])
    done_paths = set()

    for arg in chain(args, kwargs.values()):
        if type(arg) == ShadowVariable:
            remaining_paths |= arg._get_paths()

    remaining_paths -= before_masked
    if before_logical_path != MAINLINE and MAINLINE in remaining_paths:
        remaining_paths.remove(MAINLINE)

    tainted_return = {}
    push_cache_stack()
    while remaining_paths:
        if before_logical_path in remaining_paths:
            remaining_paths.remove(before_logical_path)
            next_path = before_logical_path
        else:
            next_path = remaining_paths.pop()

        set_masked_mutants((before_masked | done_paths | get_strongly_killed()) - set((next_path,)))
        set_logical_path(next_path)

        if get_logical_path() == before_logical_path:
            copied_args, copied_kwargs = args, kwargs
        else:
            copied_args, copied_kwargs = copy_args(initial_args, initial_kwargs)

        # Filter args and kwargs for currently available, they will be updated with the fork values.
        for arg in copied_args:
            if type(arg) == ShadowVariable:
                arg._keep_active(get_seen_mutants(), get_masked_mutants())

        for arg in copied_kwargs.values():
            if type(arg) == ShadowVariable:
                arg._keep_active(get_seen_mutants(), get_masked_mutants())

        try:
            res = call_maybe_cache(f, *copied_args, **copied_kwargs)
        except ShadowExceptionStop as e:
            remaining_paths -= get_strongly_killed()
            continue 
        except ShadowException as e:
            logger.debug(f"shadow exception: {e}")
            remaining_paths -= get_strongly_killed()
            continue 
        after_masked = deepcopy(get_masked_mutants())
        new_masked = after_masked - before_masked

        assert type(res) == ShadowVariable
        shadow = res._shadow

        # Update results for the current execution.
        if get_logical_path() != MAINLINE and before_logical_path == get_logical_path() and MAINLINE in shadow:
            tainted_return[MAINLINE] = shadow[MAINLINE]

        for active_mut in active_mutants() | set([get_logical_path()]):
            assert active_mut not in tainted_return
            if active_mut in shadow:
                tainted_return[active_mut] = shadow[active_mut]
            elif MAINLINE in shadow:
                tainted_return[active_mut] = shadow[MAINLINE]
            else:
                # Do not add to done paths.
                continue
            done_paths.add(active_mut)

        # Update the args with the fork values, this is for functions that mutate the arguments.
        overwrite_main = before_logical_path == get_logical_path()
        for ii, val in enumerate(copied_args):
            arg = args[ii]
            if type(arg) == ShadowVariable:
                arg._maybe_overwrite(val, get_seen_mutants(), get_masked_mutants(), overwrite_main)

        for key, val in copied_kwargs.items():
            arg = kwargs[key]
            if type(arg) == ShadowVariable:
                arg._maybe_overwrite(val, get_seen_mutants(), get_masked_mutants(), overwrite_main)

        # Update remaining paths.
        remaining_paths |= new_masked
        remaining_paths -= (done_paths | get_strongly_killed())

    pop_cache_stack()

    set_masked_mutants(before_masked | get_strongly_killed())

    # If there are no more active mutants and there is nothing to return just stop immediately.
    if len(tainted_return) == 0:
        assert get_logical_path() != MAINLINE
        assert len(active_mutants()) == 0
        try_next_logical_path()

    if before_logical_path == MAINLINE or before_logical_path in active_mutants():
        set_logical_path(before_logical_path)
    else:
        try_next_logical_path()

    res = ShadowVariable(tainted_return, from_mapping=True)
    return res._maybe_untaint()


def t_wrap(f):
    @wraps(f)
    def flow_wrapper(*args, **kwargs):
        if get_forking_context() is not None:
            return fork_wrap(f, *args, **kwargs)
        else:
            return no_fork_wrap(f, *args, **kwargs)

    f._is_shadow_wrapped = True
    return flow_wrapper