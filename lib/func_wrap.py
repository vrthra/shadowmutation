# Wrapping functions.

from copy import deepcopy
from functools import wraps
from itertools import chain
from lib.constants import MAINLINE, ShadowException, ShadowExceptionStop
from lib.fork import get_forking_context, new_forking_context, set_forking_context
from lib.path import active_mutants, get_logical_path, get_masked_mutants, get_seen_mutants, get_strongly_killed, set_logical_path, set_masked_mutants
from lib.shadow_variable import ShadowVariable, set_new_no_init, unset_new_no_init
from lib.cache import push_cache_stack, pop_cache_stack, call_maybe_cache

import logging
logger = logging.getLogger(__name__)

def fork_wrap(f, *args, **kwargs):

    # logger.debug(f"CALL {f.__name__}({args} {kwargs}) seen: {get_seen_mutants()} masked: {get_masked_mutants()}")
    old_forking_context = get_forking_context()
    old_masked_mutants = deepcopy(get_masked_mutants())

    forking_context = new_forking_context()

    push_cache_stack()
    res = call_maybe_cache(f, *args, **kwargs)
    combined_results = forking_context.wait_for_forks(fork_res=(res, args, kwargs))
    pop_cache_stack()

    # Filter args and kwargs for currently available, they will be updated with the fork values.
    for arg in args:
        if type(arg) == ShadowVariable:
            arg._keep_active(get_seen_mutants(), get_masked_mutants())

    for arg in kwargs.values():
        if type(arg) == ShadowVariable:
            arg._keep_active(get_seen_mutants(), get_masked_mutants())

    set_forking_context(old_forking_context)
    set_masked_mutants(old_masked_mutants)

    res = ShadowVariable(combined_results[0], from_mapping=True)

    for child_res in combined_results[1:]:
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
    return res._maybe_untaint()


def copy_args(args, kwargs):
    set_new_no_init()
    copied = deepcopy((args, kwargs))
    unset_new_no_init()
    return copied


def no_fork_wrap(f, *args, **kwargs):
    # TODO copy args and update them with changes
    # logger.debug(f"CALL {f.__name__}({args} {kwargs})")
    initial_args, initial_kwargs = copy_args(args, kwargs)
    before_logical_path = get_logical_path()
    before_active = active_mutants()
    before_masked = deepcopy(get_masked_mutants())

    remaining_paths = set([get_logical_path()])
    done_paths = set()

    for arg in chain(args, kwargs.values()):
        if type(arg) == ShadowVariable:
            remaining_paths |= arg._get_paths()
    remaining_paths -= before_masked

    tainted_return = {}
    push_cache_stack()
    while remaining_paths:
        if before_logical_path in remaining_paths:
            set_logical_path(before_logical_path)
            remaining_paths.remove(before_logical_path)
        else:
            set_logical_path(remaining_paths.pop())
        # logger.debug(f"cur path: {get_logical_path()} remaining: {remaining_paths}")
        set_masked_mutants((deepcopy(before_masked) | done_paths) - set((get_logical_path(), )))

        if get_logical_path() == before_logical_path:
            copied_args, copied_kwargs = args, kwargs
        else:
            copied_args, copied_kwargs = copy_args(initial_args, initial_kwargs)

        try:
            res = call_maybe_cache(f, *copied_args, **copied_kwargs)
        except ShadowExceptionStop as e:
            remaining_paths -= get_strongly_killed()
            continue 
        except ShadowException as e:
            logger.debug(f"shadow exception: {e}")
            remaining_paths -= get_strongly_killed()
            continue 
        after_active = active_mutants()
        after_masked = deepcopy(get_masked_mutants())
        # logger.debug('wrapped: %s(%s %s) -> %s (%s)', f.__name__, args, kwargs, res, type(res))
        new_active = after_active - before_active
        new_masked = after_masked - before_masked

        assert type(res) == ShadowVariable
        shadow = res._shadow

        # Update results for the current execution.
        for active_mut in active_mutants() | set([get_logical_path()]):
            assert active_mut not in tainted_return
            if active_mut in shadow:
                tainted_return[active_mut] = shadow[active_mut]
            else:
                tainted_return[active_mut] = shadow[MAINLINE]
            done_paths.add(active_mut)

        if get_logical_path() == before_logical_path:
            # Filter args and kwargs for currently available, they will be updated with the fork values.
            for arg in args:
                if type(arg) == ShadowVariable:
                    arg._keep_active(get_seen_mutants(), get_masked_mutants())

            for arg in kwargs.values():
                if type(arg) == ShadowVariable:
                    arg._keep_active(get_seen_mutants(), get_masked_mutants())

        else:
            # Update the args with the fork values, this is for functions that mutate the arguments.
            for ii, val in enumerate(copied_args):
                if type(val) == ShadowVariable:
                    args[ii]._merge(val, get_seen_mutants(), get_masked_mutants())

            for key, val in copied_kwargs.items():
                if type(val) == ShadowVariable:
                    kwargs[key]._merge(val, get_seen_mutants(), get_masked_mutants())

        done_paths |= new_active
        done_paths.add(get_logical_path())
        remaining_paths |= new_masked
        remaining_paths -= done_paths

    pop_cache_stack()

    set_logical_path(before_logical_path)
    set_masked_mutants(before_masked)

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