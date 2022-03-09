# Caching function calls.


import os
from pathlib import Path
import pickle
import tempfile
import traceback
from typing import Any, Iterable, Union
from lib.constants import MAINLINE, ShadowException, ShadowExceptionStop
from lib.mode import get_execution_mode
from lib.path import get_masked_mutants, get_seen_mutants
from lib.shadow_variable import ShadowVariable, untaint_args
from lib.fork import get_forking_context
import logging
logger = logging.getLogger(__name__)


_CACHE_PATH: Union[str, None] = None


def reinit_cache() -> None:
    global _CACHE_PATH
    mode = get_execution_mode()

    if mode.uses_cache():
        fd, name = tempfile.mkstemp()
        os.close(fd)
        _CACHE_PATH = name
    else:
        _CACHE_PATH = None
    pass


def wrap_active_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    # Takes a tuple that can be used as *args during a function call and taints all elements.
    # Does not modify the tuple in any other way.
    wrapped = []
    for arg in args:
        if type(arg) == ShadowVariable:
            wrapped.append(arg)
        else:
            wrap: Any = ShadowVariable(arg, False)._normalize(get_seen_mutants(), get_masked_mutants())
            wrapped.append(wrap)
    return tuple(wrapped)


def wrap_active_kwargs(args: dict[str, Any]) -> dict[str, Any]:
    # Takes a dict that can be used as **kwargs during a function call and taints all elements.
    # Does not modify the dict in any other way.
    wrapped = {}
    for name, arg in args.items():
        if type(arg) == ShadowVariable:
            wrapped[name] = arg
        else:
            wrap: Any = ShadowVariable(arg, False)._normalize(get_seen_mutants(), get_masked_mutants())
            wrapped[name] = wrap
    return wrapped


def maybe_clean_cache() -> None:
    if _CACHE_PATH:
        forking_context = get_forking_context()
        if forking_context is not None:
            if forking_context.is_parent:
                Path(_CACHE_PATH).unlink()


def prune_cached_muts(muts: set[int], *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    muts = set(muts) - set([MAINLINE])
    # assert MAINLINE not in muts

    arg_values: Iterable[Any] = args + tuple(kwargs.values())
    for arg in arg_values:
        if type(arg) == ShadowVariable:
            arg._prune_muts(muts)
            
    return args, kwargs


def load_cache():
    with open(_CACHE_PATH, 'rb') as cache_f:
        try:
            cache, mut_stack = pickle.load(cache_f)
        except EOFError:
            # Cache has no content yet.
            cache = {}
            mut_stack = []
    return cache, mut_stack


def save_cache(cache, mut_stack):
    with open(_CACHE_PATH, 'wb') as cache_f:
        try:
            pickle.dump((cache, mut_stack), cache_f)
        except TypeError:
            raise ValueError(f"Can't serialize: {cache} {mut_stack}")


def push_cache_stack():
    if _CACHE_PATH is not None:
        cache, mut_stack = load_cache()
        mut_stack.append(set())
        save_cache(cache, mut_stack)


def pop_cache_stack():
    if _CACHE_PATH is not None:
        cache, mut_stack = load_cache()
        mut_stack.pop()
        save_cache(cache, mut_stack)


def maybe_mark_mutation(mutations):
    if _CACHE_PATH is not None:
        cache, mut_stack = load_cache()
        muts = mutations.keys() - set([MAINLINE])
        # logger.debug(f"{cache, mut_stack}")

        for ii in range(len(mut_stack)):
            mut_stack[ii] = mut_stack[ii] | muts

        # logger.debug(f"{cache, mut_stack}")
        save_cache(cache, mut_stack)


def function_is_cacheable(func):
    try:
        func._is_shadow_wrapped
    except AttributeError:
        return True
    
    return False


def call_maybe_cache(f, *args, **kwargs):
    # TODO how to apply call_maybe_cache to non wrapped functions? -> check for _is_shadow_wrapped but where? they are not wrapped

    if _CACHE_PATH is not None:
        # OOS: caching for mainline executions, this requires knowing all mutations that are executed as part
        # of the called function, recursively. Then caching can be done for paths following mutations that not part
        # of that function.

        # TODO Check that the function is cacheable, otherwise just execute (maybe in if earlier).
        is_cacheable = function_is_cacheable(f)
        logger.debug(f"{f.__name__} {is_cacheable}")

        # Get arguments to function call for each path.
        untainted_args = untaint_args(*args, **kwargs)

        # TODO For each path check if function / args combo is cached, otherwise execute.
        result_is_cached = None

        if result_is_cached:
            # TODO Result is cached just get it
            pass
        else:
            # TODO Execute function normally for the given arguments and cache return value as well as changes in arguments.
            pass

        # TODO Update return value as well as ShadowVariable arguments.

        # TODO If cache is updated also write to the cache

        raise NotImplementedError()

        cache, mut_stack = load_cache()

        # logging.debug(f"in: {args} {kwargs} untainted: {untainted_args}")
        # logger.debug(f"cache: {cache} mut_stack: {mut_stack}")

        mut_is_cached = {}
        for mut, (mut_args, mut_kwargs) in untainted_args.items():

            mode = get_execution_mode()

            if mode == ExecutionMode.SHADOW_CACHE:
                if mut == MAINLINE:
                    continue
            elif mode == ExecutionMode.SHADOW_FORK_CACHE:
                if get_logical_path() == MAINLINE and mut == MAINLINE:
                    continue
            else:
                raise ValueError(f"Unexpected execution mode: {mode}")

            key = f"{f.__name__, mut_args, mut_kwargs}"
            if key in cache:
                cache_res = cache[key]
                mut_is_cached[mut] = cache_res
                # logger.debug(f"cached res: {key}, {cache_res}")
            
        logger.debug(f"{get_logical_path()} {get_seen_mutants()} {get_masked_mutants()} {len(mut_is_cached)} {len(untainted_args)}")
        if len(mut_is_cached) == len(untainted_args):
            # all results are cached, no need to execute function
            res = ShadowVariable(mut_is_cached, from_mapping=True)
        else:
            try:
                args, kwargs = prune_cached_muts(mut_is_cached.keys(), *args, **kwargs)
                # logger.debug(f"pruned: {args, kwargs}")
                res = f(*args, **kwargs)
            except ShadowException as e:
                res = e
            except Exception as e:
                raise NotImplementedError(f"Exceptions in wrapped functions are not supported: {e}")

            # update cache for new results
            cache, mut_stack = load_cache()

            cache_updated = False
            if type(res) == ShadowVariable:
                for mut in res._shadow:
                    # only cache if mut in input args and not introduced by called function
                    if mut in untainted_args and mut not in mut_stack[-1]:
                        mut_args, mut_kwargs = untainted_args[mut]
                        key = f"{f.__name__, mut_args, mut_kwargs}"
                        cache[key] = res._shadow[mut]
                        cache_updated = True
                        # logger.debug(f"cache res: {key}, {res}")
            elif type(res) == ShadowException:
                for mut in res._shadow:
                    # only cache if mut in input args and not introduced by called function
                    if mut in untainted_args and mut not in mut_stack[-1]:
                        mut_args, mut_kwargs = untainted_args[mut]
                        key = f"{f.__name__, mut_args, mut_kwargs}"
                        if key not in cache:
                            cache[key] = res
                            cache_updated = True
                            logger.debug(f"cache res: {key}")
            else:
                mut_args, mut_kwargs = untainted_args[MAINLINE]
                key = f"{f.__name__, mut_args, mut_kwargs}"
                if key not in cache:
                    cache[key] = res
                    cache_updated = True
                    # logger.debug(f"cache res: {key}, {res}")
                res = ShadowVariable({MAINLINE: res}, from_mapping=True)

            if cache_updated:
                save_cache(cache, mut_stack)

            # insert cached results
            # logger.debug(f"{mut_is_cached}, {res}")
            res = ShadowVariable({**mut_is_cached, **res._shadow}, from_mapping=True)

        # logger.debug(f"{res}")
        return res

    else:
        # no caching, just do it normally
        try:
            active_args = wrap_active_args(args)
            active_kwargs = wrap_active_kwargs(kwargs)
            # logger.debug(f"{f} {len(active_args), len(active_kwargs)} {active_args} {active_kwargs}")
            res = f(*active_args, **active_kwargs)
        except ShadowExceptionStop as e:
            raise e
        except ShadowException as e:
            raise e
        except Exception as e:
            message = traceback.format_exc()
            logger.error(f"Error: {message}")
            raise NotImplementedError(f"Exceptions in wrapped functions are not supported: {e}")

        res = ShadowVariable(res, from_mapping=False)
        res._keep_active(get_seen_mutants(), get_masked_mutants())
        return res


# TODO potentially there can be a sync point before the function starts to try and execute as many paths in that
# function as once