# Caching function calls.


import os
from pathlib import Path
import pickle
import tempfile
import traceback
from typing import Any, Iterable, Union
from lib.utils import MAINLINE, ShadowException, ShadowExceptionStop
from lib.mode import get_execution_mode
from lib.path import active_mutants, add_function_seen_mutants, add_masked_mutants, add_seen_mutants, get_function_seen_masked, get_logical_path, get_masked_mutants, get_seen_mutants, remove_masked_mutants, reset_function_seen_masked, set_logical_path
from lib.shadow_variable import ShadowVariable, copy_args, set_new_no_init, unset_new_no_init, untaint_args
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


def maybe_clean_cache() -> None:
    if _CACHE_PATH:
        forking_context = get_forking_context()
        if forking_context is not None:
            if forking_context.is_parent():
                Path(_CACHE_PATH).unlink()


def prune_cached_muts(muts: set[int], *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    # Always keep mainline value if it is available.
    muts = set(muts) - set([MAINLINE])

    arg_values: Iterable[Any] = args + tuple(kwargs.values())
    for arg in arg_values:
        if type(arg) == ShadowVariable:
            # Remove cached muts from shadow.
            shadow = arg._shadow
            for mut in muts:
                shadow.pop(mut, None)
            
    return args, kwargs


def load_cache():
    with open(_CACHE_PATH, 'rb') as cache_f:
        try:
            set_new_no_init()
            cache, mut_stack = pickle.load(cache_f)
        except EOFError:
            # Cache has no content yet.
            cache = {}
            mut_stack = []
        finally:
            unset_new_no_init()
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


def function_is_wrapped(func):
    try:
        func._is_shadow_wrapped
    except AttributeError:
        return False
    
    return True


def reset_path_variables(starting_logical, before_logical, before_seen, added_mask):
    # To the currently seen muts also restore those of the calling function.
    add_function_seen_mutants(before_seen)

    logical_path_after_func = get_logical_path()
    if logical_path_after_func != starting_logical:
        # Logical path was changed during execution, do not reset, keep the new path.
        pass
    else:
        # Restore logical path
        set_logical_path(before_logical)

    # Reset masked mutants to those that were already masked adding those that are new from the current call.
    _, in_function_added_mask = get_function_seen_masked()
    remove_masked_mutants(added_mask - in_function_added_mask)


def call_maybe_cache(f, *args, **kwargs):
    # OOS how to apply call_maybe_cache to non wrapped functions?

    if _CACHE_PATH is not None:
        # OOS: caching for mainline executions, this requires knowing all mutations that are executed as part
        # of the called function, recursively. Then caching can be done for paths following mutations that not part
        # of that function.

        # Get arguments to function call for each path.
        untainted_args = untaint_args(*args, **kwargs)
        for mm in get_masked_mutants():
            if mm == MAINLINE:
                continue
            if mm in untainted_args:
                del untainted_args[mm]

        # Load cache
        cache, mut_stack = load_cache()

        before_function_seen, before_function_masked = get_function_seen_masked()

        # For each path check if function / args combo is cached.
        before_logical_path = get_logical_path()
        mut_is_cached = {}
        for mut, (mut_args, mut_kwargs) in untainted_args.items():
            # Never load cached mainline if we are on mainline.
            if mut == MAINLINE == get_logical_path():
                continue
            key = f"{f.__name__, mut_args, mut_kwargs}"

            if key in cache:
                cached_return_res, cached_arguments, cached_seen_muts = cache[key]
                if mut not in cached_seen_muts:
                    add_seen_mutants(cached_seen_muts - set([MAINLINE]))
                    mut_is_cached[mut] = (cached_return_res, cached_arguments)

                    # The current followed path can be gotten from cache.
                    if mut != MAINLINE and mut == get_logical_path():
                        add_masked_mutants(set([mut]))
                        act_muts = active_mutants()

                        if not act_muts:
                            break

                        set_logical_path(act_muts.pop())

        reset_function_seen_masked()

        # Remove paths from sv arguments that are cached.
        args, kwargs = prune_cached_muts(mut_is_cached.keys(), *args, **kwargs)

        starting_logical_path = get_logical_path()

        # If current path is mainline or some args are not cached then execute the function.
        if get_logical_path() == MAINLINE or \
                set(untainted_args) == set([MAINLINE]) or \
                set(untainted_args) - set(mut_is_cached) - set([MAINLINE]):
            # Get a copy of the args before executing the function as they could be changed.
            if get_logical_path() == MAINLINE:
                mainline_args_copy = copy_args(*untainted_args[MAINLINE])
            else:
                mainline_args_copy = None

            try:
                res = f(*args, **kwargs)
            except ShadowException as e:
                message = f"{e}, {traceback.format_exc()}"
                logger.error(f"Error: {message}")
                raise NotImplementedError(f"Exceptions in wrapped functions are not supported: {message}")
            except ShadowExceptionStop as e:
                raise e
            except RecursionError as e:
                raise e
            except Exception as e:
                f"{e}, {traceback.format_exc()}"
                message = f"{e}, {traceback.format_exc()}"
                logger.error(f"Error: {message}")
                raise NotImplementedError(f"Exceptions in wrapped functions are not supported: {message}")
            finally:
                function_seen_mutants, _ = get_function_seen_masked()
                reset_path_variables(starting_logical_path, before_logical_path, before_function_seen, mut_is_cached.keys())

            assert get_logical_path() not in get_masked_mutants()
            # Convert result to SV.
            res = ShadowVariable(res, from_mapping=False)
            res._keep_active(get_seen_mutants(), get_masked_mutants())

            # Update result with cached values.
            res_shadow = res._shadow
            for mut, (cached_return_res, _) in mut_is_cached.items():
                if mut == MAINLINE:
                    if get_logical_path() == MAINLINE:
                        assert MAINLINE in res_shadow
                    continue
                assert mut not in res_shadow
                res_shadow[mut] = cached_return_res
        else:
            function_seen_mutants, _ = get_function_seen_masked()
            reset_path_variables(starting_logical_path, before_logical_path, before_function_seen, mut_is_cached.keys())

            # Initialize result with cached values.
            cached_mapping = {mut: cached_return_res for mut, (cached_return_res, _) in mut_is_cached.items()}
            res = ShadowVariable(cached_mapping, from_mapping=True)
            res._keep_active(get_seen_mutants(), get_masked_mutants())

        assert get_logical_path() not in get_masked_mutants()

        # Update res and arguments
        assert type(res) == ShadowVariable
        res_shadow = res._shadow
        # res = ShadowVariable(res, from_mapping=False)
        args_shadows = [(ii, aa._shadow) for ii, aa in enumerate(args) if type(aa) == ShadowVariable]
        kwargs_shadows = [(kk, aa._shadow) for kk, aa in kwargs.items() if type(aa) == ShadowVariable]

        # Update arguments with cached values.
        for mut, (_, (cached_args, cached_kwargs)) in mut_is_cached.items():
            if mut == MAINLINE:
                if get_logical_path() == MAINLINE:
                    assert MAINLINE in res_shadow
                continue

            for ii, aa in args_shadows:
                assert mut not in aa
                aa[mut] = cached_args[ii]

            for ii, aa in kwargs_shadows:
                assert mut not in aa
                aa[mut] = cached_kwargs[ii]


        if get_logical_path() == MAINLINE:
            # update cache for new results
            cache, mut_stack = load_cache()

            if type(res) == ShadowVariable:
                mainline_res = res._shadow[MAINLINE]
                assert mainline_args_copy is not None
                mainline_args, mainline_kwargs = mainline_args_copy
                key = f"{f.__name__, mainline_args, mainline_kwargs}"
                if key not in cache:
                    cache[key] = (mainline_res, untaint_args(*args, **kwargs)[MAINLINE], function_seen_mutants)
                    save_cache(cache, mut_stack)

        # Finally return result
        return res

    else:
        # no caching, just do it normally
        try:
            res = f(*args, **kwargs)
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


# OOS potentially there can be a sync point before the function starts to try and execute as many paths in that function as once