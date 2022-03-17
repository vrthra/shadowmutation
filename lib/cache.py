# Caching function calls.


from itertools import chain
import os
from pathlib import Path
import pickle
import tempfile
import traceback
from typing import Any, Iterable, Union
from lib.utils import MAINLINE, ShadowException, ShadowExceptionStop
from lib.mode import get_execution_mode
from lib.path import active_mutants, add_function_seen_mutants, add_masked_mutants, add_seen_mutants, get_logical_path, get_masked_mutants, get_seen_mutants, remove_masked_mutants, set_logical_path, reset_function_seen, reset_function_masked, get_function_seen
from lib.shadow_variable import ShadowVariable, copy_args, set_new_no_init, unset_new_no_init, untaint_args
from lib.fork import get_forking_context
import logging
logger = logging.getLogger(__name__)


_CACHE_PATH: Union[str, None] = None
_PARENT_FUNCTION_SEEN: list[set[int]] = []


def reinit_cache() -> None:
    global _CACHE_PATH
    global _PARENT_FUNCTION_SEEN
    mode = get_execution_mode()

    if mode.uses_cache():
        fd, name = tempfile.mkstemp()
        os.close(fd)
        _CACHE_PATH = name
        _PARENT_FUNCTION_SEEN = []
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
    global _PARENT_FUNCTION_SEEN
    if _CACHE_PATH is not None:
        fun_seen = get_function_seen()
        _PARENT_FUNCTION_SEEN.append(set())
        for par in _PARENT_FUNCTION_SEEN:
            par |= fun_seen
        reset_function_seen()
        cache, mut_stack = load_cache()
        mut_stack.append(set())
        save_cache(cache, mut_stack)


def pop_cache_stack():
    if _CACHE_PATH is not None:
        # Restore the currently seen muts to adding those of the calling function.
        par_seen = _PARENT_FUNCTION_SEEN.pop()
        add_function_seen_mutants(par_seen)
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


def reset_path_variables(starting_logical, before_logical, added_mask):
    remove_masked_mutants(added_mask)

    logical_path_after_func = get_logical_path()
    if logical_path_after_func != starting_logical:
        # Logical path was changed during execution, do not reset, keep the new path.
        pass
    else:
        # Restore logical path
        set_logical_path(before_logical)


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

        # For each path check if function / args combo is cached.
        before_logical_path = get_logical_path()
        is_cached = []
        for mut, (mut_args, mut_kwargs) in untainted_args.items():
            # Never load cached mainline if we are on mainline.
            if mut == MAINLINE == get_logical_path():
                continue
            key = f"{f.__name__, mut_args, mut_kwargs}"  # OOS more efficient key

            if key in cache:
                cached_return_res, cached_arguments, cached_seen_muts = cache[key]
                if mut == MAINLINE:
                    # All possible active mutants that get their value from mainline, that is there are no other 
                    # paths in untainted for that active mutant.
                    cached = active_mutants() - untainted_args.keys()
                    # Of those mutants only get the cached value if the mutant is not seen during execution
                    cached -=  cached_seen_muts
                elif mut not in cached_seen_muts:
                    cached = set([mut])
                else:
                    continue

                is_cached.append((cached, cached_return_res, cached_arguments, cached_seen_muts))

        mut_is_cached = {}
        if len(untainted_args.keys() - set(chain(*[cc[0] for cc in is_cached]))) == 0:
            for cc in is_cached:
                cached, cached_return_res, cached_arguments, cached_seen_muts = cc
                add_masked_mutants(cached - set([MAINLINE]))
                add_seen_mutants(cached_seen_muts - set([MAINLINE]))
                for mm in cached:
                    mut_is_cached[mm] = (cached_return_res, cached_arguments)

                    # The current followed path can be gotten from cache.
                    if mm != MAINLINE and mm == get_logical_path():
                        act_muts = active_mutants()

                        if not act_muts:
                            break

                        set_logical_path(act_muts.pop())
        else:
            assert len(mut_is_cached) == 0

        # Remove paths from sv arguments that are cached.
        args, kwargs = prune_cached_muts(mut_is_cached.keys(), *args, **kwargs)

        starting_logical_path = get_logical_path()

        # If current path is mainline or some args are not cached then execute the function.
        if get_logical_path() == MAINLINE or \
                len(set(untainted_args) - set(mut_is_cached)) > 0:
            # Get a copy of the args before executing the function as they could be changed.
            if get_logical_path() == MAINLINE:
                mainline_args_copy = copy_args(*untainted_args[MAINLINE])
            else:
                mainline_args_copy = None

            got_stop = False

            try:
                res = f(*args, **kwargs)
            except ShadowException as e:
                message = f"{e}, {traceback.format_exc()}"
                logger.error(f"Error: {message}")
                raise NotImplementedError(f"Exceptions in wrapped functions are not supported: {message}")
            except ShadowExceptionStop as e:
                # Got no result from function execution, reraise if nothing is cached, else return that.
                if len(mut_is_cached) == 0:
                    raise e
                else:
                    got_stop = True
            except RecursionError as e:
                raise e
            except Exception as e:
                f"{e}, {traceback.format_exc()}"
                message = f"{e}, {traceback.format_exc()}"
                logger.error(f"Error: {message}")
                raise NotImplementedError(f"Exceptions in wrapped functions are not supported: {message}")
            finally:
                reset_path_variables(starting_logical_path, before_logical_path, mut_is_cached.keys())

            if got_stop:
                # There is no return value, make an empty one.
                res_shadow = {}
            else:
                # Convert result to SV.
                res = ShadowVariable(res, from_mapping=False)
                res._keep_active(get_seen_mutants(), get_masked_mutants())
                res_shadow = res._shadow

        else:
            # Executing the function has been skipped.
            reset_path_variables(starting_logical_path, before_logical_path, mut_is_cached.keys())
            res_shadow = {}

        # Update result with cached values.
        for mut, (cached_return_res, _) in mut_is_cached.items():
            if mut == MAINLINE:
                if get_logical_path() == MAINLINE:
                    assert MAINLINE in res_shadow
            if MAINLINE in mut_is_cached:
                # Do not add to res if mainline does have the same result, to reduce taints.
                if mut_is_cached[MAINLINE][0] == cached_return_res:
                    try:
                        del res_shadow[mut]
                    except:
                        pass
                    continue
            if mut in res_shadow:
                assert res_shadow[mut] == cached_return_res
            else:
                res_shadow[mut] = cached_return_res
        res = ShadowVariable(res_shadow, from_mapping=True)

        # Update res and arguments
        assert type(res) == ShadowVariable
        res_shadow = res._shadow
        args_shadows = [(ii, aa._shadow) for ii, aa in enumerate(args) if type(aa) == ShadowVariable]
        kwargs_shadows = [(kk, aa._shadow) for kk, aa in kwargs.items() if type(aa) == ShadowVariable]

        # Update arguments with cached values.
        for mut, (_, (cached_args, cached_kwargs)) in mut_is_cached.items():
            if mut == MAINLINE:
                if get_logical_path() == MAINLINE:
                    assert MAINLINE in res_shadow
                continue

            for ii, aa in args_shadows:
                if MAINLINE in mut_is_cached and mut_is_cached[MAINLINE][1][0][ii] == cached_args[ii]:
                    try:
                        del aa[mut]
                    except:
                        pass
                else:
                    assert mut not in aa
                    aa[mut] = cached_args[ii]

            for ii, aa in kwargs_shadows:
                if MAINLINE in mut_is_cached and mut_is_cached[MAINLINE][1][1][ii] == cached_kwargs[ii]:
                    try:
                        del aa[mut]
                    except:
                        pass
                else:
                    assert mut not in aa
                    aa[mut] = cached_kwargs[ii]


        if get_logical_path() == MAINLINE:
            # update cache for new results
            cache, mut_stack = load_cache()
            mainline_res = res._shadow[MAINLINE]
            assert mainline_args_copy is not None
            mainline_args, mainline_kwargs = mainline_args_copy
            key = f"{f.__name__, mainline_args, mainline_kwargs}" # OOS more efficient key
            if key not in cache:
                function_seen_mutants = get_function_seen() - set([MAINLINE])
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
            message = f"Error: {e} {traceback.format_exc()}"
            logger.error(message)
            raise NotImplementedError(f"Exceptions in wrapped functions are not supported: {message}")

        res = ShadowVariable(res, from_mapping=False)
        res._keep_active(get_seen_mutants(), get_masked_mutants())
        return res


# OOS potentially there can be a sync point before the function starts to try and execute as many paths in that function as once