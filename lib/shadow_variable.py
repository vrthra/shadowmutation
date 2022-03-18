# Implementation for wrapping variables in a manager for the different mutation values.

from __future__ import annotations
from copy import deepcopy
from itertools import chain
import traceback
import types
from typing import Any, Callable, Dict, Iterable, Tuple, TypeVar, Union

from lib.path import active_mutants, add_function_seen_mutants, add_masked_mutants, add_seen_mutants, add_strongly_killed, get_logical_path, get_masked_mutants, get_seen_mutants, get_selected_mutant, get_strongly_killed, set_selected_mutant
from lib.utils import MAINLINE, PRIMITIVE_TYPES, ShadowExceptionStop, ShadowException
from lib.mode import get_execution_mode


import logging
logger = logging.getLogger(__name__)


_NEW_NO_INIT = False


SV = TypeVar('SV', bound="ShadowVariable")


ALLOWED_UNARY_DUNDER_METHODS = {
    '__abs__':   lambda args, kwargs, a, _: abs(a, *args, **kwargs),
    '__round__': lambda args, kwargs, a, _: round(a, *args, **kwargs),
    '__neg__':   lambda args, kwargs, a, _: -a,
    '__len__':   lambda args, kwargs, a, _: len(a),
    '__index__': lambda args, kwargs, a, _: a.__index__(),
}

ALLOWED_BOOL_DUNDER_METHODS = {
    '__add__':        lambda args, kwargs, a, b: a + b,
    '__sub__':        lambda args, kwargs, a, b: a - b,
    '__truediv__':    lambda args, kwargs, a, b: a / b,
    '__floordiv__':   lambda args, kwargs, a, b: a // b,
    '__mul__':        lambda args, kwargs, a, b: a * b,
    '__pow__':        lambda args, kwargs, a, b: a ** b,
    '__mod__':        lambda args, kwargs, a, b: a % b,
    '__and__':        lambda args, kwargs, a, b: a & b,
    '__or__':         lambda args, kwargs, a, b: a | b,
    '__xor__':        lambda args, kwargs, a, b: a ^ b,
    '__lshift__':     lambda args, kwargs, a, b: a << b,
    '__rshift__':     lambda args, kwargs, a, b: a >> b,
    '__eq__':         lambda args, kwargs, a, b: a == b,
    '__ne__':         lambda args, kwargs, a, b: a != b,
    '__le__':         lambda args, kwargs, a, b: a <= b,
    '__lt__':         lambda args, kwargs, a, b: a < b,
    '__ge__':         lambda args, kwargs, a, b: a >= b,
    '__gt__':         lambda args, kwargs, a, b: a > b,

    # same methods but also do the reversed side (not sure all of these exist)
    '__radd__':       lambda args, kwargs, b, a: a + b,
    '__rsub__':       lambda args, kwargs, b, a: a - b,
    '__rtruediv__':   lambda args, kwargs, b, a: a / b,
    '__rfloordiv__':  lambda args, kwargs, b, a: a // b,
    '__rmul__':       lambda args, kwargs, b, a: a * b,
    '__rpow__':       lambda args, kwargs, b, a: a ** b,
    '__rmod__':       lambda args, kwargs, b, a: a % b,
    '__rand__':       lambda args, kwargs, b, a: a & b,
    '__ror__':        lambda args, kwargs, b, a: a | b,
    '__rxor__':       lambda args, kwargs, b, a: a ^ b,
    '__rlshift__':    lambda args, kwargs, b, a: a << b,
    '__rrshift__':    lambda args, kwargs, b, a: a >> b,
    '__req__':        lambda args, kwargs, b, a: a == b,
    '__rne__':        lambda args, kwargs, b, a: a != b,
    '__rle__':        lambda args, kwargs, b, a: a <= b,
    '__rlt__':        lambda args, kwargs, b, a: a < b,
    '__rge__':        lambda args, kwargs, b, a: a >= b,
    '__rgt__':        lambda args, kwargs, b, a: a > b,
}

PASSTHROUGH_DUNDER_METHODS = {
    # '__init__' is already manually defined, after instantiation of SV the passthrough will happen through __getattribute__
    '__iter__',
    '__next__',
    '__setitem__',
    '__getitem__',
}

DISALLOWED_DUNDER_METHODS = [
    '__aenter__', '__aexit__', '__aiter__', '__anext__', '__await__',
    '__bytes__', '__call__', '__cmp__', '__complex__', '__contains__',
    '__delattr__', '__delete__', '__delitem__', '__delslice__',  
    '__enter__', '__exit__', '__fspath__',
    '__get__', '__getslice__', 
    '__import__', '__imul__', 
    '__int__', '__invert__',
    '__ior__', '__ixor__', 
    '__nonzero__',
    '__pos__', '__prepare__', '__rdiv__',
    '__rdivmod__', '__repr__', '__reversed__',
    '__set__',
    '__setslice__', '__sizeof__', '__subclasscheck__', '__subclasses__',
    '__divmod__',
    '__div__',
    

    # python enforces that the specific type is returned,
    # to my knowledge we cannot override these dunders to return
    # a ShadowVariable
    # disallow these dunders to avoid accidentally losing taint info
    '__bool__', '__float__',

    # ShadowVariable needs to be pickleable for caching to work.
    # so '__reduce__', '__reduce_ex__', '__class__' are implemented.
    # For debugging '__dir__' is not disallowed.
]

LIST_OF_IGNORED_DUNDER_METHODS = [
    '__new__', '__init__', '__init_subclass__', '__instancecheck__', '__getattribute__', 
    '__setattr__', '__str__', '__format__', 
    '__iadd__', '__getnewargs__', '__getnewargs_ex__', '__iand__', '__isub__', 
]


def convert_method_to_function(obj: object, method_name: str) -> Tuple[Callable[..., Any], bool]:
    """Take an object and the name of a method, return the method without having the object associated,
    a simple function that allows changing the 'self' parameter. Also return a boolean that is true if the method is
    built-in and false if not."""
    method = obj.__getattribute__(method_name)
    assert callable(method)
    # convert bound method to free standing function, this allows changing the self argument
    if isinstance(method, types.BuiltinFunctionType):
        # If the called method is built there is no direct way to get the
        # underlying function using .__func__ or similar (at least after my research).
        # Instead use the object __getattribute__ on the type, this gives the function instead of the method.
        return object.__getattribute__(type(obj), method_name), True
    elif isinstance(method, types.MethodWrapperType):
        return object.__getattribute__(type(obj), method_name), True
    else:
        return method.__func__, False


def shadow_get(shadow, path) -> Any:
    if path in shadow:
        return shadow[path]
    elif MAINLINE in shadow:
        return shadow[MAINLINE]
    else:
        raise ValueError()


def not_allowed(obj, method, *args, **kwargs) -> None:
    logger.error("{method} %s %s %s", obj, args, kwargs)
    raise NotImplementedError("dunder method {method} is not allowed")


class ShadowVariable():
    _shadow: dict[int, Any]
    __slots__ = ['_shadow']

    for method in ALLOWED_UNARY_DUNDER_METHODS.keys():
        exec(f"""
    def {method}(self, *args, **kwargs):
        # assert len(args) == 0 and len(kwargs) == 0, f"{{len(args)}} == 0 and {{len(kwargs)}} == 0"
        return self._do_unary_op("{method}", *args, **kwargs)
        """.strip())

    # self.{method} = lambda other, *args, **kwargs: self._do_bool_op(other, *args, **kwargs)
    for method in ALLOWED_BOOL_DUNDER_METHODS.keys():
        exec(f"""
    def {method}(self, other, *args, **kwargs):
        assert len(args) == 0 and len(kwargs) == 0, f"{{len(args)}} == 0 and {{len(kwargs)}} == 0"
        return self._do_bool_op(other, "{method}", *args, **kwargs)
        """.strip())

    for method in DISALLOWED_DUNDER_METHODS:
        exec(f"""
    def {method}(self, *args, **kwargs):
        not_allowed(self, *args, **kwargs)
        """.strip())

    for method in PASSTHROUGH_DUNDER_METHODS:
        exec(f"""
    def {method}(self, *args, **kwargs):
        return self._callable_wrap("{method}", *args, **kwargs)
        """.strip())

    def _init_from_object(self, obj: Any) -> None:
        shadow = {}
        value_type = type(obj)

        # OOS: optionally make sure there are no nested shadow variables in the values
        if value_type == ShadowVariable:
            shadow = obj._shadow

        elif value_type == tuple:
            # handle tuple values
            combined: Dict[int, Any] = {}
            combined[MAINLINE] = []
            for elem in obj:
                if type(elem) == ShadowVariable:
                    elem_shadow = elem._shadow
                    # make a copy for each path that is new
                    for path in elem_shadow.keys():
                        if path not in combined:
                            combined[path] = deepcopy(combined[MAINLINE])

                    # append the corresponding path value for each known path
                    for path in combined.keys():
                        if path in elem_shadow:
                            combined[path].append(elem_shadow[path])
                        else:
                            combined[path].append(elem_shadow[MAINLINE])

                else:
                    for elems in combined.values():
                        elems.append(elem)

            # convert each path value back to a tuple
            for path in combined.keys():
                combined[path] = tuple(combined[path])

            shadow = combined

        elif value_type == list:
            # handle list values
            active = active_mutants()
            active.add(MAINLINE)
            combined = {}
            combined[MAINLINE] = []
            for ii, elem in enumerate(obj):
                if type(elem) == ShadowVariable:
                    elem_shadow = elem._shadow
                    # make a copy for each path that is new
                    for path in active:
                        if path not in combined:
                            if MAINLINE in combined:
                                combined[path] = deepcopy(combined[MAINLINE])
                            else:
                                if ii == 0:
                                    # Still on first element and there is no mainline value, this is ok.
                                    active.remove(MAINLINE)
                                    del combined[MAINLINE]
                                else:
                                    raise ValueError("Incomplete Shadow")


                    # append the corresponding path value for each known path
                    for path in list(active):
                        if path in elem_shadow:
                            combined[path].append(elem_shadow[path])
                        elif MAINLINE in elem_shadow:
                            combined[path].append(elem_shadow[MAINLINE])
                        elif path == MAINLINE:
                            active.remove(MAINLINE)
                            del combined[MAINLINE]
                        else:
                            raise ValueError("Incomplete Shadow")

                else:
                    for elems in combined.values():
                        elems.append(elem)

            shadow = combined

        elif value_type == set:
            combined = {}
            combined[MAINLINE] = []
            for elem in obj:
                if type(elem) == ShadowVariable:
                    elem_shadow = elem._shadow
                    # make a copy for each path that is new
                    for path in elem_shadow.keys():
                        if path not in combined:
                            combined[path] = deepcopy(combined[MAINLINE])

                    # append the corresponding path value for each known path
                    for path in combined.keys():
                        if path in elem_shadow:
                            combined[path].append(elem_shadow[path])
                        else:
                            combined[path].append(elem_shadow[MAINLINE])

                else:
                    for elems in combined.values():
                        elems.append(elem)

            # convert each path value back to a tuple
            for path in combined.keys():
                combined[path] = set(combined[path])

            shadow = combined

        elif value_type == dict:
            if len(obj) == 0:
                # Empty dict
                combined = {MAINLINE: {}}

            else:
                # This is a bit tricky as dict can have untainted and ShadowVariables as key and value.
                # A few examples on the left the dict obj and on the right (->) the resulting ShadowVariable (sv):
                # {0: 0}                                       -> sv(0: {0: 0})
                # {0: sv(0: 1, 1: 2)}                          -> sv(0: {0: 1}, 1: {0: 2})
                # {sv(0: 0, 1: 1): 0}                          -> sv(0: {0: 0}, 1: {1: 0})
                # {sv(0: 0, 1: 1, 2: 2): sv(0: 0, 2: 2, 3: 3)} -> sv(0: {0: 0}, 1: {1: 0}, 2: {2: 2}, 3: {0: 3})
                #
                # There can also be multiple key value pairs.
                # {0: 0, sv(0: 1, 2: 2): 2}                    -> sv(0: {0: 0, 1: 2}, 2: {0: 0, 2: 2})

                # First expand all possible combinations for each key value pair.
                all_expanded = []
                for key, data in obj.items():
                    # Get all paths for a key, value pair.
                    expanded = {}
                    if type(key) == ShadowVariable:
                        key_shadow = key._shadow
                        key_paths = set(key_shadow.keys())
                    else:
                        # If it is an untainted value, that is equivalent to the mainline path.
                        key_paths = set([MAINLINE])

                    if type(data) == ShadowVariable:
                        data_shadow = data._shadow
                        data_paths = set(data_shadow.keys())
                    else:
                        data_paths = set([MAINLINE])

                    all_paths = key_paths | data_paths

                    # Expand each combination.
                    for path in all_paths:
                        if type(key) == ShadowVariable:
                            if path in key_shadow:
                                path_key = key_shadow[path]
                            else:
                                path_key = key_shadow[MAINLINE]
                        else:
                            path_key = key

                        if type(data) == ShadowVariable:
                            if path in data_shadow:
                                path_data = data_shadow[path]
                            else:
                                path_data = data_shadow[MAINLINE]
                        else:
                            path_data = data

                        expanded[path] = (path_key, path_data)
                    all_expanded.append(expanded)

                # Get all paths that are needed in the resulting ShadowVariable.
                combined = {}
                for path in chain(*[paths.keys() for paths in all_expanded]):
                    combined[path] = {}

                # Build the resulting path dictionaries from the expanded combinations.
                for expanded in all_expanded:
                    for src_path, (key, val) in expanded.items():
                        # If the combination is for mainline add it to all dictionaries, if they do not
                        # have an alternative value.
                        if src_path == MAINLINE:
                            for trg_path, dict_values in combined.items():
                                if trg_path == MAINLINE or (
                                    trg_path != MAINLINE and trg_path not in expanded
                                ):
                                    assert key not in dict_values
                                    dict_values[key] = val
                        else:
                            path_dict = combined[src_path]
                            assert key not in path_dict
                            path_dict[key] = val

            shadow = combined
            
        else:
            shadow[MAINLINE] = obj

        self._shadow = shadow

    def _init_from_mapping(self, values: Dict[int, Any]) -> None:
        assert type(values) == dict

        combined = {}

        if MAINLINE in values:
            # Keep mainline as initial value and add other values from there.
            mainline_val = values[MAINLINE]
            if type(mainline_val) == ShadowVariable:
                combined = mainline_val._shadow
            else:
                combined = {}
                combined[MAINLINE] = mainline_val
        else:
            combined = {}

        for mut_id, val in values.items():
            if mut_id == MAINLINE:
                continue

            if type(val) == ShadowVariable:
                val = val._shadow
                combined[mut_id] = shadow_get(val, mut_id)
            else:
                assert mut_id not in combined
                combined[mut_id] = val

        self._shadow = combined

    def __init__(self, values: Any, from_mapping: bool):
        if not from_mapping:
            self._init_from_object(values)
        else:
            self._init_from_mapping(values)

    def _duplicate_mainline(self, new_path: int) -> None:
        assert new_path not in self._shadow

        mainline_val = self._shadow[MAINLINE]
        # make a copy for all shadow variants that will be needed
        copy = object.__new__(type(mainline_val))

        queue = [(mainline_val, copy, var) for var in dir(mainline_val)]
        while queue:
            cur_main, cur_copy, var_name = queue.pop(0)
            if var_name.startswith('_'):
                continue

            to_be_copied_var = cur_main.__getattribute__(var_name)

            try:
                existing_copy_var = cur_copy.__getattribute__(var_name)
            except AttributeError:
                # Var does not exist yet, just assign it.
                try:
                    cur_copy.__setattr__(var_name, deepcopy(to_be_copied_var))
                except TypeError:
                    raise NotImplementedError("Can not deepcopy variable.")
                # All done with this var.
                continue

            # Skip if is the same bound method as for mainline.
            if callable(existing_copy_var) and hasattr(existing_copy_var, '__self__'):
                assert existing_copy_var.__func__ == to_be_copied_var.__func__
                continue

            # OOS Implement copying of objects where attributes already exist.
            raise NotImplementedError()

        self._shadow[new_path] = copy

    def __setattr__(self, name: str, value: Any) -> Any:
        if name.startswith("_"):
            # logger.debug(f"super setattr {name, value}")
            return super(ShadowVariable, self).__setattr__(name, value)

        self_shadow = get_selected(self._shadow)

        # Need to handle ShadowVariable arguments differently than normal ones.
        if type(value) == ShadowVariable:
            other_shadow = get_selected(value._shadow)

            # duplicate mainline for all paths that are not already in mainline
            # note that this updates self_shadow, in particular also the keys used during the next _do_op_safely call
            for os in other_shadow:
                if os not in self_shadow:
                    self._duplicate_mainline(os)

            # Assign the respective value for each path separately.
            res = self._do_op_safely(
                {'': lambda _1, _2, obj, new_val: obj.__setattr__(name, new_val)},
                self_shadow.keys(),
                lambda k: self_shadow[k],
                lambda k: shadow_get(other_shadow, k),
                tuple(),
                dict(),
                '',
            )
        else:
            # Just assign the same value to all paths.
            res = self._do_op_safely(
                {'': lambda _1, _2, obj, new_val: obj.__setattr__(name, new_val)},
                self_shadow.keys(),
                lambda k: self_shadow[k],
                lambda _: value,
                tuple(),
                dict(),
                '',
            )

        return res

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            # __init__ is manually defined for ShadovVariable but can also be later called during usage of ShadowVariable.
            # In the second case we want to call __init__ on the path values instead.
            # This requires a special case here.
            if name != "__init__":
                return super(ShadowVariable, self).__getattribute__(name)
        
        log_res = self._get_logical_res(get_logical_path()).__getattribute__(name)
        
        if callable(log_res):
            # OOS returning a lambda here only works if the callable is only called
            # it does not work if the callable is compared to some other function or other edge cases
            # maybe return a dedicated object instead that raises errors for edge cases / implements them correctly
            return lambda *args, **kwargs: self._callable_wrap(name, *args, **kwargs)
            
        results = {}
        for path, val in self._shadow.items():
            try:
                res = val.__getattribute__(name)
            except:
                raise NotImplementedError()

            results[path] = res

        return ShadowVariable(results, from_mapping=True)

    def __hash__(self) -> int:
        """For ShadowVariable wrapped object, there are two contexts where __hash__ can be used.
        One is during normal usage of hash(sv) or sv.__hash__(), the returned value should be a ShadowVariable.
        (Note that the first version checks that the return value is a int, making this impossible.)
        The second usage is during built-in functions such as initialization of set or dict objects. To support these
        objects it is necessary to return an actual hash. (Or alternatively create a substitution class, however,
        this requires a alternative syntax for dictionaries, for example: {key: val} -> ShadowVariableSet([(key, val), ..]).
        These are solvable problems but out of scope.)
        
        Currently only a combined hash of the different path ids and path values is returned and the context where a
        ShadowVariable should be returned is ignored."""
        # OOS: How to detect that __hash__ is called in a context that should return a SV instead of the current implementation?
        return hash(tuple(self._shadow.items()))

    def _callable_wrap(self, name: str, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
        log_val = self._get_logical_res(get_logical_path())
        logical_func, is_builtin = convert_method_to_function(log_val, name)
        shadow = self._shadow

        if is_builtin:
            # Need some special handling if __next__ is called, it has influence on control flow.
            # StopIteration is raised once all elements during the iteration have been passed,
            # this can vary if we have different lengths for the wrapped variables.
            method_is_next = name == '__next__'
            if method_is_next:
                next_results: dict[int, bool] = {}

            # The method is a builtin, there can not be any mutations in the builtins.
            # Apply the method to each path value and combine the results into a ShadowValue and return that instead.
            untainted_args = untaint_args(*args, **kwargs)

            all_paths = set(shadow.keys()) | set(untainted_args.keys())
            results = {}
            for path in sorted(all_paths):  # Do mainline first.
                if path != MAINLINE and path not in active_mutants():
                    # Skip inactive mutants.
                    continue

                # Get the applicable path value.
                if path in shadow:
                    initial_path_val = shadow[path]
                else:
                    initial_path_val = log_val

                # Check that path and log would use the same function.
                path_func, _ = convert_method_to_function(initial_path_val, name)
                if path_func != logical_func:
                    raise NotImplementedError()

                # If the path value is the logical/mainline value copy it as it might be used several times.
                # Note that the copy step can change the actual function being called, for example a dict_keyiterator
                # will be turned into a list_iterator. For this reason, avoid copying for __next__.
                if not method_is_next:
                    path_val = deepcopy(initial_path_val)
                else:
                    path_val = initial_path_val

                # Get the arguments for the current path, otherwise use mainline.
                if path in untainted_args:
                    path_args, path_kwargs = deepcopy(untainted_args[path])
                else:
                    path_args, path_kwargs = deepcopy(untainted_args[MAINLINE])


                try:
                    results[path] = logical_func(path_val, *path_args, **path_kwargs)
                except IndexError as e:
                    if name == '__getitem__':
                        if path == MAINLINE:
                            if get_logical_path() == MAINLINE:
                                raise NotImplementedError()

                            # Not on mainline, kill all active mutants that do not have an alternative path here.
                            for mut in active_mutants() - all_paths:
                                add_strongly_killed(mut)
                        else:
                            add_strongly_killed(path)
                        continue
                    else:
                        message = traceback.format_exc()
                        logger.error(f"Error: {e} {message}")
                        raise NotImplementedError()
                except StopIteration as e:
                    if method_is_next:
                        next_results[path] = False
                        continue
                    else:
                        raise NotImplementedError()
                except Exception as e:
                    message = traceback.format_exc()
                    logger.error(f"Error: {message}")
                    raise NotImplementedError()

                if method_is_next:
                    next_results[path] = True

                if initial_path_val != path_val:
                    self._shadow[path] = path_val

            if method_is_next:
                # Handle different lengths of iterables.
                if t_cond(ShadowVariable(next_results, from_mapping=True)):
                    return ShadowVariable(results, from_mapping=True)
                else:
                    raise StopIteration

            return ShadowVariable(results, from_mapping=True)
        
        else:
            # Treat this method as a forkable function call, the self parameter is the ShadowVariable.
            diverging_mutants = []
            companion_mutants = []

            for path, val in self._shadow.items():
                if path == MAINLINE or path == get_logical_path():
                    continue

                val_func, _ = convert_method_to_function(val, name)
                if val_func == logical_func:
                    companion_mutants.append(path)
                else:
                    diverging_mutants.append(path)

            from .fork import get_forking_context
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
                        return self._callable_wrap(name, *args, **kwargs)
                    else:
                        add_masked_mutants(set(diverging_mutants))
            else:
                add_masked_mutants(set(diverging_mutants))

            from lib.func_wrap import t_wrap
            wrapped_fun = t_wrap(logical_func)
            return wrapped_fun(self, *args, **kwargs)

    def __repr__(self):
        return f"ShadowVariable({self._shadow})"

    def __getstate__(self) -> dict[int, Any]:
        return self._shadow

    def __setstate__(self, attributes):
        self._shadow = attributes

    def _get_paths(self):
        return self._shadow.keys()

    def _keep_active(self, seen: set[int], masked: set[int]) -> None:
        self._shadow = get_active(self._shadow, seen, masked)

    def _get_logical_res(self, logical_path: int) -> Any:
        shadow = self._shadow
        if logical_path in shadow:
            return shadow[logical_path]
        else:
            return shadow[MAINLINE]

    def _all_path_results(self, seen_mutants, masked_mutants):
        shadow = self._shadow
        paths = seen_mutants - masked_mutants - set([MAINLINE])  # self._get_paths()

        if MAINLINE in shadow:
            yield MAINLINE, shadow[MAINLINE]

        for path in paths:
            yield path, shadow_get(shadow, path)

    def _maybe_untaint(self) -> Union[SV, Any]:
        shadow = self._shadow
        # Only return a untainted version if shadow only contains the mainline value and that value is a primitive type.
        if len(shadow) == 1 and MAINLINE in shadow:
            mainline_type = type(self._shadow[MAINLINE])
            if mainline_type in PRIMITIVE_TYPES:
                return self._shadow[MAINLINE]

        return self

    def _normalize(self, seen: set[int], masked: set[int]) -> SV:
        self._keep_active(seen, masked)
        return self._maybe_untaint()

    def _merge(self, other: Any, seen: set[int], masked: set[int]):
        self_shadow = self._shadow
        other_type = type(other)
        if other_type == ShadowVariable:
            for path, val in other._all_path_results(seen, masked):
                if path != MAINLINE:
                    if path in self_shadow:
                        assert self_shadow[path] == val
                    else:
                        self_shadow[path] = val
        elif other_type == dict:
            assert False, f"merge with type not handled: {other}"
        else:
            for aa in seen - masked:
                self._add_mut_result(aa, other)
                if aa in self_shadow:
                    assert self_shadow[aa] == other
                else:
                    self_shadow[aa] = other

    def _maybe_overwrite(self, other: Any, seen: set[int], masked: set[int], including_main: bool):
        self_shadow = self._shadow
        other_type = type(other)
        if other_type == ShadowVariable:
            for path, val in other._all_path_results(seen, masked):
                if path == MAINLINE and not including_main:
                    continue
                if shadow_get(self_shadow, path) != val:
                    self_shadow[path] = val
        elif other_type == dict:
            assert False, f"merge with type not handled: {other}"
        else:
            if including_main:
                self_shadow[MAINLINE] = val
            for aa in seen - masked:
                assert aa != MAINLINE
                if shadow_get(self_shadow, aa) != other:
                    self_shadow[aa] = val

    def _copy(self) -> ShadowVariable:
        set_new_no_init()
        try:
            var = object.__new__(ShadowVariable)
            var._shadow = deepcopy(self._shadow)
        finally:
            unset_new_no_init()
        return var

    def _copy_and_prune_muts(self, muts: dict[int, Any]) -> ShadowVariable:
        "Copies shadow, does not modify in place."
        assert MAINLINE not in muts
        set_new_no_init()
        shadow = deepcopy(self._shadow)
        unset_new_no_init()
        for mut in muts:
            shadow.pop(mut, None)
        res = ShadowVariable(shadow, from_mapping=True)
        return res

    def _do_op_safely(self, ops: dict[str, Callable[..., Any]], paths: Iterable[int],
            left: Callable[..., Any], right: Callable[..., Any],
            args: tuple[Any, ...], kwargs: dict[str, Any], op: str) -> Any:
        global STRONGLY_KILLED

        if MAINLINE in paths:
            paths = set(paths)
            paths.remove(MAINLINE)
            first = set([MAINLINE])
        else:
            first = set()
        
        res = {}
        for k in (*first, *paths):
            if k != MAINLINE and k not in active_mutants():
                continue
            op_func = ops[op]
            try:
                k_res = op_func(args, kwargs, left(k), right(k))
            except (ZeroDivisionError, OverflowError, TypeError, ValueError) as e:
                if k != MAINLINE:
                    add_strongly_killed(k)
                else:
                    # logger.debug(f"mainline value exception {e}")
                    raise ShadowException(e)
                continue
            except Exception as e:
                logger.error(f"Unknown Exception: {e}")
                raise e

            res[k] = k_res

        return res

    def _do_unary_op(self, op: str, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> ShadowVariable:
        # logger.debug("op: %s %s", self, op)
        self_shadow = get_selected(self._shadow)
        # res = self._do_op_safely(self_shadow.keys(), lambda k: self_shadow[k].__getattribute__(op)(*args, **kwargs))
        res = self._do_op_safely(
            ALLOWED_UNARY_DUNDER_METHODS,
            self_shadow.keys(),
            lambda k: self_shadow[k],
            lambda k: None,
            args,
            kwargs,
            op,
        )
        # logger.debug("res: %s %s", res, type(res[0]))
        return ShadowVariable(res, from_mapping=True)

    def _do_bool_op(self, other, op, *args, **kwargs):
        assert len(args) == 0 and len(kwargs) == 0, f"{args} {kwargs}"

        # logger.debug("op: %s %s %s", self, other, op)
        self_shadow = get_selected(self._shadow)
        if type(other) == ShadowVariable:
            other_shadow = get_selected(other._shadow)
            # notice that both self and other has taints.
            # the result we need contains taints from both.
            masked_shadows = get_masked_mutants() - set([MAINLINE])
            common_shadows = {k for k in self_shadow if k in other_shadow} - masked_shadows
            only_self_shadows = self_shadow.keys() - common_shadows - set([MAINLINE]) - masked_shadows
            only_other_shadows = other_shadow.keys() - common_shadows - set([MAINLINE]) - masked_shadows

            if only_self_shadows:
                other_main = other_shadow[MAINLINE]
                vs_ = self._do_op_safely(
                    ALLOWED_BOOL_DUNDER_METHODS,
                    only_self_shadows,
                    lambda k: self_shadow[k],
                    lambda k: other_main,
                    args,
                    kwargs,
                    op,
                )
            else:
                vs_ = {}

            if only_other_shadows:
                self_main = self_shadow[MAINLINE]
                vo_ = self._do_op_safely(
                    ALLOWED_BOOL_DUNDER_METHODS,
                    only_other_shadows,
                    lambda k: self_main,
                    lambda k: other_shadow[k],
                    args,
                    kwargs,
                    op,
                )
            else:
                vo_ = {}

            # if there was a pre-existing taint of the same name, this mutation was
            # already executed. So, use that value.
            cs_ = self._do_op_safely(
                ALLOWED_BOOL_DUNDER_METHODS,
                common_shadows,
                lambda k: self_shadow[k],
                lambda k: other_shadow[k],
                args,
                kwargs,
                op,
            )
            res = {**vs_, **vo_, **cs_}
            # logger.debug("res: %s", res)
            return ShadowVariable(res, from_mapping=True)
        else:
            res = self._do_op_safely(
                ALLOWED_BOOL_DUNDER_METHODS,
                self_shadow.keys(),
                lambda k: self_shadow[k],
                lambda k: other,
                args,
                kwargs,
                op,
            )
            # logger.debug("res: %s %s", res, type(res[0]))
            return ShadowVariable(res, from_mapping=True)


def t_class(orig_class):
    orig_new = orig_class.__new__

    def wrap_new(cls, *args, **kwargs):
        new = orig_new(cls)

        if _NEW_NO_INIT:
            # If loading from pickle (happens when combining forks), no need to wrap in ShadowVariable
            return new

        else:
            # For actual usage wrap the object inside a ShadowVariable
            obj = ShadowVariable(new, False)

            # only call __init__ if instance of cls is returned
            # https://docs.python.org/3/reference/datamodel.html#object.__new__
            if isinstance(new, cls):
                obj.__init__(*args, **kwargs)

            return obj

    orig_class._orig_new = orig_new
    orig_class.__new__ = wrap_new
    return orig_class


def untaint_args(*args: tuple[Union[ShadowVariable, Any], ...], **kwargs: dict[str, Union[ShadowVariable, Any]]) -> dict[int, Any]:
    """Get a mapping of each path to args and kwargs for each path available from the ShadowVariable in the arguments."""
    all_muts = set([MAINLINE])
    for arg in args + tuple(kwargs.values()):
        if type(arg) == ShadowVariable:
            all_muts |= arg._get_paths()

    mainline_incomplete = False

    untainted_args = {}
    for mut in all_muts:

        mut_args = []
        for arg in args:
            if type(arg) == ShadowVariable:
                arg_shadow = arg._shadow
                if mut in arg_shadow:
                    mut_args.append(arg_shadow[mut])
                elif MAINLINE in arg_shadow:
                    mut_args.append(arg_shadow[MAINLINE])
                else:
                    mainline_incomplete = True
                    continue
            else:
                mut_args.append(arg)

        mut_kwargs = {}
        for name, arg in kwargs.items():
            if type(arg) == ShadowVariable:
                arg_shadow = arg._shadow
                if mut in arg_shadow:
                    mut_kwargs[name] = arg_shadow[mut]
                elif MAINLINE in arg_shadow:
                    mut_kwargs[name] = arg_shadow[MAINLINE]
                else:
                    mainline_incomplete = True
                    continue
            else:
                mut_kwargs[name] = arg

        untainted_args[mut] = (tuple(mut_args), dict(mut_kwargs))

    # Could not get a value for mainline for every argument. (Some sv do not have a mainline value)
    if mainline_incomplete:
        # This should only happen for non-mainline paths.
        assert get_logical_path() != MAINLINE
        del untainted_args[MAINLINE]

    return untainted_args


def get_selected(mutations: dict[int, Any]) -> dict[int, Any]:
    if get_selected_mutant() is not None:
        return { path: val for path, val in mutations.items() if path in [MAINLINE, get_selected_mutant()] }
    else:
        return mutations


def get_active(mutations: dict[int, Any], seen: set[int], masked: set[int]) -> dict[int, Any]:
    filtered_mutations = { path: val for path, val in mutations.items() if path in seen - masked }

    # logger.debug(f"log_path: {get_logical_path()}")
    if MAINLINE in mutations:
        filtered_mutations[MAINLINE] = mutations[MAINLINE]
    if get_logical_path() in mutations:
        filtered_mutations[get_logical_path()] = mutations[get_logical_path()]

    return filtered_mutations


def get_active_shadow(val: Any, seen: set[int], masked: set[int]) -> Union[dict[int, Any], None]:
    if type(val) == ShadowVariable:
        return get_active(val._shadow, seen, masked)

    else:
        return None


def t_combine_shadow(mutations: dict[int, Any]) -> Any:

    if get_logical_path() == MAINLINE:
        add_seen_mutants(set(mutations.keys()) - get_masked_mutants() - set([MAINLINE]))
        add_function_seen_mutants(set(mutations.keys()))

    evaluated_mutations = {}
    for mut, res in mutations.items():
        # Skip inactive mutants
        if (mut not in active_mutants()) and mut != MAINLINE:
            continue

        if type(res) != ShadowVariable and callable(res):
            if mut != MAINLINE:
                set_selected_mutant(mut)
            try:
                res = res()
            except ShadowExceptionStop as e:
                raise e
            except ShadowException as e:
                # Mainline value causes an exception
                if mut == MAINLINE:
                    if get_logical_path() == MAINLINE:
                        # The current path is mainline, in this case the original evaluation caused an exception.
                        # Meaning, the test suite is not green, just raise the error nothing that can be done.
                        raise e

                    # Not following mainline but mainline value fails, paths that do not have an own value in the
                    # mutation list would fail as well, mark them as killed.
                    for active_mut in active_mutants():
                        if active_mut not in mutations:
                            add_strongly_killed(active_mut)

                # Non-Mainline value causes an exception.
                else:
                    if mut not in get_strongly_killed():
                        # Just mark it as killed
                        add_strongly_killed(mut)

                continue
            except Exception as e:
                raise e
            finally:
                set_selected_mutant(None)

        evaluated_mutations[mut] = res

    res = ShadowVariable(evaluated_mutations, from_mapping=True)
    res._keep_active(get_seen_mutants(), get_masked_mutants())
    return res


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

        from .fork import get_forking_context
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

    elif type(cond) == bool:
        return cond
    
    else:
        raise ValueError(f"Unhandled t_cond type: {cond}")


def shadow_assert(cmp_result):
    if type(cmp_result) == ShadowVariable:
        # Do the actual assertion as would be done in the unchanged program but only for mainline execution
        if get_logical_path() == MAINLINE:
            # This assert should never fail for a green test suite
            assert cmp_result._shadow[MAINLINE] is True, f"{cmp_result}"

        for path, res in cmp_result._all_path_results(get_seen_mutants(), get_masked_mutants()):
            if path == MAINLINE and get_logical_path() != MAINLINE:
                continue
            assert type(res) == bool
            if not res: # assert fails for mutation
                add_strongly_killed(path)

    else:
        if not cmp_result is True:
            if get_logical_path() is not MAINLINE:
                # If we are not following mainline, mark all active mutants as killed
                for mut in active_mutants():
                    add_strongly_killed(mut)
            else:
                # If we are following mainline the test suite is not green
                assert cmp_result, f"Failed original assert"


def set_new_no_init() -> None:
    global _NEW_NO_INIT
    _NEW_NO_INIT = True


def unset_new_no_init() -> None:
    global _NEW_NO_INIT
    _NEW_NO_INIT = False


def copy_args(args, kwargs):
    set_new_no_init()
    copied = deepcopy((args, kwargs))
    unset_new_no_init()
    return copied