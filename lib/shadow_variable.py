# Implementation for wrapping variables in a manager for the different mutation values.

from __future__ import annotations
from copy import deepcopy
from itertools import chain
import os
import traceback
import types
from typing import Any, Callable, Dict, Iterable, Tuple, TypeVar, Union

from lib.path import active_mutants, add_function_seen_mutants, add_masked_mutants, add_seen_mutants, add_strongly_killed, get_logical_path, get_masked_mutants, get_seen_mutants, get_selected_mutant, set_selected_mutant
from lib.utils import MAINLINE, PRIMITIVE_TYPES, ShadowExceptionStop
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
        logger.error("{method} %s %s %s", self, args, kwargs)
        raise ValueError("dunder method {method} is not allowed")
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
                if mut_id in val:
                    combined[mut_id] = val[mut_id]
                else:
                    combined[mut_id] = val[MAINLINE]
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
                # var does not exist yet
                cur_copy.__setattr__(var_name, to_be_copied_var)
            else:
                # TODO implement copying of complex nested objects
                raise NotImplementedError()
                # if to_be_copied_var == existing_copy_var:
                #     continue

                # try:
                #     copied_var = deepcopy(copied_var)
                # except TypeError:
                #     raise NotImplementedError()

                # logger.debug(f"{copied_var}")

                # cur_copy.__setattr__(var_name, copied_var)

                #     parts = dir(cur)
                #     logger.debug(f"{parts}")
                #     for part in parts:
                #         if part.startswith("_"):
                #             continue
                #         cur_path.append(part)
                #         val = cur.__getattribute__(part)
                #         logger.debug(f"{cur_path} = {val}")
                #         part_type = type(val)
                #         if part_type == ShadowVariable:
                #             logger.debug(f"found sv: {cur_path}")
                #             locations.append((deepcopy(cur_path), val))
                #             contained_shadow_paths |= set(val._get_paths())
                #         else:
                #             raise NotImplementedError(f"Unhandled type: {part_type}")
                #         # get list of all locations that contain shadow variables
                #         # make recursive, with loop detection

                # return locations, contained_shadow_paths

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
                lambda k: value._get(k),
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
                lambda k: value,
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
            # Currently assert that all variables always either return something or raise StopIteration in the same call.
            method_is_next = name == '__next__'
            if method_is_next:
                next_returns = 0
                next_raises = 0

            # The method is a builtin, there can not be any mutations in the builtins.
            # Apply the method to each path value and combine the results into a ShadowValue and return that instead.
            untainted_args = untaint_args(*args, **kwargs)

            # Get the arguments for the current logical path, otherwise use mainline.
            if get_logical_path() in untainted_args:
                log_args = untainted_args[get_logical_path()]
            else:
                log_args = untainted_args[MAINLINE]

            all_paths = set(shadow.keys()) | set(untainted_args.keys())
            results = {}
            for path in all_paths:
                if path == MAINLINE and get_logical_path() != MAINLINE and get_logical_path() in all_paths:
                    continue

                # Otherwise the value might be used several times, in that case make a copy.
                if path in shadow:
                    path_val = shadow[path]
                else:
                    path_val = log_val

                # If the path value is the logical/mainline value copy it as it might be used several times.
                # Note that the copy step can change the actual function being called, for example a dict_keyiterator
                # will be turned into a list_iterator. For this reason, avoid copying for __next__, there is no need
                # for a copy anyway as __next__ does not take arguments.
                if not method_is_next and (path == MAINLINE or path == get_logical_path()):
                    path_val = deepcopy(path_val)

                # Check that path and log would use the same function.
                path_func, _ = convert_method_to_function(path_val, name)
                if path_func != logical_func:
                    raise NotImplementedError()

                # As for path val, make a copy if needed.
                if path != MAINLINE and path != get_logical_path() and path in untainted_args:
                    path_args, path_kwargs = untainted_args[path] or untainted_args.get(MAINLINE)
                else:
                    path_args, path_kwargs = deepcopy(log_args)

                try:
                    results[path] = logical_func(path_val, *path_args, **path_kwargs)
                except StopIteration as e:
                    if method_is_next:
                        next_raises += 1
                        continue
                    else:
                        raise NotImplementedError()
                except Exception as e:
                    message = traceback.format_exc()
                    logger.error(f"Error: {message}")
                    raise NotImplementedError()

                if method_is_next:
                    next_returns += 1

                self._shadow[path] = path_val

            if method_is_next:
                # OOS: Currently all iterables are expected to iterate the same amount of times.
                # To support unequal amounts this function needs to support forking.
                assert next_returns == 0 or next_raises == 0
                if next_raises > 0:
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

            from lib.fork import get_forking_context
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

    def _get(self, mut) -> Any:
        return self._shadow[mut]

    def _get_logical_res(self, logical_path: int) -> Any:
        if logical_path in self._shadow:
            return self._shadow[logical_path]
        else:
            return self._shadow[MAINLINE]

    def _all_path_results(self, seen_mutants, masked_mutants):
        shadow = self._shadow
        paths = seen_mutants - masked_mutants - set([MAINLINE])  # self._get_paths()

        if MAINLINE in shadow:
            yield MAINLINE, shadow[MAINLINE]

        for path in paths:
            if path in shadow:
                yield path, shadow[path]
            else:
                yield path, shadow[MAINLINE]
            # yield path, self._get(path)

        # for path in seen_mutants - masked_mutants - paths:
        #     yield path, self._get(MAINLINE)

    def _add_mut_result(self, mut: int, res: Any) -> None:
        assert mut not in self._shadow
        self._shadow[mut] = res

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
        other_type = type(other)
        if other_type == ShadowVariable:
            for path, val in other._all_path_results(seen, masked):
                if path != MAINLINE:
                    self._add_mut_result(path, val)
        elif other_type == dict:
            assert False, f"merge with type not handled: {other}"
        else:
            for aa in seen - masked:
                self._add_mut_result(aa, other)

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
        
        res = {}
        for k in paths:
            op_func = ops[op]
            try:
                k_res = op_func(args, kwargs, left(k), right(k))
            except (ZeroDivisionError, OverflowError) as e:
                if k != MAINLINE:
                    add_strongly_killed(k)
                else:
                    # logger.debug(f"mainline value exception {e}")
                    raise e
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
            common_shadows = {k for k in self_shadow if k in other_shadow}
            only_self_shadows = self_shadow.keys() - common_shadows - set([MAINLINE])
            only_other_shadows = other_shadow.keys() - common_shadows - set([MAINLINE])

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
        if (mut not in get_seen_mutants() or mut in get_masked_mutants()) and mut != MAINLINE:
            continue
        if type(res) != ShadowVariable and callable(res):
            if mut != MAINLINE:
                set_selected_mutant(mut)
            try:
                res = res()
            except ShadowExceptionStop as e:
                raise e
            except Exception as e:
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
                    # Just mark it as killed
                    add_strongly_killed(mut)

                continue
            finally:
                set_selected_mutant(None)

        evaluated_mutations[mut] = res

    if get_execution_mode().is_shadow_variant():
        res = ShadowVariable(evaluated_mutations, from_mapping=True)
        res._keep_active(get_seen_mutants(), get_masked_mutants())
    else:
        raise NotImplementedError()
    return res


def shadow_assert(cmp_result):
    if type(cmp_result) == ShadowVariable:
        # Do the actual assertion as would be done in the unchanged program but only for mainline execution
        if get_logical_path() == MAINLINE:
            # This assert should never fail for a green test suite
            assert cmp_result._get(MAINLINE) is True, f"{cmp_result}"

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