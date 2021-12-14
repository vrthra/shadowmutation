from json.decoder import JSONDecodeError
import os
import sys
import json
from collections import defaultdict
from enum import Enum
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from copy import deepcopy
from functools import wraps, partial
from contextlib import contextmanager

import logging
from typing import Union
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(process)d %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MAINLINE = 0
LOGICAL_PATH = 0

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class ExecutionMode(Enum):
    NOT_SPECIFIED = 0
    SPLIT_STREAM = 1 # split stream execution
    MODULO_EQV = 2 # split stream + modulo equivalence pruning execution
    SHADOW = 3 # shadow types and no forking
    SHADOW_FORK = 4 # shadow types + forking

    def get_mode(mode):
        if mode is None:
            return ExecutionMode.NOT_SPECIFIED
        elif mode == 'split':
            return ExecutionMode.SPLIT_STREAM
        elif mode == 'modulo':
            return ExecutionMode.MODULO_EQV
        elif mode == 'shadow':
            return ExecutionMode.SHADOW
        elif mode == 'shadow_fork':
            return ExecutionMode.SHADOW_FORK
        else:
            raise ValueError("Unknown Execution Mode", mode)

    def should_start_forker(self):
        if self in [ExecutionMode.SPLIT_STREAM, ExecutionMode.MODULO_EQV, ExecutionMode.SHADOW_FORK]:
            return True
        else:
            return False

    def is_split_stream_variant(self):
        if self in [ExecutionMode.SPLIT_STREAM, ExecutionMode.MODULO_EQV]:
            return True
        else:
            return False


class Forker():
    def __init__(self):
        self.is_parent = True
        self.sync_dir = Path(mkdtemp())
        (self.sync_dir/'paths').mkdir()
        (self.sync_dir/'results').mkdir()

    def my_pid(self):
        return os.getpid()

    def maybe_fork(self, path):
        global LOGICAL_PATH

        # # Don't fork if current process is a child process, all forks need to depart from mainline.
        # if not self.is_parent:
        #     return False

        # Only fork once, from then on follow that path.
        path_file = self.sync_dir.joinpath('paths', str(path))
        if path_file.is_file():
            return False
        path_file.touch()

        # Try to fork
        forked_pid = os.fork()
        logger.debug(f"Forking for path: {path} got pid: {forked_pid}")
        if forked_pid == -1:
            # Error during forking. Not much we can do.
            raise ValueError(f"Could not fork for path: {path}!")
        elif forked_pid != 0:
            # We are in parent, record the child pid and path.
            path_file.write_text(str(forked_pid))
            try:
                os.waitpid(forked_pid, 0)
            except ChildProcessError as e:
                pass
            return False
        else:
            # Update that this is the child.
            self.is_parent = False

            # Update which path child is supposed to follow
            LOGICAL_PATH = path
            return True

    def wait_for_forks(self):
        # if child, write results and exit
        if not self.is_parent:
            pid = self.my_pid()
            path = LOGICAL_PATH
            logger.debug(f"Child with pid: {pid} and path: {path} "
                         f"has reached sync point.")
            res_path = self.sync_dir/'results'/str(pid)
            logger.debug(f"Writing results to: {res_path}")
            with open(res_path, 'wt') as f:
                results = t_get_killed()
                for res in ['strong', 'weak', 'active', 'masked']:
                    results[res] = list(results[res])
                results['pid'] = pid
                results['path'] = path
                json.dump(results, f)

            # exit the child immediately, this might cause problems for programs actually using multiprocessing
            # but this is a prototype and this fixes problems with pytest
            os._exit(0)

        # wait for all child processes to end
        all_results = t_get_killed()
        while True:
            is_done = True
            for path_file in (self.sync_dir/'paths').glob("*"):
                is_done = False
                try:
                    child_pid = int(path_file.read_text())
                except ValueError:
                    continue

                logger.debug(f"Waiting for pid: {child_pid}")

                try:
                    os.waitpid(child_pid, 0)
                except ChildProcessError as e:
                    pass

                result_file = self.sync_dir/'results'/str(child_pid)
                if result_file.is_file():
                    with open(result_file, 'rt') as f:
                        try:
                            child_results = json.load(f)
                        except JSONDecodeError:
                            # Child has not yet written the results.
                            continue

                    for res in ['strong', 'weak', 'active']:
                        child_results[res] = set(child_results[res])

                    for res in ['strong', 'weak']:
                        add_res = child_results[res] & child_results['active']
                        all_results[res] |= add_res

                    logger.debug(f"child results: {child_results}")

                    path_file.unlink()
                    result_file.unlink()
            
            if is_done:
                break
        logger.debug(f"Done waiting for forks.")

        (self.sync_dir/'paths').rmdir()
        (self.sync_dir/'results').rmdir()
        self.sync_dir.rmdir()


STRONGLY_KILLED = None
WEAKLY_KILLED = None
ACTIVE_MUTANTS = None
MASKED_MUTANTS = None
EXECUTION_MODE = None
RESULT_FILE = None
FORKING_CONTEXT: Union[None, Forker] = None


def reinit(logical_path: str=None, execution_mode: Union[None, str]=None):
    logger.info("Reinit global shadow state")
    # initializing shadow
    global LOGICAL_PATH
    global STRONGLY_KILLED
    global WEAKLY_KILLED
    global ACTIVE_MUTANTS
    global MASKED_MUTANTS
    global EXECUTION_MODE
    global FORKING_CONTEXT
    global RESULT_FILE

    RESULT_FILE = os.environ.get('RESULT_FILE')

    if logical_path is not None:
        LOGICAL_PATH = logical_path
    else:
        LOGICAL_PATH = int(os.environ.get('LOGICAL_PATH', MAINLINE))

    if execution_mode is not None:
        EXECUTION_MODE = ExecutionMode.get_mode(execution_mode)
    else:
        EXECUTION_MODE = ExecutionMode.get_mode(os.environ.get('EXECUTION_MODE'))

    WEAKLY_KILLED = set()
    STRONGLY_KILLED = set()
    ACTIVE_MUTANTS = None
    MASKED_MUTANTS = set()

    if EXECUTION_MODE.should_start_forker():
        logger.debug("Initializing forker")
        FORKING_CONTEXT = Forker()
    else:
        FORKING_CONTEXT = None


# Init when importing shadow
reinit()


def t_wait_for_forks():
    global FORKING_CONTEXT
    if FORKING_CONTEXT is not None:
        FORKING_CONTEXT.wait_for_forks()


def t_get_killed():
    return {
        'strong': STRONGLY_KILLED,
        'weak': WEAKLY_KILLED,
        'active': ACTIVE_MUTANTS,
        'masked': MASKED_MUTANTS,
    }


def t_gather_results():
    t_wait_for_forks()

    results = t_get_killed()
    results['execution_mode'] = EXECUTION_MODE.name
    if RESULT_FILE is not None:
        with open(RESULT_FILE, 'wt') as f:
            json.dump(results, f, cls=SetEncoder)
    return results


def t_get_logical_path():
    return LOGICAL_PATH


def t_cond(cond):
    global WEAKLY_KILLED
    global FORKING_CONTEXT
    global ACTIVE_MUTANTS
    global MASKED_MUTANTS

    shadow = get_active_shadow(cond)

    if shadow is not None:
        logger.debug(f"shadow {shadow} {LOGICAL_PATH}")

        logical_path_is_diverging = False

        diverging_mutants = []
        companion_mutants = []
        

        mainline_res = shadow[MAINLINE]
        if LOGICAL_PATH in shadow:
            res = shadow[LOGICAL_PATH]
            assert type(res) == bool
        else:
            res = mainline_res

        assert type(mainline_res) == bool

        logger.debug("t_cond: (%s, %s) %s", LOGICAL_PATH, res, shadow)

        for path, path_val in shadow.items():
            if path_val != res:
                if path == MAINLINE:
                    logical_path_is_diverging = True
                    continue
                diverging_mutants.append(path)
                if LOGICAL_PATH == MAINLINE:
                    logger.info(f"t_cond weakly_killed: {path}")
                    WEAKLY_KILLED.add(path)
                if ACTIVE_MUTANTS is not None:
                    ACTIVE_MUTANTS.discard(path)
                MASKED_MUTANTS.add(path)
            else:
                companion_mutants.append(path)

        if FORKING_CONTEXT:
            # Fork if enabled
            if diverging_mutants:
                logger.debug(f"diverging mutants: {diverging_mutants}")
                path = diverging_mutants[0]
                if FORKING_CONTEXT.maybe_fork(path):
                    # this execution is in forked child
                    MASKED_MUTANTS = set()
                    ACTIVE_MUTANTS = set()
                    for path in diverging_mutants:
                        ACTIVE_MUTANTS.add(path)
                    return t_cond(cond)
        else:
            # Follow the logical path, if that is not the same as mainline mark other mutations as inactive
            if logical_path_is_diverging:
                if ACTIVE_MUTANTS is None:
                    ACTIVE_MUTANTS = set()
                    for path in companion_mutants:
                        ACTIVE_MUTANTS.add(path)

        return res
    else:
        return cond


def get_active(mutations):
    logger.debug(f"{mutations}")
    filtered_mutations = {
        path: val for path, val in mutations.items()
        if path not in STRONGLY_KILLED and path not in WEAKLY_KILLED and path not in MASKED_MUTANTS
    }

    if ACTIVE_MUTANTS is not None:
        filtered_mutations = { path: val for path, val in mutations.items() if path in ACTIVE_MUTANTS }

    logger.debug(f"log_path: {LOGICAL_PATH}")
    filtered_mutations[MAINLINE] = mutations[MAINLINE]
    if LOGICAL_PATH in mutations:
        filtered_mutations[LOGICAL_PATH] = mutations[LOGICAL_PATH]

    return filtered_mutations

def get_active_shadow(val):
    if type(val) == ShadowVariable:
        return get_active(val._shadow)

    else:
        return None


def shadow_assert(cmp_result):
    global STRONGLY_KILLED
    global WEAKLY_KILLED
    shadow = get_active_shadow(cmp_result)
    logger.debug(f"t_assert {cmp_result} {shadow}")
    if shadow is not None:
        for path, val in shadow.items():
            # The mainline assertion is done after the for loop
            if path == MAINLINE:
                continue
            if not val:
                STRONGLY_KILLED.add(path)
                logger.info(f"t_assert strongly killed: {path}")

        if ACTIVE_MUTANTS is not None and not shadow[MAINLINE]:
            assert LOGICAL_PATH in ACTIVE_MUTANTS
            for mut in ACTIVE_MUTANTS:
                if mut not in shadow:
                    STRONGLY_KILLED.add(mut)
                    logger.info(f"t_assert strongly killed: {path}")

        # Do the actual assertion as would be done in the unchanged program but only for mainline execution
        if LOGICAL_PATH == MAINLINE:
            assert shadow[MAINLINE]
    else:
        if not cmp_result:
            for mut in ACTIVE_MUTANTS:
                STRONGLY_KILLED.add(mut)
                logger.info(f"t_assert strongly killed: {mut}")


def split_assert(cmp_result):
    logger.debug(f"t_assert {cmp_result}")
    assert type(cmp_result) == bool
    if cmp_result:
        return
    else:
        for mut in ACTIVE_MUTANTS:
            STRONGLY_KILLED.add(mut)
            logger.info(f"t_assert strongly killed: {mut}")


def t_assert(cmp_result):
    if EXECUTION_MODE in [ExecutionMode.SHADOW, ExecutionMode.SHADOW_FORK]:
        shadow_assert(cmp_result)
    elif EXECUTION_MODE in [ExecutionMode.SPLIT_STREAM, ExecutionMode.MODULO_EQV]:
        split_assert(cmp_result)
    else:
        raise ValueError("Unknown execution mode: {EXECUTION_MODE}")


def untaint(obj):
    if hasattr(obj, '_shadow'):
        return obj._shadow[MAINLINE]
    return obj


def combine_split_stream(mutations):
    global ACTIVE_MUTANTS
    if LOGICAL_PATH == MAINLINE:
        for mut_id, val in mutations.items():
            if mut_id in [MAINLINE, LOGICAL_PATH]:
                continue
            if FORKING_CONTEXT.maybe_fork(mut_id):
                ACTIVE_MUTANTS = set([mut_id])
                return val
    try:
        return mutations[LOGICAL_PATH]
    except KeyError:
        return mutations[MAINLINE]


def combine_modulo_eqv(mutations):
    global ACTIVE_MUTANTS
    global MASKED_MUTANTS

    mutations = get_active(mutations)

    log_res = mutations[MAINLINE]
    if LOGICAL_PATH in mutations:
        log_res = mutations[LOGICAL_PATH]

    if LOGICAL_PATH == MAINLINE:
        combined = defaultdict(list)
        for mut_id, val in mutations.items():
            if mut_id in [MAINLINE, LOGICAL_PATH]:
                continue
            combined[val].append(mut_id)

        for val, mut_ids in combined.items():
            if val != log_res:
                main_mut_id = mut_ids[0]
                if ACTIVE_MUTANTS is not None:
                    ACTIVE_MUTANTS -= set(mut_ids)
                MASKED_MUTANTS |= set(mut_ids)
                if FORKING_CONTEXT.maybe_fork(main_mut_id):
                    MASKED_MUTANTS = set()
                    ACTIVE_MUTANTS = set(mut_ids)
                    if 2 in mut_ids:
                        print("hi")
                    return val
    try:
        return mutations[LOGICAL_PATH]
    except KeyError:
        return mutations[MAINLINE]


def t_combine(mutations):
    if EXECUTION_MODE is ExecutionMode.SPLIT_STREAM:
        return combine_split_stream(mutations)
    elif EXECUTION_MODE is ExecutionMode.MODULO_EQV:
        return combine_modulo_eqv(mutations)
    else:
        return ShadowVariable(mutations)


###############################################################################
# tainted types


# class ShadowProxy():
#     __slots__ = ['_shadow', '_thing']

#     def __init__(self, shadow, thing):
#         print(shadow, thing)
#         self._shadow = shadow
#         self._thing = thing

LIST_OF_ALLOWED_DUNDER_METHODS = [
    '__abs__', '__add__', '__and__', '__div__', '__divmod__', '__eq__', 
    '__ne__', '__le__', '__len__', 
    '__ge__', '__gt__', '__sub__', '__lt__',
]

LIST_OF_DISALLOWED_DUNDER_METHODS = [
    '__aenter__', '__aexit__', '__aiter__', '__anext__', '__await__',
    '__bool__', '__bytes__', '__call__', '__class__', '__cmp__', '__complex__', '__contains__',
    '__delattr__', '__delete__', '__delitem__', '__delslice__', '__dir__', 
    '__enter__', '__exit__', '__float__', '__floordiv__', '__fspath__',
    '__get__', '__getitem__', '__getnewargs__', '__getslice__', 
    '__hash__', '__import__', '__imul__', '__index__',
    '__int__', '__invert__',
    '__ior__', '__iter__', '__ixor__', '__lshift__', 
    '__mod__', '__mul__', '__neg__', '__next__', '__nonzero__',
    '__or__', '__pos__', '__pow__', '__prepare__', '__radd__', '__rand__', '__rdiv__',
    '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rfloordiv__',
    '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__',
    '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__set__', '__setitem__',
    '__setslice__', '__sizeof__', '__subclasscheck__', '__subclasses__',
    '__truediv__', '__xor__', 
]

LIST_OF_IGNORED_DUNDER_METHODS = [
    '__new__', '__init__', '__init_subclass__', '__instancecheck__', '__getattribute__', 
    '__setattr__', '__str__', '__format__', 
    '__iadd__', '__iand__', '__isub__', 
]

class ShadowVariable():
    __slots__ = ['_shadow']

    for method in LIST_OF_ALLOWED_DUNDER_METHODS:
        exec(f"""
    def {method}(self, other, *args, **kwargs):
        assert len(args) == 0 and len(kwargs) == 0
        return self._do_op(other, "{method}")
        """.strip())

    for method in LIST_OF_DISALLOWED_DUNDER_METHODS:
        exec(f"""
    def {method}(self, *args, **kwargs):
        raise ValueError("dunder method {method} is not allowed")
        """.strip())

    def __init__(self, values):
        combined = {}
        for mut_id, val in values.items():
            if type(val) == ShadowVariable:
                val = val._shadow
                if mut_id in val:
                    combined[mut_id] = val[mut_id]
                else:
                    combined[mut_id] = val[MAINLINE]
            else:
                combined[mut_id] = val

        self._shadow = combined

    # def __getattribute__(self, name: str):
    #     if name in ["_shadow", "_do_op"]:
    #         return super().__getattribute__(name)
    #     raise NotImplementedError()

    def __repr__(self):
        return f"{self._shadow}"

    def _do_op(self, other, op):
        logger.debug("op: %s %s %s", self, other, op)
        self_shadow = self._shadow
        if type(other) == ShadowVariable:
            other_shadow = other._shadow
            # notice that both self and other has taints.
            # the result we need contains taints from both.
            other_main = other_shadow[MAINLINE]
            self_main = self_shadow[MAINLINE]
            common_shadows = {k for k in self_shadow if k in other_shadow}
            vs_ = { k: self_shadow[k].__getattribute__(op)(other_main) for k in self_shadow if k not in other_shadow }
            vo_ = { k: other_shadow[k].__getattribute__(op)(self_main) for k in other_shadow if k not in self_shadow }

            # if there was a preexisint taint of the same name, this mutation was
            # already executed. So, use that value.
            cs_ = { k: self_shadow[k].__getattribute__(op)(other_shadow[k]) for k in common_shadows}
            #assert vs[MAINLINE] == os[MAINLINE]
            res = {**vs_, **vo_, **cs_}
            logger.debug("res: %s", res)
            return ShadowVariable(res)
        else:
            res = { k: self_shadow[k].__getattribute__(op)(other) for k in self_shadow }
            return ShadowVariable(res)


def losing_taint(self):
    raise NotImplementedError(
        "Casting to a plain bool loses all taint information. "
        "Raise exception here to avoid unexpectedly losing information."
    )


def proxy_function(cls, name, f):
    @wraps(f)
    def proxied_f(*args, **kwargs):
        res = f(*args, **kwargs)
        # logger.debug('%s %s: %s %s -> %s (%s)', cls, name, args, kwargs, res, type(res))
        return res
    return proxied_f


def taint(orig_class):
    cls_proxy = partial(proxy_function, orig_class)
    for func in dir(orig_class):
        if func in ['__bool__']:
            setattr(orig_class, func, losing_taint)
            continue
        if func in [
            '_shadow',
            '__new__', '__init__', '__class__', '__dict__', '__getattribute__', '__repr__'
        ]:
             continue
        orig_func = getattr(orig_class, func)
        # logging.debug("%s %s", orig_class, func)
        setattr(orig_class, func, cls_proxy(func, orig_func))


    return orig_class


@taint
class t_tuple():
    def __init__(self, *args, **kwargs):
        self.val = tuple(*args, **kwargs)
        self.len = t_combine({MAINLINE: len(self.val)})

    def __iter__(self):
        for elem in self.val:
            yield elem

    def __eq__(self, other):
        res = self.len == other.__len__()
        if not t_cond(res):
            return res

        for a, b in zip(self, other):
            raise NotImplementedError()

        return res

    def __len__(self):
        return self.len

    def __str__(self):
        return f"t_tuple {getattr(self, 'len', None)} {getattr(self, 'val', None)}"

    def __repr__(self):
        return f"t_tuple {getattr(self, 'len', None)} {getattr(self, 'val', None)}"


@taint
class t_list():
    pass