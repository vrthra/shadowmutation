# mypy: ignore-errors
# This file requires too much metaprogramming to provide correct typing for mypy.

from json.decoder import JSONDecodeError
import os
import sys
import json
import time
import atexit
from collections import Counter
from typing import Any
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from copy import deepcopy
from functools import wraps, partial
from itertools import chain
from contextlib import contextmanager

import logging
from typing import Union
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(process)d %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MAINLINE = 0
LOGICAL_PATH = 0

LAST_TRACED_LINE = None
SUBJECT_COUNTER = 0
TOOL_COUNTER = 0
SUBJECT_COUNTER_DICT = None

OLD_TRACE = sys.gettrace()


IGNORE_FILES = set([
    "__init__.py",
    "decoder.py",
    "encoder.py",
    "threading.py",
    "genericpath.py",
    "posixpath.py",
    "types.py",
    "enum.py",
    "copy.py",
    "abc.py",
    "os.py",
    "re.py",
    "warnings.py",
    "sre_compile.py",
    "sre_parse.py",
    "functools.py",
    "tempfile.py",
    "random.py",
    "pathlib.py",
    "codecs.py",
    "fnmatch.py",
    "typing.py",
    "_collections_abc.py",
    "_weakrefset.py",
    "_bootlocale.py",
    "<frozen importlib._bootstrap>",
    "<string>",
])


TOOL_FILES = set([
    "shadow.py",
])


def reset_lines():
    global SUBJECT_COUNTER
    global SUBJECT_COUNTER_DICT
    global TOOL_COUNTER
    SUBJECT_COUNTER = 0
    SUBJECT_COUNTER_DICT = defaultdict(int)
    TOOL_COUNTER = 0


def trace_func(frame, event, arg):
    global LAST_TRACED_LINE
    global SUBJECT_COUNTER
    global SUBJECT_COUNTER_DICT
    global TOOL_COUNTER

    fname = frame.f_code.co_filename
    fname_sub = Path(fname)
    fname_sub_name = fname_sub.name

    if fname_sub_name in IGNORE_FILES:
        # logger.debug(f"ignored: {fname_sub.name} {frame.f_code.co_name} {frame.f_code.co_firstlineno}")
        return trace_func

    # frame.f_trace_opcodes = True

    # logger.debug(f"{dir(frame)}")
    # logger.debug(f"{frame}")
    if event != 'line':
        return trace_func

    if frame.f_code.co_name in [
        "tool_line_counting", "subject_line_counting", "t_gather_results", "disable_line_counting"
    ]:
        return trace_func

    is_subject_file = fname_sub.parent.parent.parent.name == "shadowmutation" and \
        fname_sub.parent.parent.name == "tmp"
    is_tool_file = fname_sub_name in TOOL_FILES
    assert not (is_subject_file and is_tool_file)
    if not (is_subject_file or is_tool_file):
        assert False, f"Unknown file: {fname}"


    if is_tool_file:
        # logger.debug(f"tool: {fname_sub.name} {frame.f_code.co_name} {frame.f_code.co_firstlineno}")
        TOOL_COUNTER += 1
    else:
        cur_line = (fname_sub.name, frame.f_lineno)
        # logger.debug(f"{cur_line} {LAST_TRACED_LINE}")
        if cur_line == LAST_TRACED_LINE:
            return trace_func
        LAST_TRACED_LINE = cur_line

        # logger.debug(f"subject: {fname_sub.name} {frame.f_code.co_name} {frame.f_lineno}")
        SUBJECT_COUNTER_DICT[cur_line] += 1
        SUBJECT_COUNTER += 1

    return trace_func

def disable_line_counting():
    sys.settrace(OLD_TRACE)


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
    SHADOW_FORK = 4 # shadow and forking

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

    def __del__(self):
        if self.is_parent:
            (self.sync_dir/'paths').rmdir()
            (self.sync_dir/'results').rmdir()
            self.sync_dir.rmdir()

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
            # Now wait for the forked child, this minimizes the forked child processes and make
            # this effectively single threaded.
            try:
                os.waitpid(forked_pid, 0)
            except ChildProcessError as e:
                pass
            logger.debug(f"Counter parent: {SUBJECT_COUNTER} {TOOL_COUNTER}")
            return False
        else:
            # Update that this is the child.
            self.is_parent = False

            # Update which path child is supposed to follow
            LOGICAL_PATH = path
            reset_lines()
            logger.debug(f"Counter child: {SUBJECT_COUNTER} {TOOL_COUNTER}")
            return True

    def wait_for_forks(self, fork_res=None):
        global ACTIVE_MUTANTS
        global SUBJECT_COUNTER
        global TOOL_COUNTER
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
                results['subject_count'] = SUBJECT_COUNTER
                results['subject_count_lines'] = {'::'.join(str(a) for a in k): v for k, v in SUBJECT_COUNTER_DICT.items()}
                results['tool_count'] = TOOL_COUNTER
                logger.debug(f"{fork_res}")
                if type(fork_res) == ShadowVariable:
                    results['fork_res'] = fork_res._shadow
                else:
                    assert type(fork_res) != dict
                    results['fork_res'] = fork_res
                json.dump(results, f)

            # exit the child immediately, this might cause problems for programs actually using multiprocessing
            # but this is a prototype
            os._exit(0)

        # wait for all child processes to end
        combined_fork_res = [fork_res]
        all_results = t_get_killed()
        while True:
            is_done = True
            time.sleep(.1)
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
                    if e.errno != 10:
                        logger.debug(f"{e}")

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
                    SUBJECT_COUNTER += child_results['subject_count']
                    TOOL_COUNTER += child_results['tool_count']
                    for k, v in child_results['subject_count_lines'].items():
                        key = k.split("::")
                        key[1] = int(key[1])
                        SUBJECT_COUNTER_DICT[tuple(key)] += v

                    child_fork_res = child_results['fork_res']
                    if type(child_fork_res) == dict:
                        combined_fork_res.append(child_results)
                    else:
                        combined_fork_res.append(child_results)

                    path_file.unlink()
                    result_file.unlink()
            
            if is_done:
                break
        logger.debug(f"Done waiting for forks.")
        return combined_fork_res


STRONGLY_KILLED = None
WEAKLY_KILLED = None
ACTIVE_MUTANTS = None
MASKED_MUTANTS = None
EXECUTION_MODE = None
RESULT_FILE = None
FORKING_CONTEXT: Union[None, Forker] = None


def reinit(logical_path: str=None, execution_mode: Union[None, str]=None, no_atexit=False):
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

    if os.environ.get('GATHER_ATEXIT', '0') == '1':
        atexit.register(t_gather_results)
    else:
        atexit.unregister(t_gather_results)

    if os.environ.get("TRACE", "0") == "1":
        sys.settrace(trace_func)
        reset_lines()


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


def t_counter_results():
    res = {}
    res['subject_count'] = SUBJECT_COUNTER
    res['subject_count_lines'] = sorted(SUBJECT_COUNTER_DICT.items(), key=lambda x: x[0])
    res['tool_count'] = TOOL_COUNTER
    return res


def t_gather_results() -> Any:
    disable_line_counting()
    t_wait_for_forks()

    results = t_get_killed()
    results['execution_mode'] = EXECUTION_MODE.name
    results = {**results, **t_counter_results()}
    if RESULT_FILE is not None:
        with open(RESULT_FILE, 'wt') as f:
            json.dump(results, f, cls=SetEncoder)
    logging.info(f"{results}")
    return results


def t_get_logical_path():
    return LOGICAL_PATH


def fork_wrap(f, *args, **kwargs):
    global FORKING_CONTEXT
    global ACTIVE_MUTANTS
    global MASKED_MUTANTS
    old_forking_context = FORKING_CONTEXT
    old_active_mutants = deepcopy(ACTIVE_MUTANTS)
    old_masked_mutants = deepcopy(MASKED_MUTANTS)

    FORKING_CONTEXT = Forker()
    try:
        res = f(*args, **kwargs)
    except Exception:
        raise NotImplementedError("Exceptions in wrapped functions are not supported.")
    combined_results = FORKING_CONTEXT.wait_for_forks(fork_res=res)

    FORKING_CONTEXT = old_forking_context
    ACTIVE_MUTANTS = old_active_mutants
    MASKED_MUTANTS = old_masked_mutants

    as_shadow = {}
    # mainline value is always in first
    as_shadow = {MAINLINE: combined_results[0]}
    for child_res in combined_results[1:]:
        child_fork_res = child_res['fork_res']
        if type(child_fork_res) == dict:
            for active in child_res['active']:
                as_shadow[active] = ShadowVariable({int(k): v for k, v in child_fork_res.items()})
        else:
            for active in child_res['active']:
                as_shadow[active] = child_fork_res

    res = ShadowVariable(as_shadow)

    # If only mainline in return value untaint it
    if len(res._shadow) == 1:
        assert MAINLINE in res._shadow, f"{res}"
        res = res._shadow[MAINLINE]
        return res

    return res


def no_fork_wrap(f, *args, **kwargs):
    global LOGICAL_PATH
    global ACTIVE_MUTANTS
    global MASKED_MUTANTS
    logger.debug(f"CALL {f.__name__}({args} {kwargs})")
    before_active = deepcopy(ACTIVE_MUTANTS)
    before_masked = deepcopy(MASKED_MUTANTS)

    remaining_paths = set([0])
    done_paths = set()

    for arg in chain(args, kwargs.values()):
        if type(arg) == ShadowVariable:
            remaining_paths |= arg._get_paths()
    remaining_paths -= before_masked

    tainted_return = {}
    while remaining_paths:
        LOGICAL_PATH = remaining_paths.pop()
        logger.debug(f"cur path: {LOGICAL_PATH} remaining: {remaining_paths}")
        ACTIVE_MUTANTS = deepcopy(before_active)
        MASKED_MUTANTS = deepcopy(before_masked)
        try:
            res = f(*args, **kwargs)
        except Exception:
            raise NotImplementedError("Exceptions in wrapped functions are not supported.")
        after_active = deepcopy(ACTIVE_MUTANTS)
        after_masked = deepcopy(MASKED_MUTANTS)
        logger.debug('wrapped: %s(%s %s) -> %s (%s)', f.__name__, args, kwargs, res, type(res))
        new_active = (after_active or set()) - (before_active or set())
        new_masked = after_masked - before_masked
        
        logger.debug("masked: %s %s %s", before_masked, after_masked, new_masked)
        logger.debug("active: %s %s %s", before_active, after_active, new_active)
        if type(res) != ShadowVariable:
            assert LOGICAL_PATH not in tainted_return, f"{LOGICAL_PATH} {tainted_return}"
            tainted_return[LOGICAL_PATH] = res

            if new_active:
                assert LOGICAL_PATH in new_active, new_active

                for path in after_active or set():
                    # already recorded the LOGICAL_PATH result before the loop
                    if path == LOGICAL_PATH:
                        continue
                    assert path not in tainted_return, f"{path} {tainted_return}"
                    tainted_return[path] = res
        elif type(res) == ShadowVariable:
            shadow = get_active(res._shadow)
            unused_active = new_active

            if LOGICAL_PATH in shadow:
                unused_active.discard(LOGICAL_PATH)
                log_res = shadow[LOGICAL_PATH]
            else:
                log_res = shadow[MAINLINE]

            if LOGICAL_PATH in tainted_return:
                assert tainted_return[LOGICAL_PATH] == log_res, f"{tainted_return[LOGICAL_PATH]} == {log_res}"
            else:
                tainted_return[LOGICAL_PATH] = log_res

            for path, val in shadow.items():
                if path == MAINLINE or path == LOGICAL_PATH or path in new_masked:
                    continue
                if path in tainted_return:
                    assert tainted_return[path] == val, f"{tainted_return[path]} == {val}"
                else:
                    tainted_return[path] = val
                unused_active.discard(path)
                done_paths.add(path)
            
            for path in unused_active:
                tainted_return[path] = shadow[MAINLINE]
                done_paths.add(path)

        # logger.debug(f"cur return {tainted_return}")

        done_paths |= new_active
        done_paths.add(LOGICAL_PATH)
        remaining_paths |= new_masked
        remaining_paths -= done_paths

    LOGICAL_PATH = MAINLINE
    ACTIVE_MUTANTS = before_active
    MASKED_MUTANTS = before_masked

    logger.debug("return: %s", tainted_return)

    # If only mainline in return value untaint it
    if len(tainted_return) == 1:
        assert MAINLINE in tainted_return, f"{tainted_return}"
        res = tainted_return[MAINLINE]
        return res

    res = t_combine(tainted_return)
    return res


def t_wrap(f):
    @wraps(f)
    def flow_wrapper(*args, **kwargs):
        if FORKING_CONTEXT:
            return fork_wrap(f, *args, **kwargs)
        else:
            return no_fork_wrap(f, *args, **kwargs)

    return flow_wrapper


def t_cond(cond: Any) -> bool:
    global WEAKLY_KILLED
    global FORKING_CONTEXT
    global ACTIVE_MUTANTS
    global MASKED_MUTANTS

    shadow = get_active_shadow(cond)

    if shadow is not None:
        # logger.debug(f"shadow {shadow} {LOGICAL_PATH}")

        logical_path_is_diverging = False

        diverging_mutants = []
        companion_mutants = []
        

        mainline_res = shadow[MAINLINE]
        if LOGICAL_PATH in shadow:
            res = shadow[LOGICAL_PATH]
            assert type(res) == bool, f"{type(res)}"
        else:
            res = mainline_res

        assert type(mainline_res) == bool, f"{type(mainline_res)}"

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
    filtered_mutations = {
        path: val for path, val in mutations.items()
        if path not in MASKED_MUTANTS
    }

    if ACTIVE_MUTANTS is not None:
        filtered_mutations = { path: val for path, val in mutations.items() if path in ACTIVE_MUTANTS }

    # logger.debug(f"log_path: {LOGICAL_PATH}")
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
    # logger.debug(f"t_assert {cmp_result} {shadow}")
    if shadow is not None:
        for path, val in shadow.items():
            # The mainline assertion is done after the for loop
            if path == MAINLINE:
                continue
            if not val:
                STRONGLY_KILLED.add(path)
                logger.info(f"t_assert strongly killed: {path}")

        if ACTIVE_MUTANTS is not None and not shadow[MAINLINE]:
            assert LOGICAL_PATH in ACTIVE_MUTANTS, f"{ACTIVE_MUTANTS}"
            for mut in ACTIVE_MUTANTS:
                if mut not in shadow:
                    STRONGLY_KILLED.add(mut)
                    logger.info(f"t_assert strongly killed: {path}")

        # Do the actual assertion as would be done in the unchanged program but only for mainline execution
        if LOGICAL_PATH == MAINLINE:
            assert shadow[MAINLINE], f"{shadow}"
    else:
        if not cmp_result:
            if ACTIVE_MUTANTS is not None:
                for mut in ACTIVE_MUTANTS:
                    STRONGLY_KILLED.add(mut)
                    logger.info(f"t_assert strongly killed: {mut}")
            else:
                assert cmp_result, f"Failed original assert"


def split_assert(cmp_result):
    # logger.debug(f"t_assert {cmp_result}")
    assert type(cmp_result) == bool, f"{type(cmp_result)}"
    if cmp_result:
        return
    else:
        for mut in ACTIVE_MUTANTS:
            STRONGLY_KILLED.add(mut)
            # logger.info(f"t_assert strongly killed: {mut}")


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
    global MASKED_MUTANTS
    if LOGICAL_PATH == MAINLINE:
        all_muts = set(mutations.keys())
        for mut_id, val in mutations.items():
            if mut_id in [MAINLINE, LOGICAL_PATH]:
                continue
            if FORKING_CONTEXT.maybe_fork(mut_id):
                ACTIVE_MUTANTS = set([mut_id])
                MASKED_MUTANTS = all_muts - ACTIVE_MUTANTS
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


def t_combine(mutations: dict[int, Any]) -> Any:
    if EXECUTION_MODE is ExecutionMode.SPLIT_STREAM:
        res = combine_split_stream(mutations)
    elif EXECUTION_MODE is ExecutionMode.MODULO_EQV:
        res = combine_modulo_eqv(mutations)
    else:
        res = ShadowVariable(mutations)
    return res


###############################################################################
# tainted types

ALLOWED_DUNDER_METHODS = {
    'unary_ops': ['__abs__', '__round__', '__neg__', ],
    'bool_ops': [
        '__add__', '__and__', '__div__', '__truediv__', '__rtruediv__', '__divmod__', '__eq__', 
        '__ne__', '__le__', '__len__', '__pow__', '__mod__', 
        '__ge__', '__gt__', '__sub__', '__lt__', '__mul__', 
        '__radd__', '__rmul__', '__rmod__', 
    ],
}

DISALLOWED_DUNDER_METHODS = [
    '__aenter__', '__aexit__', '__aiter__', '__anext__', '__await__',
    '__bytes__', '__call__', '__class__', '__cmp__', '__complex__', '__contains__',
    '__delattr__', '__delete__', '__delitem__', '__delslice__', '__dir__', 
    '__enter__', '__exit__', '__floordiv__', '__fspath__',
    '__get__', '__getitem__', '__getnewargs__', '__getslice__', 
    '__hash__', '__import__', '__imul__', '__index__',
    '__int__', '__invert__',
    '__ior__', '__iter__', '__ixor__', '__lshift__', 
    '__next__', '__nonzero__',
    '__or__', '__pos__', '__prepare__', '__rand__', '__rdiv__',
    '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rfloordiv__',
    '__rlshift__', '__ror__', '__rpow__', '__rrshift__',
    '__rshift__', '__rsub__', '__rxor__', '__set__', '__setitem__',
    '__setslice__', '__sizeof__', '__subclasscheck__', '__subclasses__',
    '__xor__', 

    # python enforces that the specific type is returned,
    # to my knowledge we cannot override these dunders to return
    # a ShadowVariable
    # disallow these dunders to avoid accidentally losing taint info
    '__bool__', '__float__',
]

LIST_OF_IGNORED_DUNDER_METHODS = [
    '__new__', '__init__', '__init_subclass__', '__instancecheck__', '__getattribute__', 
    '__setattr__', '__str__', '__format__', 
    '__iadd__', '__iand__', '__isub__', 
]


class ShadowVariable():
    _shadow: dict[int, Any]
    __slots__ = ['_shadow']

    for method in ALLOWED_DUNDER_METHODS['unary_ops']:
        exec(f"""
    def {method}(self, *args, **kwargs):
        # assert len(args) == 0 and len(kwargs) == 0, f"{{len(args)}} == 0 and {{len(kwargs)}} == 0"
        return self._do_unary_op("{method}", *args, **kwargs)
        """.strip())

    # self.{method} = lambda other, *args, **kwargs: self._do_bool_op(other, *args, **kwargs)
    for method in ALLOWED_DUNDER_METHODS['bool_ops']:
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

    def __init__(self, values):
        combined = {}
        mainline_val = values[MAINLINE]
        if type(mainline_val) == ShadowVariable:
            combined = mainline_val._shadow
        else:
            combined[MAINLINE] = mainline_val
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

        self._shadow = get_active(combined)

    # def __getattribute__(self, name: str):
    #     if name in ["_shadow", "_do_op"]:
    #         return super().__getattribute__(name)
    #     raise NotImplementedError()

    def __repr__(self):
        return f"ShadowVariable({self._shadow})"

    def _get_paths(self):
        return self._shadow.keys()

    def _check_op_available(shadow, op):
        global STRONGLY_KILLED

        killed = []
        for k in shadow:
            if k in [LOGICAL_PATH, MAINLINE]:
                continue
            try:
                shadow.__getattribute__(op)
            except AttributeError:
                STRONGLY_KILLED.add(k)
                killed.append(k)
        for k in killed:
            del shadow[k]

    def _do_op_safely(self, paths, left, right, op, context):
        global STRONGLY_KILLED
        try_other_side = False
        
        res = {}
        for k in paths:
            try:
                k_res = context(left(k), right(k), op)
            except AttributeError:
                try_other_side = True
                k_res = None
            except ZeroDivisionError:
                STRONGLY_KILLED.add(k)
                continue

            if k_res == NotImplemented:
                try_other_side = True

            if try_other_side:
                # try other side application as well
                # left.__sub__(right) -> right.__rsub__(left)
                # or
                # left.__rsub__(right) -> right.__sub__(left)
                if op[:3] == "__r":
                    # remove r
                    other_op = op[:2] + op[3:]
                else:
                    # add r
                    other_op = op[:2] + "r" + op[2:]
                try:
                    k_res = context(right(k), left(k), other_op)
                except AttributeError:
                    STRONGLY_KILLED.add(k)
                    continue
                if k_res == NotImplemented:
                    STRONGLY_KILLED.add(k)
                    continue

            res[k] = k_res

        return res

    def _do_unary_op(self, op, *args, **kwargs):
        # logger.debug("op: %s %s", self, op)
        self_shadow = self._shadow
        # res = self._do_op_safely(self_shadow.keys(), lambda k: self_shadow[k].__getattribute__(op)(*args, **kwargs))
        res = self._do_op_safely(
            self_shadow.keys(),
            lambda k: self_shadow[k],
            lambda k: None,
            op,
            lambda left, right, op: left.__getattribute__(op)(*args, **kwargs))
        # logger.debug("res: %s %s", res, type(res[0]))
        return ShadowVariable(res)

    def _do_bool_op(self, other, op, *args, **kwargs):

        # logger.debug("op: %s %s %s", self, other, op)
        self_shadow = self._shadow
        if type(other) == ShadowVariable:
            other_shadow = other._shadow
            # notice that both self and other has taints.
            # the result we need contains taints from both.
            other_main = other_shadow[MAINLINE]
            self_main = self_shadow[MAINLINE]
            common_shadows = {k for k in self_shadow if k in other_shadow}
            vs_ = self._do_op_safely(
                (k for k in self_shadow if k not in other_shadow),
                lambda k: self_shadow[k],
                lambda k: other_main,
                op,
                lambda left, right, op: left.__getattribute__(op)(right, *args, **kwargs)
            )
            vo_ = self._do_op_safely(
                (k for k in other_shadow if k not in self_shadow),
                lambda k: other_shadow[k],
                lambda k: self_main,
                op,
                lambda left, right, op: left.__getattribute__(op)(right, *args, **kwargs)
            )

            # if there was a preexisint taint of the same name, this mutation was
            # already executed. So, use that value.
            cs_ = self._do_op_safely(
                common_shadows,
                lambda k: self_shadow[k],
                lambda k: other_shadow[k],
                op,
                lambda left, right, op: left.__getattribute__(op)(right, *args, **kwargs)
            )
            res = {**vs_, **vo_, **cs_}
            # logger.debug("res: %s", res)
            return ShadowVariable(res)
        else:
            res = self._do_op_safely(
                self_shadow.keys(),
                lambda k: self_shadow[k],
                lambda k: other,
                op,
                lambda left, right, op: left.__getattribute__(op)(right, *args, **kwargs)
            )
            # logger.debug("res: %s %s", res, type(res[0]))
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




# Init when importing shadow
reinit()