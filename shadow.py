# mypy: ignore-errors

from json.decoder import JSONDecodeError
import pickle
import os
import sys
import json
import tempfile
import time
import atexit
import traceback
import types
from collections.abc import Iterator
from typing import Any, Callable, TypeVar, Tuple
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
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(process)d %(filename)s:%(lineno)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


SV = TypeVar('SV', bound="ShadowVariable")


# This is used to decide what return values should be untainted when returning from a function.
PRIMITIVE_TYPES = [bool, int, float]

MAINLINE = 0
LOGICAL_PATH = 0

LAST_TRACED_LINE = None
SUBJECT_COUNTER = 0
TOOL_COUNTER = 0
SUBJECT_COUNTER_DICT = {}

CACHE_PATH = None

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
    "copyreg.py",
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
    global CACHE_PATH

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
        assert False, f"Unknown file: {fname}, add it to the top of shadow.py"


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
        SUBJECT_COUNTER += 1
        SUBJECT_COUNTER_DICT[cur_line] += 1

    return trace_func

def disable_line_counting():
    sys.settrace(OLD_TRACE)


class ShadowException(Exception):
    pass


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
    SHADOW_CACHE = 5 # shadow types and no forking with caching
    SHADOW_FORK_CACHE = 6 # shadow and forking with caching

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
        elif mode == 'shadow_cache':
            return ExecutionMode.SHADOW_CACHE
        elif mode == 'shadow_fork_cache':
            return ExecutionMode.SHADOW_FORK_CACHE
        else:
            raise ValueError("Unknown Execution Mode", mode)

    def should_start_forker(self):
        if self in [
            ExecutionMode.SPLIT_STREAM,
            ExecutionMode.MODULO_EQV,
            ExecutionMode.SHADOW_FORK,
            ExecutionMode.SHADOW_FORK_CACHE,
        ]:
            return True
        else:
            return False

    def is_shadow_variant(self):
        if self in [ExecutionMode.SHADOW, ExecutionMode.SHADOW_CACHE, ExecutionMode.SHADOW_FORK, ExecutionMode.SHADOW_FORK_CACHE]:
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
        (self.sync_dir/'forks').mkdir()
        (self.sync_dir/'results').mkdir()

    def __del__(self):
        if self.is_parent:
            (self.sync_dir/'paths').rmdir()
            (self.sync_dir/'forks').rmdir()
            (self.sync_dir/'results').rmdir()
            self.sync_dir.rmdir()

    def my_pid(self):
        return os.getpid()

    def maybe_fork(self, path):
        global LOGICAL_PATH

        # Only fork once, from then on follow that path.
        path_file = self.sync_dir.joinpath('paths', str(path))
        if path_file.is_file():
            return False
        path_file.touch()

        # Try to fork
        forked_pid = os.fork()
        # logger.debug(f"Forking for path: {path} got pid: {forked_pid}")
        if forked_pid == -1:
            # Error during forking. Not much we can do.
            raise ValueError(f"Could not fork for path: {path}!")
        elif forked_pid != 0:
            # We are in parent, record the child pid and path.
            try:
                os.waitpid(forked_pid, 0)
            except ChildProcessError as e:
                if e.errno != 10:
                    logger.debug(f"{e}")
            path_file.write_text(str(forked_pid))
            return False
        else:
            # Update that this is the child.
            self.is_parent = False

            # Update which path child is supposed to follow
            LOGICAL_PATH = path
            forked_pid = self.my_pid()
            # fork_file = self.sync_dir.joinpath('forks', str(forked_pid)) # this is used to indicate when child can start
            # while not fork_file.is_file():
            #     # Wait until parent finishes.
            #     time.sleep(.1)
            # fork_file.unlink()

            # logger.debug(f"Child starting for path: {path}, with pid: {forked_pid}")
            reset_lines()
            return True

    def child_end(self, fork_res=None):
        assert not self.is_parent
        pid = self.my_pid()
        path = LOGICAL_PATH
        # logger.debug(f"Child with pid: {pid} and path: {path} has reached sync point.")
        res_path = self.sync_dir/'results'/str(pid)
        # logger.debug(f"Writing results to: {res_path}")
        with open(res_path, 'wb') as f:
            results = t_get_killed()
            for res in ['strong', 'masked'] + ['seen'] if EXECUTION_MODE.is_shadow_variant() else ['active']:
                results[res] = list(results[res])
            results['pid'] = pid
            results['path'] = path
            results['subject_count'] = SUBJECT_COUNTER
            results['subject_count_lines'] = {'::'.join(str(a) for a in k): v for k, v in SUBJECT_COUNTER_DICT.items()}
            results['tool_count'] = TOOL_COUNTER
            if fork_res is None:
                results['fork_res'] = None
            else:
                return_val, args, kwargs = fork_res
                assert type(return_val) != dict

                results_args = {}
                for ii, arg in enumerate(args):
                    if type(arg) == ShadowVariable:
                        results_args[ii] = arg

                results_kwargs = {}
                for kk, val in kwargs.items():
                    if type(val) == ShadowVariable:
                        results_kwargs[kk] = val

                results['fork_res'] = (return_val, results_args, results_kwargs)
            # logger.debug(f"child results to write: {results}")
            pickle.dump(results, f)

        # exit the child immediately, this might cause problems for programs actually using multiprocessing
        # but this is a prototype
        os._exit(0)

    def wait_for_forks(self, fork_res: dict[int: Any]=None):
        global PICKLE_LOAD
        global SUBJECT_COUNTER
        global TOOL_COUNTER
        # if child, write results and exit
        if not self.is_parent:
            self.child_end(fork_res)

        # wait for all child processes to end
        if fork_res is not None:
            return_val, _, _ = fork_res
        else:
            return_val = None

        combined_fork_res = [get_active_shadow(return_val, SEEN_MUTANTS, MASKED_MUTANTS)]
        all_results = t_get_killed()
        while True:
            is_done = True
            for path_file in (self.sync_dir/'paths').glob("*"):
                is_done = False
                try:
                    child_pid = int(path_file.read_text())
                except ValueError:
                    continue

                # sync_pid_go_file = (self.sync_dir/'forks').joinpath(str(child_pid))
                # logger.debug(f"Waiting for pid: {child_pid} {sync_pid_go_file}")

                # Signal that child can start.
                # sync_pid_go_file.touch()
                while True:
                    time.sleep(.01)

                    try:
                        os.waitpid(child_pid, 0)
                    except ChildProcessError as e:
                        if e.errno != 10:
                            logger.debug(f"{e}")

                    result_file = self.sync_dir/'results'/str(child_pid)
                    if result_file.is_file():
                        with open(result_file, 'rb') as f:
                            try:
                                PICKLE_LOAD = True
                                child_results = pickle.load(f)
                            except JSONDecodeError:
                                # Child has not yet written the results.
                                continue
                            finally:
                                PICKLE_LOAD = False

                        for res in ['strong', 'masked'] + ['seen'] if EXECUTION_MODE.is_shadow_variant() else ['active']:
                            child_results[res] = set(child_results[res])

                        if EXECUTION_MODE.is_split_stream_variant():
                            for res in ['strong']:
                                add_res = child_results[res] & child_results['active']
                                all_results[res] |= add_res
                        else:
                            assert not (child_results['strong'] - child_results['seen']), f"{child_results}"

                        all_results['strong'] |= child_results['strong']

                        # logger.debug(f"child results: {child_results}")
                        SUBJECT_COUNTER += child_results['subject_count']
                        TOOL_COUNTER += child_results['tool_count']
                        for k, v in child_results['subject_count_lines'].items():
                            key = k.split("::")
                            key[1] = int(key[1])
                            SUBJECT_COUNTER_DICT[tuple(key)] += v

                        combined_fork_res.append(child_results)

                        path_file.unlink()
                        result_file.unlink()
                        break
            
            if is_done:
                break
        return combined_fork_res


STRONGLY_KILLED = None
NS_ACTIVE_MUTANTS = None # NS = Non Shadow
SEEN_MUTANTS = None
MASKED_MUTANTS = None
SELECTED_MUTANT = None
EXECUTION_MODE = None
RESULT_FILE = None
FORKING_CONTEXT: Union[None, Forker] = None


def reinit(logical_path: str=None, execution_mode: Union[None, str]=None, no_atexit=False):
    # logger.info("Reinit global shadow state")
    # initializing shadow
    global LOGICAL_PATH
    global STRONGLY_KILLED
    global NS_ACTIVE_MUTANTS
    global SEEN_MUTANTS
    global MASKED_MUTANTS
    global SELECTED_MUTANT
    global EXECUTION_MODE
    global FORKING_CONTEXT
    global RESULT_FILE
    global CACHE_PATH
    global PICKLE_LOAD

    PICKLE_LOAD = False
    RESULT_FILE = os.environ.get('RESULT_FILE')

    if logical_path is not None:
        LOGICAL_PATH = logical_path
    else:
        LOGICAL_PATH = int(os.environ.get('LOGICAL_PATH', MAINLINE))

    if execution_mode is not None:
        EXECUTION_MODE = ExecutionMode.get_mode(execution_mode)
    else:
        EXECUTION_MODE = ExecutionMode.get_mode(os.environ.get('EXECUTION_MODE'))

    STRONGLY_KILLED = set()
    SELECTED_MUTANT = None

    if EXECUTION_MODE in [ExecutionMode.SHADOW, ExecutionMode.SHADOW_CACHE, ExecutionMode.SHADOW_FORK, ExecutionMode.SHADOW_FORK_CACHE]:
        SEEN_MUTANTS = set()
        MASKED_MUTANTS = set()
    elif EXECUTION_MODE in [ExecutionMode.SPLIT_STREAM, ExecutionMode.MODULO_EQV]:
        NS_ACTIVE_MUTANTS = None
        MASKED_MUTANTS = set()
    elif EXECUTION_MODE == ExecutionMode.NOT_SPECIFIED:
        pass
    else:
        raise ValueError(f"Unknown execution mode: {EXECUTION_MODE}")

    if EXECUTION_MODE.should_start_forker():
        # logger.debug("Initializing forker")
        FORKING_CONTEXT = Forker()
    else:
        FORKING_CONTEXT = None

    if EXECUTION_MODE in [ExecutionMode.SHADOW_CACHE, ExecutionMode.SHADOW_FORK_CACHE]:
        fd, name = tempfile.mkstemp()
        os.close(fd)
        CACHE_PATH = name
    else:
        CACHE_PATH = None

    if os.environ.get('GATHER_ATEXIT', '0') == '1':
        atexit.register(t_gather_results)
    else:
        atexit.unregister(t_gather_results)

    if os.environ.get("TRACE", "0") == "1":
        sys.settrace(trace_func)
        reset_lines()


def add_strongly_killed(mut):
    global STRONGLY_KILLED
    global MASKED_MUTANTS
    MASKED_MUTANTS.add(mut)
    STRONGLY_KILLED.add(mut)
    # logger.debug(f"Strongly killed: {mut}")


def active_mutants():
    global SEEN_MUTANTS
    global MASKED_MUTANTS
    return SEEN_MUTANTS - MASKED_MUTANTS


def t_wait_for_forks():
    global FORKING_CONTEXT
    if FORKING_CONTEXT is not None:
        FORKING_CONTEXT.wait_for_forks()


def t_get_killed():
    res = {
        'strong': STRONGLY_KILLED,
        'masked': MASKED_MUTANTS,
    }
    if EXECUTION_MODE.is_shadow_variant():
        res['seen'] = SEEN_MUTANTS
    elif EXECUTION_MODE.is_split_stream_variant():
        res['active'] = NS_ACTIVE_MUTANTS
    return res


def t_counter_results():
    res = {}
    res['subject_count'] = SUBJECT_COUNTER
    res['subject_count_lines'] = sorted(SUBJECT_COUNTER_DICT.items(), key=lambda x: x[0])
    res['tool_count'] = TOOL_COUNTER
    return res


def maybe_clean_cache():
    if CACHE_PATH:
        if FORKING_CONTEXT is not None:
            if FORKING_CONTEXT.is_parent:
                Path(CACHE_PATH).unlink()


def t_gather_results() -> Any:
    disable_line_counting()
    t_wait_for_forks()
    maybe_clean_cache()

    results = t_get_killed()
    results['execution_mode'] = EXECUTION_MODE.name
    results = {**results, **t_counter_results()}
    if RESULT_FILE is not None:
        with open(RESULT_FILE, 'wt') as f:
            json.dump(results, f, cls=SetEncoder)
    logging.info(f"{results}")
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


def t_get_logical_path():
    return LOGICAL_PATH


def untaint_args(*args, **kwargs):
    all_muts = set([MAINLINE])
    for arg in args + tuple(kwargs.values()):
        if type(arg) == ShadowVariable:
            all_muts |= arg._get_paths()

    untainted_args = {}
    for mut in all_muts:

        mut_args = []
        for arg in args:
            if type(arg) == ShadowVariable:
                arg_shadow = arg._shadow
                if mut in arg_shadow:
                    mut_args.append(arg_shadow[mut])
                else:
                    mut_args.append(arg_shadow[MAINLINE])
            else:
                mut_args.append(arg)

        mut_kwargs = {}
        for name, arg in kwargs.items():
            if type(arg) == ShadowVariable:
                arg_shadow = arg._shadow
                if mut in arg_shadow:
                    mut_kwargs[name] = arg_shadow[mut]
                else:
                    mut_kwargs[name] = arg_shadow[MAINLINE]
            else:
                mut_kwargs[name] = arg

        untainted_args[mut] = (tuple(mut_args), dict(mut_kwargs))

    return untainted_args


def prune_cached_muts(muts, *args, **kwargs):
    muts = set(muts) - set([MAINLINE])
    # assert MAINLINE not in muts

    for arg in args + tuple(kwargs.values()):
        if type(arg) == ShadowVariable:
            arg._prune_muts(muts)
            
    return args, kwargs


def load_cache():
    with open(CACHE_PATH, 'rb') as cache_f:
        try:
            cache, mut_stack = pickle.load(cache_f)
        except EOFError:
            # Cache has no content yet.
            cache = {}
            mut_stack = []
    return cache, mut_stack


def save_cache(cache, mut_stack):
    with open(CACHE_PATH, 'wb') as cache_f:
        try:
            pickle.dump((cache, mut_stack), cache_f)
        except TypeError:
            raise ValueError(f"Can't serialize: {cache} {mut_stack}")


def push_cache_stack():
    if CACHE_PATH is not None:
        cache, mut_stack = load_cache()
        mut_stack.append(set())
        save_cache(cache, mut_stack)


def pop_cache_stack():
    if CACHE_PATH is not None:
        cache, mut_stack = load_cache()
        mut_stack.pop()
        save_cache(cache, mut_stack)


def maybe_mark_mutation(mutations):
    if CACHE_PATH is not None:
        cache, mut_stack = load_cache()
        muts = mutations.keys() - set([MAINLINE])
        # logger.debug(f"{cache, mut_stack}")

        for ii in range(len(mut_stack)):
            mut_stack[ii] = mut_stack[ii] | muts

        # logger.debug(f"{cache, mut_stack}")
        save_cache(cache, mut_stack)


def wrap_active_args(args: tuple) -> tuple:
    # Takes a tuple that can be used as *args during a function call and taints all elements.
    # Does not modify the tuple in any other way.
    wrapped = []
    for arg in args:
        if type(arg) == ShadowVariable:
            wrapped.append(arg)
        else:
            wrap = ShadowVariable(arg, False)._normalize(SEEN_MUTANTS, MASKED_MUTANTS)
            wrapped.append(wrap)
    return tuple(wrapped)


def wrap_active_kwargs(args: dict) -> dict:
    # Takes a dict that can be used as **kwargs during a function call and taints all elements.
    # Does not modify the dict in any other way.
    wrapped = {}
    for name, arg in args.items():
        if type(arg) == ShadowVariable:
            wrapped[name] = arg
        else:
            wrap = ShadowVariable(arg, False)._normalize(SEEN_MUTANTS, MASKED_MUTANTS)
            wrapped[name] = wrap
    return wrapped


def call_maybe_cache(f, *args, **kwargs):
    # TODO caching only for subtree of functions that are not mutated
    untainted_args = untaint_args(*args, **kwargs)
    # logging.debug(f"in: {args} {kwargs} untainted: {untainted_args}")
    if False and CACHE_PATH is not None:
        cache, mut_stack = load_cache()

        # logger.debug(f"cache: {cache} mut_stack: {mut_stack}")

        mut_is_cached = {}
        for mut, (mut_args, mut_kwargs) in untainted_args.items():

            if EXECUTION_MODE == ExecutionMode.SHADOW_CACHE:
                if mut == MAINLINE:
                    continue
            elif EXECUTION_MODE == ExecutionMode.SHADOW_FORK_CACHE:
                if LOGICAL_PATH == MAINLINE and mut == MAINLINE:
                    continue
            else:
                raise ValueError(f"Unexpected execution mode: {EXECUTION_MODE}")

            key = f"{f.__name__, mut_args, mut_kwargs}"
            if key in cache:
                cache_res = cache[key]
                mut_is_cached[mut] = cache_res
                # logger.debug(f"cached res: {key}, {cache_res}")
            
        logger.debug(f"{LOGICAL_PATH} {SEEN_MUTANTS} {MASKED_MUTANTS} {len(mut_is_cached)} {len(untainted_args)}")
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
        except ShadowException as e:
            raise e
        except Exception as e:
            message = traceback.format_exc()
            logger.error(f"Error: {message}")
            raise NotImplementedError(f"Exceptions in wrapped functions are not supported: {e}")

        res = ShadowVariable(res, from_mapping=False)
        res._keep_active(SEEN_MUTANTS, MASKED_MUTANTS)
        return res


def fork_wrap(f, *args, **kwargs):
    global FORKING_CONTEXT
    global MASKED_MUTANTS
    global STRONGLY_KILLED
    # logger.debug(f"CALL {f.__name__}({args} {kwargs}) seen: {SEEN_MUTANTS} masked: {MASKED_MUTANTS}")
    old_forking_context = FORKING_CONTEXT
    old_masked_mutants = deepcopy(MASKED_MUTANTS)

    FORKING_CONTEXT = Forker()

    push_cache_stack()
    res = call_maybe_cache(f, *args, **kwargs)
    combined_results = FORKING_CONTEXT.wait_for_forks(fork_res=(res, args, kwargs))
    pop_cache_stack()

    # Filter args and kwargs for currently available, they will be updated with the fork values.
    for arg in args:
        if type(arg) == ShadowVariable:
            arg._keep_active(SEEN_MUTANTS, MASKED_MUTANTS)

    for arg in kwargs.values():
        if type(arg) == ShadowVariable:
            arg._keep_active(SEEN_MUTANTS, MASKED_MUTANTS)

    FORKING_CONTEXT = old_forking_context
    MASKED_MUTANTS = old_masked_mutants

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


# def no_fork_wrap(f, *args, **kwargs):
#     global LOGICAL_PATH
#     global SEEN_MUTANTS
#     global MASKED_MUTANTS
#     # logger.debug(f"CALL {f.__name__}({args} {kwargs})")
#     before_logical_path = LOGICAL_PATH
#     before_active = deepcopy(ACTIVE_MUTANTS)
#     before_masked = deepcopy(MASKED_MUTANTS)

#     remaining_paths = set([0])
#     done_paths = set()

#     for arg in chain(args, kwargs.values()):
#         if type(arg) == ShadowVariable:
#             remaining_paths |= arg._get_paths()
#     remaining_paths -= before_masked

#     tainted_return = {}
#     push_cache_stack()
#     while remaining_paths:
#         LOGICAL_PATH = remaining_paths.pop()
#         logger.debug(f"cur path: {LOGICAL_PATH} remaining: {remaining_paths}")
#         ACTIVE_MUTANTS = deepcopy(before_active)
#         MASKED_MUTANTS = deepcopy(before_masked)

#         try:
#             res = call_maybe_cache(f, *args, **kwargs)
#         except ShadowException as e:
#             logger.debug(f"shadow exception: {e}")
#             remaining_paths -= STRONGLY_KILLED
#             continue 
#         after_active = deepcopy(ACTIVE_MUTANTS)
#         after_masked = deepcopy(MASKED_MUTANTS)
#         # logger.debug('wrapped: %s(%s %s) -> %s (%s)', f.__name__, args, kwargs, res, type(res))
#         new_active = (after_active or set()) - (before_active or set())
#         new_masked = after_masked - before_masked
        
#         # logger.debug("masked: %s %s %s", before_masked, after_masked, new_masked)
#         # logger.debug("active: %s %s %s", before_active, after_active, new_active)
#         if type(res) == list:
#             raise NotImplementedError("List returns are not supported.")
#             # if LOGICAL_PATH == MAINLINE:
#             #     assert MAINLINE not in tainted_return
#             #     tainted_list = []
#             #     for el in res:
#             #         if type(el) == ShadowVariable:
#             #             tainted_list.append(el)
#             #         else:
#             #             tainted_list.append(ShadowVariable({MAINLINE: el}))
#             #     tainted_return[MAINLINE] = tainted_list
#             # else:
#             #     backup_active_mutants = ACTIVE_MUTANTS
#             #     ACTIVE_MUTANTS = None
#             #     tainted_list = tainted_return[MAINLINE]
#             #     assert type(tainted_list) == list
#             #     if len(res) == len(tainted_return[MAINLINE]):
#             #         for ii in range(len(tainted_list)):
#             #             old = tainted_list[ii]
#             #             new = res[ii]
#             #             if type(new) == ShadowVariable:
#             #                 new_dict = new._shadow
#             #             else:
#             #                 new_dict = {LOGICAL_PATH: new}
#             #             tainted_list[ii] = t_combine({MAINLINE: old, **new_dict})
#             #     else:
#             #         raise NotImplementedError("Mark logical path as killed, as it has different length.")
#             #     ACTIVE_MUTANTS = backup_active_mutants
#         elif type(res) != ShadowVariable:
#             assert LOGICAL_PATH not in tainted_return, f"{LOGICAL_PATH} {tainted_return}"

#             tainted_return[LOGICAL_PATH] = res

#             if new_active:
#                 assert LOGICAL_PATH in new_active, new_active

#                 for path in after_active or set():
#                     # already recorded the LOGICAL_PATH result before the loop
#                     if path == LOGICAL_PATH:
#                         continue
#                     if path in tainted_return:
#                         tainted_return[path] == res
#                     else:
#                         tainted_return[path] = res

#         elif type(res) == ShadowVariable:
#             logger.debug(res)
#             shadow = get_active(res._shadow)
#             unused_active = new_active

#             if LOGICAL_PATH in shadow:
#                 unused_active.discard(LOGICAL_PATH)
#                 log_res = shadow[LOGICAL_PATH]
#             else:
#                 log_res = shadow[MAINLINE]

#             if LOGICAL_PATH in tainted_return:
#                 assert tainted_return[LOGICAL_PATH] == log_res, f"{tainted_return[LOGICAL_PATH]} == {log_res}"
#             else:
#                 tainted_return[LOGICAL_PATH] = log_res

#             for path, val in shadow.items():
#                 if path == MAINLINE or path == LOGICAL_PATH or path in new_masked:
#                     continue
#                 if path in tainted_return:
#                     assert tainted_return[path] == val, f"{tainted_return[path]} == {val}"
#                 else:
#                     tainted_return[path] = val
#                 unused_active.discard(path)
#                 done_paths.add(path)
            
#             for path in unused_active:
#                 tainted_return[path] = shadow[MAINLINE]
#                 done_paths.add(path)

#         # logger.debug(f"cur return {tainted_return}")

#         done_paths |= new_active
#         done_paths.add(LOGICAL_PATH)
#         remaining_paths |= new_masked
#         remaining_paths -= done_paths

#     pop_cache_stack()

#     LOGICAL_PATH = before_logical_path
#     ACTIVE_MUTANTS = before_active
#     MASKED_MUTANTS = before_masked

#     # logger.debug("return: %s", tainted_return)

#     # If only mainline in return value untaint it
#     if len(tainted_return) == 1:
#         assert MAINLINE in tainted_return, f"{tainted_return}"
#         res = tainted_return[MAINLINE]
#         return res

#     res = t_combine(tainted_return)
#     return res


def t_wrap(f):
    @wraps(f)
    def flow_wrapper(*args, **kwargs):
        if FORKING_CONTEXT:
            return fork_wrap(f, *args, **kwargs)
        else:
            raise NotImplementedError()
            # return no_fork_wrap(f, *args, **kwargs)

    return flow_wrapper


def t_cond(cond: Any) -> bool:
    global FORKING_CONTEXT
    global MASKED_MUTANTS

    if type(cond) == ShadowVariable:
        diverging_mutants = []
        companion_mutants = []

        # get the logical path result, this is used to decide which mutations follow the logical path and which do not
        logical_result = cond._get_logical_res(LOGICAL_PATH)
        assert type(logical_result) == bool, f"{cond}"

        for path, val in cond._all_path_results(SEEN_MUTANTS, MASKED_MUTANTS):
            if path == MAINLINE or path == LOGICAL_PATH:
                continue
            assert type(val) == bool, f"{cond}"
            if val == logical_result:
                companion_mutants.append(path)
            else:
                diverging_mutants.append(path)

        if FORKING_CONTEXT:
            original_path = LOGICAL_PATH
            # Fork if enabled
            if diverging_mutants:
                # logger.debug(f"path: {LOGICAL_PATH} masked: {MASKED_MUTANTS} seen: {SEEN_MUTANTS} companion: {companion_mutants} diverging: {diverging_mutants}")
                # select the path to follow, just pick first
                path = diverging_mutants[0]
                if FORKING_CONTEXT.maybe_fork(path):
                    # we are now in the forked child
                    MASKED_MUTANTS |= set(companion_mutants + [original_path])
                    return t_cond(cond)
                else:
                    MASKED_MUTANTS |= set(diverging_mutants)
        else:
            raise NotImplementedError()
            # # Follow the logical path, if that is not the same as mainline mark other mutations as inactive
            # if logical_path_is_diverging:
            #     # if ACTIVE_MUTANTS is None:
            #     ACTIVE_MUTANTS = set()
            #     MASKED_MUTANTS |= set(diverging_mutants)
            #     # logger.debug(f"masked {MASKED_MUTANTS}")
            #     for path in companion_mutants:
            #         ACTIVE_MUTANTS.add(path)

        return logical_result

    else:
        return cond
        # logger.debug("vanilla")
        # raise NotImplementedError()


def get_selected(mutations):
    if SELECTED_MUTANT is not None:
        return { path: val for path, val in mutations.items() if path in [MAINLINE, SELECTED_MUTANT] }
    else:
        return mutations


def get_active(mutations, seen, masked):
    filtered_mutations = { path: val for path, val in mutations.items() if path in seen - masked }

    # logger.debug(f"log_path: {LOGICAL_PATH}")
    filtered_mutations[MAINLINE] = mutations[MAINLINE]
    if LOGICAL_PATH in mutations:
        filtered_mutations[LOGICAL_PATH] = mutations[LOGICAL_PATH]

    return filtered_mutations


def get_ns_active(mutations, active, masked):
    if active is not None:
        filtered_mutations = { path: val for path, val in mutations.items() if path in active }
    else:
        filtered_mutations = { path: val for path, val in mutations.items() if path not in masked }

    # logger.debug(f"log_path: {LOGICAL_PATH}")
    filtered_mutations[MAINLINE] = mutations[MAINLINE]
    if LOGICAL_PATH in mutations:
        filtered_mutations[LOGICAL_PATH] = mutations[LOGICAL_PATH]

    return filtered_mutations


def get_active_shadow(val, seen, masked):
    if type(val) == ShadowVariable:
        return get_active(val._shadow, seen, masked)

    else:
        return None


def shadow_assert(cmp_result):
    global STRONGLY_KILLED
    if type(cmp_result) == ShadowVariable:
        # Do the actual assertion as would be done in the unchanged program but only for mainline execution
        if LOGICAL_PATH == MAINLINE:
            # This assert should never fail for a green test suite
            assert cmp_result._get(MAINLINE) is True, f"{cmp_result}"

        for path, res in cmp_result._all_path_results(SEEN_MUTANTS, MASKED_MUTANTS):
            assert type(res) == bool
            if not res: # assert fails for mutation
                add_strongly_killed(path)

    else:
        if not cmp_result is True:
            if LOGICAL_PATH is not MAINLINE:
                # If we are not following mainline, mark all active mutants as killed
                for mut in active_mutants():
                    add_strongly_killed(mut)
                    logger.info(f"t_assert strongly killed: {mut}")
            else:
                # If we are following mainline the test suite is not green
                assert cmp_result, f"Failed original assert"


def split_assert(cmp_result):
    # logger.debug(f"t_assert {cmp_result}")
    assert type(cmp_result) == bool, f"{type(cmp_result)}"
    if cmp_result:
        return
    else:
        if NS_ACTIVE_MUTANTS is None:
            logger.warning("NS_ACTIVE_MUTANTS is None")
            raise ValueError("NS_ACTIVE_MUTANTS is None")
        for mut in NS_ACTIVE_MUTANTS:
            add_strongly_killed(mut)
            # logger.info(f"t_assert strongly killed: {mut}")


def t_assert(cmp_result):
    if EXECUTION_MODE in [
        ExecutionMode.SHADOW,
        ExecutionMode.SHADOW_FORK,
        ExecutionMode.SHADOW_CACHE,
        ExecutionMode.SHADOW_FORK_CACHE,
    ]:
        shadow_assert(cmp_result)
    elif EXECUTION_MODE in [ExecutionMode.SPLIT_STREAM, ExecutionMode.MODULO_EQV]:
        split_assert(cmp_result)
    else:
        raise ValueError("Unknown execution mode: {EXECUTION_MODE}")


def t_logical_path():
    return LOGICAL_PATH


def t_seen_mutants():
    return SEEN_MUTANTS


def t_masked_mutants():
    return MASKED_MUTANTS


def t_ns_active_mutants():
    return NS_ACTIVE_MUTANTS


def untaint(obj):
    if hasattr(obj, '_shadow'):
        return obj._shadow[MAINLINE]
    return obj


def combine_split_stream(mutations):
    global NS_ACTIVE_MUTANTS
    global MASKED_MUTANTS

    if LOGICAL_PATH == MAINLINE:
        all_muts = set(mutations.keys())
        for mut_id, val in mutations.items():
            if mut_id in [MAINLINE, LOGICAL_PATH]:
                continue

            if isinstance(val, Exception):
                logger.debug(f"val is exception: {val}")

            if FORKING_CONTEXT.maybe_fork(mut_id):
                NS_ACTIVE_MUTANTS = set([mut_id])
                MASKED_MUTANTS = all_muts - NS_ACTIVE_MUTANTS
                return val

    try:
        return_val = mutations[LOGICAL_PATH]
    except KeyError:
        return_val = mutations[MAINLINE]

    if isinstance(return_val, Exception):
        if LOGICAL_PATH == MAINLINE:
            logger.debug(f"mainline return_val is exception: {return_val}")
            raise NotImplementedError()
        add_strongly_killed(LOGICAL_PATH)
        FORKING_CONTEXT.child_end()

    return return_val


def combine_modulo_eqv(mutations):
    global NS_ACTIVE_MUTANTS
    global MASKED_MUTANTS

    mutations = get_ns_active(mutations, NS_ACTIVE_MUTANTS, MASKED_MUTANTS)

    if LOGICAL_PATH in mutations:
        log_res = mutations[LOGICAL_PATH]
    else:
        log_res = mutations[MAINLINE]

    # if LOGICAL_PATH == MAINLINE:
    combined = defaultdict(list)
    for mut_id in set(mutations.keys()) | (NS_ACTIVE_MUTANTS or set()):
        if mut_id in [MAINLINE, LOGICAL_PATH]:
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
            if NS_ACTIVE_MUTANTS is not None:
                NS_ACTIVE_MUTANTS -= set(mut_ids)
            MASKED_MUTANTS |= set(mut_ids)
            # logger.debug(f"masked: {MASKED_MUTANTS}")
            if FORKING_CONTEXT.maybe_fork(main_mut_id):
                MASKED_MUTANTS = set()
                NS_ACTIVE_MUTANTS = set(mut_ids)
                return val

    if isinstance(log_res, Exception):
        if LOGICAL_PATH != MAINLINE:
            assert NS_ACTIVE_MUTANTS is not None
            for mut_id in NS_ACTIVE_MUTANTS:
                add_strongly_killed(mut_id)
            FORKING_CONTEXT.child_end()
        else:
            msg = f"Mainline value has exception, this indicates a not green test suite: {log_res}"
            logger.error(msg)
            raise ValueError(msg)

    try:
        return mutations[LOGICAL_PATH]
    except KeyError:
        return mutations[MAINLINE]


def t_combine_shadow(mutations: dict[int, Any]) -> Any:
    global SELECTED_MUTANT
    global SEEN_MUTANTS
    if LOGICAL_PATH == MAINLINE:
        new_paths = set(mutations.keys()) - MASKED_MUTANTS - set([MAINLINE])
        SEEN_MUTANTS |= new_paths

    evaluated_mutations = {}
    for mut, res in mutations.items():
        if (mut not in SEEN_MUTANTS or mut in MASKED_MUTANTS) and mut != MAINLINE:
            continue
        if type(res) != ShadowVariable and callable(res):
            if mut != MAINLINE:
                SELECTED_MUTANT = mut
            try:
                res = res()
            except Exception as e:
                if mut == MAINLINE:
                    for mm in active_mutants():
                        add_strongly_killed(mm)
                else:
                    add_strongly_killed(mut)
                # logger.debug(f"mut exception: {mutations} {mut} {traceback.format_exc()}")
                continue
            finally:
                SELECTED_MUTANT = None

        evaluated_mutations[mut] = res

    if EXECUTION_MODE.is_shadow_variant():
        maybe_mark_mutation(evaluated_mutations)
        res = ShadowVariable(evaluated_mutations, from_mapping=True)
        res._keep_active(SEEN_MUTANTS, MASKED_MUTANTS)
    else:
        raise NotImplementedError()
    return res


def t_combine_split_stream(mutations: dict[int, Any]) -> Any:
    global NS_ACTIVE_MUTANTS

    evaluated_mutations = {}
    for mut, res in mutations.items():
        if NS_ACTIVE_MUTANTS is not None and mut != MAINLINE and mut not in NS_ACTIVE_MUTANTS:
            continue

        if type(res) != ShadowVariable and callable(res):
            try:
                res = res()
            except Exception as e:
                res = e

        evaluated_mutations[mut] = res

    if EXECUTION_MODE is ExecutionMode.SPLIT_STREAM:
        res = combine_split_stream(evaluated_mutations)
    elif EXECUTION_MODE is ExecutionMode.MODULO_EQV:
        res = combine_modulo_eqv(evaluated_mutations)
    else:
        raise NotImplementedError()
    return res


def t_combine(mutations: dict[int, Any]) -> Any:
    if EXECUTION_MODE.is_shadow_variant():
        return t_combine_shadow(mutations)
    elif EXECUTION_MODE.is_split_stream_variant():
        return t_combine_split_stream(mutations)


###############################################################################
# tainted types

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


def convert_method_to_function(obj: object, method_name: str) -> Tuple[Callable, bool]:
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

    def _init_from_object(self, obj: Any):
        shadow = {}
        value_type = type(obj)

        # TODO optionally make sure there are no nested shadow variables in the values
        if value_type == ShadowVariable:
            shadow = obj._shadow

        elif value_type == tuple:
            # handle tuple values
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

    def _init_from_mapping(self, values: dict):
        assert type(values) == dict

        combined = {}

        try:
            mainline_val = values[MAINLINE]
        except KeyError as e:
            # mainline value is faulty, this can happen in non-mainline paths
            # example: non-mainline path goes out of bounds on array accesses
            # mark all active mutations and/or current logical path as killed
            # also if forking, stop the fork, for shadow execution just return to wrapper
            if LOGICAL_PATH != MAINLINE:
                add_strongly_killed(LOGICAL_PATH)
            if SELECTED_MUTANT == None:
                for mut in active_mutants():
                    add_strongly_killed(mut)
                if FORKING_CONTEXT is not None:
                    assert not FORKING_CONTEXT.is_parent
                    FORKING_CONTEXT.child_end() # child fork ends here
                else:
                    raise ShadowException(e)
            else:
                raise ShadowException(e)
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

        self._shadow = combined

    def __init__(self, values: Any, from_mapping: bool):
        if not from_mapping:
            self._init_from_object(values)
        else:
            self._init_from_mapping(values)

    def _duplicate_mainline(self, new_path):
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

    def __setattr__(self, name: str, value: Any) -> None:
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
                [lambda _1, _2, obj, new_val: obj.__setattr__(name, new_val)],
                self_shadow.keys(),
                lambda k: self_shadow[k],
                lambda k: value._get(k),
                tuple(),
                dict(),
                0,
            )
        else:
            # Just assign the same value to all paths.
            res = self._do_op_safely(
                [lambda _1, _2, obj, new_val: obj.__setattr__(name, new_val)],
                self_shadow.keys(),
                lambda k: self_shadow[k],
                lambda k: value,
                tuple(),
                dict(),
                0,
            )

        return res

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            # __init__ is manually defined for ShadovVariable but can also be later called during usage of ShadowVariable.
            # In the second case we want to call __init__ on the path values instead.
            # This requires a special case here.
            if name != "__init__":
                return super(ShadowVariable, self).__getattribute__(name)
        
        log_res = self._get_logical_res(LOGICAL_PATH).__getattribute__(name)
        
        if callable(log_res):
            # TODO returning a lambda here only works if the callable is only called
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

    def _callable_wrap(self, name: str, *args, **kwargs) -> Any:
        global MASKED_MUTANTS
        log_val = self._get_logical_res(LOGICAL_PATH)
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
            if LOGICAL_PATH in untainted_args:
                log_args = untainted_args[LOGICAL_PATH]
            else:
                log_args = untainted_args[MAINLINE]

            all_paths = set(shadow.keys()) | set(untainted_args.keys())
            results = {}
            for path in all_paths:
                if path == MAINLINE and LOGICAL_PATH != MAINLINE and LOGICAL_PATH in all_paths:
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
                if not method_is_next and (path == MAINLINE or path == LOGICAL_PATH):
                    path_val = deepcopy(path_val)

                # Check that path and log would use the same function.
                path_func, _ = convert_method_to_function(path_val, name)
                if path_func != logical_func:
                    raise NotImplementedError()

                # As for path val, make a copy if needed.
                if path != MAINLINE and path != LOGICAL_PATH and path in untainted_args:
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
                except KeyError as e:
                    # TODO handle exceptions in method calls
                    raise ShadowException(e)
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
                if path == MAINLINE or path == LOGICAL_PATH:
                    continue

                val_func, _ = convert_method_to_function(val, name)
                if val_func == logical_func:
                    companion_mutants.append(path)
                else:
                    diverging_mutants.append(path)

            if FORKING_CONTEXT:
                original_path = LOGICAL_PATH
                # Fork if enabled
                if diverging_mutants:
                    # logger.debug(f"path: {LOGICAL_PATH} masked: {MASKED_MUTANTS} seen: {SEEN_MUTANTS} companion: {companion_mutants} diverging: {diverging_mutants}")
                    # select the path to follow, just pick first
                    path = diverging_mutants[0]
                    if FORKING_CONTEXT.maybe_fork(path):
                        # we are now in the forked child
                        MASKED_MUTANTS |= set(companion_mutants + [original_path])
                        return self._callable_wrap(name, *args, **kwargs)
                    else:
                        MASKED_MUTANTS |= set(diverging_mutants)
            else:
                raise NotImplementedError()

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

    def _keep_active(self, seen, masked):
        self._shadow = get_active(self._shadow, seen, masked)

    def _get(self, mut) -> Any:
        return self._shadow[mut]

    def _get_logical_res(self, logical_path):
        if logical_path in self._shadow:
            return self._shadow[logical_path]
        else:
            return self._shadow[MAINLINE]

    def _all_path_results(self, seen_mutants, masked_mutants):
        paths = self._get_paths()
        for path in paths:
            if path in masked_mutants:
                continue
            yield path, self._get(path)

        for path in seen_mutants - masked_mutants - paths:
            yield path, self._get(MAINLINE)

    def _add_mut_result(self, mut: int, res: Any) -> None:
        assert mut not in self._shadow
        self._shadow[mut] = res

    def _maybe_untaint(self) -> Any:
        # Only return a untainted version if shadow only contains the mainline value and that value is a primitive type.
        if len(self._shadow) == 1:
            assert MAINLINE in self._shadow, f"{self}"

            mainline_type = type(self._shadow[MAINLINE])
            if mainline_type in PRIMITIVE_TYPES:
                return self._shadow[MAINLINE]

        return self

    def _normalize(self, seen, masked):
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
                # as_shadow[active] = child_fork_res

    def _prune_muts(self, muts):
        "Copies shadow, does not modify in place."
        assert MAINLINE not in muts
        shadow = deepcopy(self._shadow)
        for mut in muts:
            shadow.pop(mut, None)
        return ShadowVariable(shadow, from_mapping=True)

    def _do_op_safely(self, ops, paths, left, right, args, kwargs, op):
        global STRONGLY_KILLED
        
        res = {}
        for k in paths:
            op_func = ops[op]
            try:
                k_res = op_func(args, kwargs, left(k), right(k))
            except (ZeroDivisionError, OverflowError):
                if k != MAINLINE:
                    add_strongly_killed(k)
                continue
            except Exception as e:
                logger.error(f"Unknown Exception: {e}")
                raise e

            res[k] = k_res

        return res

    def _do_unary_op(self, op, *args, **kwargs):
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
            other_main = other_shadow[MAINLINE]
            self_main = self_shadow[MAINLINE]
            common_shadows = {k for k in self_shadow if k in other_shadow}
            only_self_shadows = list(k for k in self_shadow if k not in other_shadow)
            if only_self_shadows:
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
            vo_ = self._do_op_safely(
                ALLOWED_BOOL_DUNDER_METHODS,
                (k for k in other_shadow if k not in self_shadow),
                lambda k: self_main,
                lambda k: other_shadow[k],
                args,
                kwargs,
                op,
            )

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

        if PICKLE_LOAD:
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


# Init when importing shadow
reinit()
