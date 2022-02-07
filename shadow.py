# mypy: ignore-errors
# This file requires too much metaprogramming to provide correct typing for mypy.

from json.decoder import JSONDecodeError
import pickle
import os
import sys
import json
import tempfile
import time
import atexit
import traceback
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
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(process)d %(filename)s:%(lineno)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

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
            if type(fork_res) == ShadowVariable:
                results['fork_res'] = fork_res
            else:
                assert type(fork_res) != dict
                results['fork_res'] = fork_res
            # logger.debug(f"child results to write: {results}")
            pickle.dump(results, f)

        # exit the child immediately, this might cause problems for programs actually using multiprocessing
        # but this is a prototype
        os._exit(0)

    def wait_for_forks(self, fork_res=None):
        global SUBJECT_COUNTER
        global TOOL_COUNTER
        # if child, write results and exit
        if not self.is_parent:
            self.child_end(fork_res)

        # wait for all child processes to end
        combined_fork_res = [get_active_shadow(fork_res, SEEN_MUTANTS, MASKED_MUTANTS)]
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
                                child_results = pickle.load(f)
                            except JSONDecodeError:
                                # Child has not yet written the results.
                                continue

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
    logger.info("Reinit global shadow state")
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


def call_maybe_cache(f, *args, **kwargs):
    untainted_args = untaint_args(*args, **kwargs)
    # logging.debug(f"in: {args} {kwargs} untainted: {untainted_args}")
    if CACHE_PATH is not None:
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
            res = ShadowVariable(mut_is_cached, from_mapping=True, wraps_container=False)
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
                res = ShadowVariable({MAINLINE: res}, from_mapping=True, wraps_container=False)

            if cache_updated:
                save_cache(cache, mut_stack)

            # insert cached results
            # logger.debug(f"{mut_is_cached}, {res}")
            res = ShadowVariable({**mut_is_cached, **res._shadow}, from_mapping=True, wraps_container=False)

        # logger.debug(f"{res}")
        return res

    else:
        # no caching, just do it normally
        try:
            active_args = tuple(ShadowVariable(aa, False, False)._normalize(SEEN_MUTANTS, MASKED_MUTANTS) for aa in args)
            active_kwargs = dict({kk: ShadowVariable(aa, False, False)._normalize(SEEN_MUTANTS, MASKED_MUTANTS) for kk, aa in kwargs.items()})
            logger.debug(f"{f} {len(active_args), len(active_kwargs)} {active_args} {active_kwargs}")
            res = f(*active_args, **active_kwargs)
        except ShadowException as e:
            raise e
        except Exception as e:
            message = traceback.format_exc()
            logger.error(f"Error: {message}")
            raise NotImplementedError(f"Exceptions in wrapped functions are not supported: {e}")

        res = ShadowVariable(res, False, False)
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
    combined_results = FORKING_CONTEXT.wait_for_forks(fork_res=res)
    pop_cache_stack()

    FORKING_CONTEXT = old_forking_context
    MASKED_MUTANTS = old_masked_mutants

    res = ShadowVariable(combined_results[0], from_mapping=True, wraps_container=False)

    for child_res in combined_results[1:]:
        child_fork_res = child_res['fork_res']
        seen = child_res['seen']
        masked = child_res['masked']
        res._merge(child_fork_res, seen, masked)

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
            assert cmp_result._get(MAINLINE), f"{cmp_result}"

        for path, res in cmp_result._all_path_results(SEEN_MUTANTS, MASKED_MUTANTS):
            assert type(res) == bool
            if not res: # assert fails for mutation
                add_strongly_killed(path)

    else:
        if not cmp_result: # if assert is false
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
                # logger.debug(f"mut exception: {mutations} {mut} {e}")
                continue
            finally:
                SELECTED_MUTANT = None

        evaluated_mutations[mut] = res

    if EXECUTION_MODE.is_shadow_variant():
        maybe_mark_mutation(evaluated_mutations)
        res = ShadowVariable(evaluated_mutations, True, False)
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

DISALLOWED_DUNDER_METHODS = [
    '__aenter__', '__aexit__', '__aiter__', '__anext__', '__await__',
    '__bytes__', '__call__', '__cmp__', '__complex__', '__contains__',
    '__delattr__', '__delete__', '__delitem__', '__delslice__',  
    '__enter__', '__exit__', '__fspath__',
    '__get__', '__getitem__', '__getnewargs__', '__getslice__', 
    '__hash__', '__import__', '__imul__', '__index__',
    '__int__', '__invert__',
    '__ior__', '__iter__', '__ixor__', 
    '__next__', '__nonzero__',
    '__pos__', '__prepare__', '__rdiv__',
    '__rdivmod__', '__repr__', '__reversed__',
    '__set__', '__setitem__',
    '__setslice__', '__sizeof__', '__subclasscheck__', '__subclasses__',
    '__divmod__',
    '__div__',
    '__len__',
    

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
    '__iadd__', '__iand__', '__isub__', 
]

class ShadowVariable():
    _shadow: dict[int, Any]
    __slots__ = ['_shadow', '_container']

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

    # TODO instead of bools maybe create specific init functions
    def __init__(self, values: dict[int, Any], from_mapping: bool, wraps_container: bool):
        value_type = type(values)

        if value_type == ShadowVariable:
            # TODO self._shadow = deepcopy(values._shadow)
            self._shadow = values._shadow
            self._container = values._container
            return

        self._container = wraps_container
        combined = {}
        if not from_mapping:
            # TODO could make sure there are no shadow variables in the values
            assert value_type != dict # for testing
            combined[MAINLINE] = values
        else:
            assert value_type == dict

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
                # if mut_id != LOGICAL_PATH:
                #     assert mut_id not in combined
                if type(val) == ShadowVariable:
                    val = val._shadow
                    if mut_id in val:
                        combined[mut_id] = val[mut_id]
                    else:
                        combined[mut_id] = val[MAINLINE]
                else:
                    if mut_id not in combined:
                        combined[mut_id] = val
                    else:
                        assert False
                        assert combined[mut_id] == val

        self._shadow = combined

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            # logger.debug(f"super setattr {name, value}")
            return super(ShadowVariable, self).__setattr__(name, value)

        # TODO duplicate contained object if new path is added, otherwise just update
        self_shadow = get_selected(self._shadow)
        other_shadow = get_selected(value._shadow)

        mainline_val = self._shadow[MAINLINE]

        for other_path in other_shadow:
            if other_path not in self_shadow:
                # TODO this should be put with the class taint wrapper code
                if mainline_val._init:
                    self._shadow[other_path] = deepcopy(mainline_val)
                else:
                    # This fails when __init__ has not yet been called, start this off 
                    new = mainline_val._orig_new(type(mainline_val))
                    self._shadow[other_path] = new

        # res = self._do_op_safely(self_shadow.keys(), lambda k: self_shadow[k].__getattribute__(op)(*args, **kwargs))
        res = self._do_op_safely(
            [lambda _1, _2, obj, new_val: obj.__setattr__(name, new_val)],
            self_shadow.keys(),
            lambda k: self_shadow[k],
            lambda k: value._get(k),
            tuple(),
            dict(),
            0,
        )
        logger.debug(f"setattr {name, value} -> {res}")

    def __getattr__(self, name: str) -> Any:
        logger.debug(f"getattr {name}")
        if name.startswith("_"):
            res = super(ShadowVariable, self).__getattr__(name)
            logger.debug(f"res {res}")
            return res
        res = object.__getattr__(self, name)
        raise NotImplementedError()

    def __getattribute__(self, name: str) -> Any:
        res = super(ShadowVariable, self).__getattribute__(name)
        if name.startswith("_"):
            if name != "__init__":
                return res
        
        log_res = self._get_logical_res(LOGICAL_PATH).__getattribute__(name)
        
        if callable(log_res):
            # TODO returning a lambda here works if the callable is only called
            # it does not work if the callable is compared to some other function or other edge cases
            # maybe return a dedicated function instead that raises errors for edge cases / implements them correctly
            return lambda *args, **kwargs: self._callable_wrap(name, *args, **kwargs)
            
        logger.debug(f"getattribute {name} {log_res} {type(log_res)}")
        raise NotImplementedError()
        # return res

    def _callable_wrap(self, name: str, *args, **kwargs) -> Any:
        global MASKED_MUTANTS
        log_res = self._get_logical_res(LOGICAL_PATH).__getattribute__(name)
        assert callable(log_res)
        # convert bound method to free standing function, this allows changing the self argument
        log_res = log_res.__func__

        logger.debug(f"callable {name} {args} {kwargs} {log_res}")

        diverging_mutants = []
        companion_mutants = []

        for path, val in self._shadow.items():
            if path == MAINLINE or path == LOGICAL_PATH:
                continue
            val_attr = val.__getattribute__(name)
            logger.debug(f"{path} {val} {val_attr}")
            if val_attr == log_res:
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
                    # TODO really return here?
                    return self._callable_wrap(name, *args, **kwargs)
                else:
                    MASKED_MUTANTS |= set(diverging_mutants)
        else:
            raise NotImplementedError()

        wrapped_fun = t_wrap(log_res)
        return wrapped_fun(self, *args, **kwargs)

        # TODO split up arguments as is done in function wrapping (unify if possible)
        # TODO for each path in self and arguments apply do the function, creating new paths in self if needed
        # TODO if different functions, mask paths not following mainline
        # TODO correctly wrap returns into one shadowvariable
        raise NotImplementedError()

    def __repr__(self):
        return f"ShadowVariable({self._shadow})"

    def __reduce__(self):
        return (self.__class__, (self._shadow, True)) # TODO test reduce on wrapped object

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

        for path in seen_mutants - masked_mutants:
            if path not in paths:
                yield path, self._get(MAINLINE)

    def _add_mut_result(self, mut: int, res: Any) -> None:
        assert mut not in self._shadow
        self._shadow[mut] = res

    def _maybe_untaint(self) -> Any:
        if self._container:
            return self
        elif len(self._shadow) == 1:
            assert MAINLINE in self._shadow, f"{self}"
            return self._shadow[MAINLINE]
        else:
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
        return ShadowVariable(res, from_mapping=True, wraps_container=False)

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
            return ShadowVariable(res, from_mapping=True, wraps_container=False)
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
            return ShadowVariable(res, from_mapping=True, wraps_container=False)


# Init when importing shadow
reinit()
