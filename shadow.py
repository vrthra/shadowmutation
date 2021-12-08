from json.decoder import JSONDecodeError
import os
import sys
import json
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from copy import deepcopy
from functools import wraps, partial
from contextlib import contextmanager

import logging
from typing import Union
logging.basicConfig(stream=sys.stdout, level=logging.WARN, format='%(process)d %(message)s')
logger = logging.getLogger(__name__)


MAINLINE = '0'
LOGICAL_PATH = '0'
SPLIT_STREAM_EXECUTION = True


class SplitStreamForker():
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
        path_file = self.sync_dir.joinpath('paths', path)
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
            return False
        else:
            # We are the child, update that this is the child.
            self.is_parent = False

            # Upodate which path child is supposed to follow
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
                for res in ['strong', 'weak', 'masked']:
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

                    for res in ['strong', 'weak', 'masked']:
                        child_results[res] = set(child_results[res])

                    for res in ['strong', 'weak']:
                        all_results[res] |= child_results[res] - child_results['masked']

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
MASKED = None
FORKING_CONTEXT: Union[None, SplitStreamForker] = None


def reinit(logical_path: str=None, split_stream: bool=None):
    logger.info("Reinit global shadow state")
    # initializing shadow
    global LOGICAL_PATH
    global STRONGLY_KILLED
    global WEAKLY_KILLED
    global MASKED
    global SPLIT_STREAM_EXECUTION
    global FORKING_CONTEXT

    if logical_path is not None:
        LOGICAL_PATH = logical_path
    else:
        LOGICAL_PATH = os.environ.get('LOGICAL_PATH', MAINLINE)

    if split_stream is not None:
        SPLIT_STREAM_EXECUTION = split_stream
    else:
        SPLIT_STREAM_EXECUTION = os.environ.get('SPLIT_STREAM_EXECUTION', '1') == '1'

    WEAKLY_KILLED = set()
    STRONGLY_KILLED = set()
    MASKED = set()

    if SPLIT_STREAM_EXECUTION:
        FORKING_CONTEXT = SplitStreamForker()
    else:
        FORKING_CONTEXT = None


# Init when importing shadow
reinit()


def wait_for_forks():
    global FORKING_CONTEXT
    if FORKING_CONTEXT is not None:
        FORKING_CONTEXT.wait_for_forks()


def t_get_killed():
    global WEAKLY_KILLED
    global STRONGLY_KILLED
    global MASKED

    return {
        'strong': STRONGLY_KILLED,
        'weak': WEAKLY_KILLED,
        'masked': MASKED,
    }


def t_cond(cond):
    global WEAKLY_KILLED
    global FORKING_CONTEXT
    global MASKED

    shadow = get_shadow(cond)

    if shadow is not None:
        logger.debug(f"shadow {shadow} {LOGICAL_PATH}")
        res = shadow.get(LOGICAL_PATH, shadow.get(MAINLINE))
        logger.debug("t_cond: (%s, %s) %s", LOGICAL_PATH, res, shadow)

        # mark all others weakly killed.
        forking_path = None
        for path, path_val in shadow.items():
            if path in WEAKLY_KILLED:
                continue

            if path != MAINLINE and path_val != res:
                if SPLIT_STREAM_EXECUTION and forking_path is None:
                    forking_path = (path, path_val)
                logger.info(f"t_cond weakly_killed: {path}")
                WEAKLY_KILLED.add(path)

        if forking_path is not None:
            logger.debug(f"fork path: {forking_path}")
            fork_path, fork_val = forking_path
            if FORKING_CONTEXT.maybe_fork(fork_path):
                # in forked child
                for path, path_val in shadow.items():
                    if path_val == fork_val:
                        WEAKLY_KILLED.discard(path)
                    else:
                        MASKED.add(path)
                return t_cond(cond)
        
        return res
    else:
        return cond


def get_shadow(val):
    if hasattr(val, '_shadow'):
        shadow = val._shadow
        logger.debug(f"{shadow}")
        filtered_shadow = {
            path: val for path, val in shadow.items()
            if path not in STRONGLY_KILLED and path not in WEAKLY_KILLED and path not in MASKED
        }

        logger.debug(f"log_path: {LOGICAL_PATH}")
        if LOGICAL_PATH not in shadow:
            filtered_shadow[MAINLINE] = shadow[MAINLINE]
        else:
            filtered_shadow[LOGICAL_PATH] = shadow[LOGICAL_PATH]

        return filtered_shadow

    else:
        return None


def t_assert(bval):
    global STRONGLY_KILLED
    global WEAKLY_KILLED
    shadow = get_shadow(bval)
    logger.debug(f"t_assert {bval} {shadow}")
    if shadow is not None:
        for path, val in shadow.items():
            if path == MAINLINE:
                continue
            if not val:
                STRONGLY_KILLED.add(path)
                logger.info(f"t_assert strongly killed: {path}")

        # Do the actual assertion as would be done in the unchanged program but only for mainline execution
        if LOGICAL_PATH == MAINLINE:
            assert shadow[MAINLINE]
    else:
        if MAINLINE != LOGICAL_PATH:
            STRONGLY_KILLED.add(LOGICAL_PATH)
            logger.info(f'STRONGLY_KILLED: {LOGICAL_PATH}')
        else:
            assert bval


def untaint(obj):
    if hasattr(obj, '_shadow'):
        return obj._shadow[MAINLINE]
    return obj


@contextmanager
def t_context():
    yield


def t_assign(mutation_counter, right):

    if isinstance(right, bool):
        return t_bool({
            MAINLINE: right, # mainline, no mutation
            f'{mutation_counter}.1': not right, # mutation +1
        })

    if isinstance(right, int):
        return t_int({
            MAINLINE:  right, # mainline, no mutation
            f'{mutation_counter}.1': right + 1, # mutation +1
        })

    return right


def t_aug_add(mutation_counter, left, right):
    if isinstance(left, (int, t_int)) and isinstance(right, (int, t_int)):
        return t_int({
            MAINLINE: left + right, # mainline -- no mutation
            f'{mutation_counter}.1': untaint(left - right), # mutation op +/-
        }) + t_int({
            MAINLINE:0, # main line -- no mutation
            f'{mutation_counter}.2':1, # mutation +1
        })
    else:
        raise ValueError(f"Unhandled tainted add types: {right} {left}")


def t_aug_sub(mutation_counter, left, right):
    if isinstance(left, (int, t_int)) and isinstance(right, (int, t_int)):
        return t_int({
            MAINLINE: left - right, # mainline -- no mutation
            f'{mutation_counter}.1': untaint(left + right), # mutation op +/-
        }) + t_int({
            MAINLINE:0, # main line -- no mutation
            f'{mutation_counter}.2':1, # mutation +1
        })
    else:
        raise ValueError(f"Unhandled tainted add types: {right} {left}")


def t_aug_mult(mutation_counter, left, right):
    if isinstance(left, (int, t_int)) and isinstance(right, (int, t_int)):
        return t_int({
            MAINLINE: left * right, # mainline -- no mutation
            f'{mutation_counter}.1': untaint(left / right), # mutation op *//
        })
    else:
        raise ValueError(f"Unhandled tainted add types: {right} {left}")


###############################################################################
# tainted types


def init_shadow(cls, ty, shadow):
    # logger.debug("init_shadow %s %s %s", cls, ty, shadow)
    res = {}
    mainline_shadow = shadow[MAINLINE]
    if isinstance(mainline_shadow, cls):
        res |= mainline_shadow._shadow
    else:
        res[MAINLINE] = mainline_shadow

    for mut_id, val in shadow.items():
        if mut_id == MAINLINE:
            continue
        assert type(val) == ty
        res[mut_id] = val

    return res


def taint_primitive(val):
    if isinstance(val, bool):
        return t_bool({MAINLINE: val})
    elif isinstance(val, int):
        return t_int({MAINLINE: val})
    else:
        raise NotImplementedError(f"Unknown primitive type, can not taint it: {val} {type(val)}")


def tainted_op(first, other, op, primitive_kind):
    vs = first._shadow
    if type(other) in primitive_kind:
        # now do the operation on all values.
        res = {k: op(vs[k], other) for k in vs}
        return res
    elif hasattr(other, '_shadow'):
        vo = other._shadow
        # notice that both self and other has taints.
        # the result we need contains taints from both.
        other_main = vo[MAINLINE]
        self_main = vs[MAINLINE]
        cs = {k for k in vs if k in vo}
        vs_ = {k:op(vs[k],other_main) for k in vs if k not in vo}
        vo_ = {k:op(vo[k],self_main) for k in vo if k not in vs}

        # if there was a preexisint taint of the same name, this mutation was
        # already executed. So, use that value.
        cs_ = {k:op(vs[k], vo[k]) for k in cs}
        #assert vs[MAINLINE] == os[MAINLINE]
        res = {**vs_, **vo_, **cs_}
        logger.debug("res: %s", res)
        return res
    else:
        logger.warning(f"unexpected arguments to tainted_op: {first} {other} {op} {primitive_kind}")
        assert False


def maybe_tainted_op(first, other, op, primitive_kind):
    logger.debug("maybe tainted op: %s %s", first, other)
    if hasattr(first, '_shadow'):
        return tainted_op(first, other, op, primitive_kind)
    elif hasattr(other, '_shadow'):
        first = taint_primitive(first, primitive_kind)
        return tainted_op(first, other, op, primitive_kind)
    else:
        return {MAINLINE: op(first, other)}


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
class t_bool():
    __slots__ = ['_shadow']

    def __init__(self, shadow):
        self._shadow = init_shadow(type(self), bool, shadow)

    def __eq__(self, other):
        vs = tainted_op(self, other, lambda x, y: x == y, {bool, int, float})
        return t_bool({**self._shadow, **vs})

    def __ne__(self, other):
        vs = tainted_op(self, other, lambda x, y: x != y, {bool, int, float})
        return t_bool({**self._shadow, **vs})

    def __or__(self, other: int) -> int:
        vs = tainted_op(self, other, lambda x, y: x | y, {bool, int, float})
        return t_bool({**self._shadow, **vs})

    def __ror__(self, other: int) -> int:
        vs = tainted_op(self, other, lambda x, y: x | y, {bool, int, float})
        return t_bool({**self._shadow, **vs})

    def __and__(self, other: int) -> int:
        vs = tainted_op(self, other, lambda x, y: x & y, {bool, int, float})
        return t_bool({**self._shadow, **vs})

    def __rand__(self, other: int) -> int:
        vs = tainted_op(self, other, lambda x, y: x & y, {bool, int, float})
        return t_bool({**self._shadow, **vs})

    # def __str__(self):
    #     return "%s" % bool(self)

    def __repr__(self):
        return "t_bool %s" % self._shadow


@taint
class t_int():
    __slots__ = ['_shadow']

    def __init__(self, shadow):
        self._shadow = init_shadow(type(self), int, shadow)

    def __int__(self):
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        vs = tainted_op(self, other, lambda x, y: x + y, {int})
        return self.__class__({**self._shadow, **vs})

    def __sub__(self, other):
        vs = tainted_op(self, other, lambda x, y: x - y, {int})
        return self.__class__({**self._shadow, **vs})

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        vs = tainted_op(self, other, lambda x, y: x * y, {int})
        return self.__class__({**self._shadow, **vs})

    def __div__(self, other):
        vs = tainted_op(self, other, lambda x, y: x / y, {int})
        return t_float({**self._shadow, **vs})

    def __eq__(self, other):
        vs = tainted_op(self, other, lambda x, y: x == y, {int})
        return t_bool({**self._shadow, **vs})

    def __ne__(self, other):
        logger.debug("%s %s", self, other)
        vs = tainted_op(self, other, lambda x, y: x != y, {int})
        return t_bool({**self._shadow, **vs})

    def __lt__(self, other):
        vs = tainted_op(self, other, lambda x, y: x < y, {int})
        return t_bool({**self._shadow, **vs})

    def __le__(self, other):
        vs = tainted_op(self, other, lambda x, y: x <= y, {int})
        return t_bool({**self._shadow, **vs})

    def __ge__(self, other):
        vs = tainted_op(self, other, lambda x, y: x >= y, {int})
        return t_bool({**self._shadow, **vs})

    def __gt__(self, other):
        vs = tainted_op(self, other, lambda x, y: x > y, {int})
        return t_bool({**self._shadow, **vs})

    def __str__(self):
        return "t_int %s" % self._shadow

    def __repr__(self):
        return "t_int %s" % self._shadow


@taint
class t_float():
    __slots__ = ['_shadow']

    def __init__(self, shadow):
        self._shadow = init_shadow(type(self), float, shadow)

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        vs = tainted_op(self, other, lambda x, y: x + y, {float})
        return self.__class__({**self._shadow, **vs})

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        vs = tainted_op(self, other, lambda x, y: x * y, {float})
        return self.__class__({**self._shadow, **vs})

    def __div__(self, other):
        vs = tainted_op(self, other, lambda x, y: x / y, {float})
        return t_float({**self._shadow, **vs})

    def __eq__(self, other):
        vs = tainted_op(self, other, lambda x, y: x == y, {float})
        return t_bool({**self._shadow, **vs})

    def __lt__(self, other):
        vs = tainted_op(self, other, lambda x, y: x < y, {float})
        return t_bool({**self._shadow, **vs})

    def __ne__(self, other):
        vs = tainted_op(self, other, lambda x, y: x != y, {int})
        return t_bool({**self._shadow, **vs})

    def __lt__(self, other):
        vs = tainted_op(self, other, lambda x, y: x < y, {int})
        return t_bool({**self._shadow, **vs})

    def __le__(self, other):
        vs = tainted_op(self, other, lambda x, y: x <= y, {int})
        return t_bool({**self._shadow, **vs})

    def __ge__(self, other):
        vs = tainted_op(self, other, lambda x, y: x >= y, {int})
        return t_bool({**self._shadow, **vs})

    def __gt__(self, other):
        vs = tainted_op(self, other, lambda x, y: x > y, {int})
        return t_bool({**self._shadow, **vs})

    def __str__(self):
        return "%f" % float(self)

    def __repr__(self):
        return "float_t(%f)" % float(self)


@taint
class t_tuple():
    def __init__(self, *args, **kwargs):
        self.val = tuple(*args, **kwargs)
        self.len = t_int({MAINLINE: len(self.val)})

    def __iter__(self):
        for elem in self.val:
            yield elem

    def __eq__(self, other):
        res = self.len == other.__len__()
        if not t_cond(res):
            return res

        for a, b in zip(self, other):
            new_res = t_bool(maybe_tainted_op(a, b, lambda x, y: x == y, {bool, int}))
            res &= new_res

        return res

        return t_bool({**self._shadow, **vs})

    def __len__(self):
        # return self.len._shadow[MAINLINE]
        return self.len

    def __str__(self):
        return f"t_tuple {getattr(self, 'len', None)} {getattr(self, 'val', None)}"

    def __repr__(self):
        return f"t_tuple {getattr(self, 'len', None)} {getattr(self, 'val', None)}"


@taint
class t_list():
    pass