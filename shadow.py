import os
import sys
from copy import deepcopy
from functools import wraps, partial
from contextlib import contextmanager

import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


MAINLINE = '0'
LOGICAL_PATH = '0'

STRONGLY_KILLED = {}
WEAKLY_KILLED = {}


def reinit():
    logger.info("Reinit global shadow state")
    # initializing shadow
    global LOGICAL_PATH
    global WEAKLY_KILLED
    global STRONGLY_KILLED

    LOGICAL_PATH = os.environ.get('LOGICAL_PATH', '0')
    WEAKLY_KILLED = {}
    STRONGLY_KILLED = {}


# Init when importing shadow
reinit()


def t_get_killed():
    global WEAKLY_KILLED
    global STRONGLY_KILLED

    return {
        'strong': STRONGLY_KILLED,
        'weak': WEAKLY_KILLED,
    }


def t_cond(cond):
    global WEAKLY_KILLED
    if hasattr(cond, '_shadow'):
        vs = deepcopy(cond._shadow)
        res = vs.get(LOGICAL_PATH, vs[MAINLINE])
        logger.debug("t_cond: %s %s", vs, res)
        # mark all others weakly killed.
        for k in vs:
            if vs[k] != res:
                logger.debug(f"t_cond WEAKLY_KILLED: {k}")
                WEAKLY_KILLED[k] = True
        return res
    else:
        return cond


def t_assert(bval):
    global STRONGLY_KILLED
    global WEAKLY_KILLED
    if hasattr(bval, '_shadow'):
        vs = bval._shadow
        logger.info('STRONGLY_KILLED')
        for k in sorted(vs):
            if not vs[k]:
                STRONGLY_KILLED[k] = True
                logger.info("killed: %s %s", k, vs[k])

        logger.info('WEAKLY_KILLED')
        for k in WEAKLY_KILLED:
            logger.info(k)

        # Do the actual assertion as would be done in the unchanged program
        assert vs['0']
    else:
        if MAINLINE != LOGICAL_PATH:
            STRONGLY_KILLED[LOGICAL_PATH] = True
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
            '0': right, # mainline, no mutation
            f'{mutation_counter}.1': not right, # mutation +1
        })

    if isinstance(right, int):
        return t_int({
            '0':  right, # mainline, no mutation
            f'{mutation_counter}.1': right + 1, # mutation +1
        })

    return right


def t_aug_add(mutation_counter, left, right):
    if isinstance(left, (int, t_int)) and isinstance(right, (int, t_int)):
        return t_int({
            '0': left + right, # mainline -- no mutation
            f'{mutation_counter}.1': untaint(left - right), # mutation op +/-
        }) + t_int({
            '0':0, # main line -- no mutation
            f'{mutation_counter}.2':1, # mutation +1
        })
    else:
        raise ValueError(f"Unhandled tainted add types: {right} {left}")


def t_aug_sub(mutation_counter, left, right):
    if isinstance(left, (int, t_int)) and isinstance(right, (int, t_int)):
        return t_int({
            '0': left - right, # mainline -- no mutation
            f'{mutation_counter}.1': untaint(left + right), # mutation op +/-
        }) + t_int({
            '0':0, # main line -- no mutation
            f'{mutation_counter}.2':1, # mutation +1
        })
    else:
        raise ValueError(f"Unhandled tainted add types: {right} {left}")


def t_aug_mult(mutation_counter, left, right):
    if isinstance(left, (int, t_int)) and isinstance(right, (int, t_int)):
        return t_int({
            '0': left * right, # mainline -- no mutation
            f'{mutation_counter}.1': untaint(left / right), # mutation op *//
        })
    else:
        raise ValueError(f"Unhandled tainted add types: {right} {left}")


###############################################################################
# tainted types


def init_shadow(cls, ty, shadow):
    logger.debug("init_shadow %s %s %s", cls, ty, shadow)
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
    logger.debug("tainted op: %s %s", first, other)
    vs = first._shadow
    if type(other) in primitive_kind:
        # now do the operation on all values.
        res = {k: op(vs[k], other) for k in vs}
        logger.debug("primitive res: %s", res)
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
        return {'0': op(first, other)}


def losing_taint(self):
    raise NotImplementedError(
        "Casting to a plain bool loses all taint information. "
        "Raise exception here to avoid unexpectedly losing information."
    )


def proxy_function(cls, name, f):
    @wraps(f)
    def proxied_f(*args, **kwargs):
        res = f(*args, **kwargs)
        logger.debug('%s %s: %s %s -> %s (%s)', cls, name, args, kwargs, res, type(res))
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
        logging.debug("%s %s", orig_class, func)
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
        self.len = t_int({'0': len(self.val)})

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