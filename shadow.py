import os
import sys
from copy import deepcopy
from contextlib import contextmanager

import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


MAINLINE = '0'
LOGICAL_PATH = '0'
WEAKLY_KILLED = {}


def init():
    # initializing shadow
    global LOGICAL_PATH
    LOGICAL_PATH = os.environ.get('LOGICAL_PATH', '0')


def t_cond(cond):
    if hasattr(cond, '_vhash'):
        vs = deepcopy(cond._vhash)
        res = vs.get(LOGICAL_PATH, vs[MAINLINE])
        # mark all others weakly killed.
        for k in vs:
            if vs[k] != res:
                logger.debug(f"t_cond WEAKLY_KILLED: {k}")
                WEAKLY_KILLED[k] = True
        return res
    else:
        return cond


def t_assert(bval):
    if hasattr(bval, '_vhash'):
        vs = bval._vhash
        logger.info('STRONGLY_KILLED')
        for k in sorted(vs):
            if not vs[k]:
                logger.info("killed: %s %s", k, vs[k])

        logger.info('WEAKLY_KILLED')
        for k in WEAKLY_KILLED:
            logger.info(k)

        # Do the actual assertion as would be done in the unchanged program
        assert vs['0']
    else:
        if MAINLINE != LOGICAL_PATH:
            logger.info(f'STRONGLY_KILLED: {LOGICAL_PATH}')
        else:
            assert bval


def tainted_op(first, other, op, primitive_kind):
    logger.debug("op: %s %s", first, other)
    vs = first._vhash
    if type(other) in primitive_kind:
        # now do the operation on all values.
        return {k:op(vs[k],other) for k in vs}
    elif hasattr(other, '_vhash'):
        vo = other._vhash
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
        return {**vs_, **vo_, **cs_}
    else:
        assert False


def untaint(obj):
    if hasattr(obj, '_vhash'):
        return obj._vhash[MAINLINE]
    return obj


class tint(int):
    def __new__(cls, vhash, *args, **kwargs):
        oval = vhash[MAINLINE]
        if type(oval) == tint:
            # mainline contains taints, to transmit them.
            vs = oval._vhash
            res =  super(cls, cls).__new__(cls, vs[MAINLINE])
            res._vhash = {**vhash, **vs}
        else:
            res =  super(cls, cls).__new__(cls, oval)
            res._vhash = vhash
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        vs = tainted_op(self, other, lambda x, y: x + y, {int})
        return self.__class__({**self._vhash, **vs})

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        vs = tainted_op(self, other, lambda x, y: x * y, {int})
        return self.__class__({**self._vhash, **vs})

    def __div__(self, other):
        vs = tainted_op(self, other, lambda x, y: x / y, {int})
        return tfloat_({**self._vhash, **vs})

    def __eq__(self, other):
        vs = tainted_op(self, other, lambda x, y: x == y, {int})
        return tbool({**self._vhash, **vs})

    def __ne__(self, other):
        vs = tainted_op(self, other, lambda x, y: x != y, {int})
        return tbool({**self._vhash, **vs})

    def __lt__(self, other):
        vs = tainted_op(self, other, lambda x, y: x < y, {int})
        return tbool({**self._vhash, **vs})

    def __le__(self, other):
        vs = tainted_op(self, other, lambda x, y: x <= y, {int})
        return tbool({**self._vhash, **vs})

    def __ge__(self, other):
        vs = tainted_op(self, other, lambda x, y: x >= y, {int})
        return tbool({**self._vhash, **vs})

    def __gt__(self, other):
        vs = tainted_op(self, other, lambda x, y: x > y, {int})
        return tbool({**self._vhash, **vs})

    def __str__(self):
        return "%d" % int(self)

    def __repr__(self):
        return "int_t(%d)" % int(self)

class tbool(int):
    def __new__(cls, vhash, *args, **kwargs):
        oval = vhash[MAINLINE]
        res =  super(cls, cls).__new__(cls, oval)
        res._vhash = vhash
        return res

    def __eq__(self, other):
        vs = tainted_op(self, other, lambda x, y: x == y, {bool, int, float, str})
        return tbool({**self._vhash, **vs})

    def __ne__(self, other):
        vs = tainted_op(self, other, lambda x, y: x != y, {int, float})
        return tbool({**self._vhash, **vs})

    def __str__(self):
        return "%s" % bool(self)

    def __repr__(self):
        return "bool_t(%s)" % bool(self)


class tfloat_(float):
    def __new__(cls, vhash, *args, **kwargs):
        oval = vhash[MAINLINE]
        res =  super(cls, cls).__new__(cls, oval)
        res._vhash = vhash
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        vs = tainted_op(self, other, lambda x, y: x + y, {float})
        return self.__class__({**self._vhash, **vs})

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        vs = tainted_op(self, other, lambda x, y: x * y, {float})
        return self.__class__({**self._vhash, **vs})

    def __div__(self, other):
        vs = tainted_op(self, other, lambda x, y: x / y, {float})
        return tfloat_({**self._vhash, **vs})

    def __eq__(self, other):
        vs = tainted_op(self, other, lambda x, y: x == y, {float})
        return tbool({**self._vhash, **vs})

    def __lt__(self, other):
        vs = tainted_op(self, other, lambda x, y: x < y, {float})
        return tbool({**self._vhash, **vs})

    def __ne__(self, other):
        vs = tainted_op(self, other, lambda x, y: x != y, {int})
        return tbool({**self._vhash, **vs})

    def __lt__(self, other):
        vs = tainted_op(self, other, lambda x, y: x < y, {int})
        return tbool({**self._vhash, **vs})

    def __le__(self, other):
        vs = tainted_op(self, other, lambda x, y: x <= y, {int})
        return tbool({**self._vhash, **vs})

    def __ge__(self, other):
        vs = tainted_op(self, other, lambda x, y: x >= y, {int})
        return tbool({**self._vhash, **vs})

    def __gt__(self, other):
        vs = tainted_op(self, other, lambda x, y: x > y, {int})
        return tbool({**self._vhash, **vs})


    def __str__(self):
        return "%f" % float(self)

    def __repr__(self):
        return "float_t(%f)" % float(self)


@contextmanager
def t_context():
    yield


def t_assign(mutation_counter, right):

    if isinstance(right, bool):
        return tbool({
            '0': right, # mainline, no mutation
            f'{mutation_counter}.1': not right, # mutation +1
        })

    if isinstance(right, int):
        return tint({
            '0':  right, # mainline, no mutation
            f'{mutation_counter}.1': right + 1, # mutation +1
        })

    return right


def t_aug_add(mutation_counter, left, right):
    if isinstance(left, (int, tint)) and isinstance(right, (int, tint)):
        return tint({
            '0': left + right, # mainline -- no mutation
            f'{mutation_counter}.1': untaint(left - right), # mutation op +/-
        }) + tint({
            '0':0, # main line -- no mutation
            f'{mutation_counter}.2':1, # mutation +1
        })
    else:
        raise ValueError(f"Unhandled tainted add types: {right} {left}")


def t_aug_sub(mutation_counter, left, right):
    if isinstance(left, (int, tint)) and isinstance(right, (int, tint)):
        return tint({
            '0': left - right, # mainline -- no mutation
            f'{mutation_counter}.1': untaint(left + right), # mutation op +/-
        }) + tint({
            '0':0, # main line -- no mutation
            f'{mutation_counter}.2':1, # mutation +1
        })
    else:
        raise ValueError(f"Unhandled tainted add types: {right} {left}")


def t_aug_mult(mutation_counter, left, right):
    if isinstance(left, (int, tint)) and isinstance(right, (int, tint)):
        return tint({
            '0': left * right, # mainline -- no mutation
            f'{mutation_counter}.1': untaint(left / right), # mutation op *//
        })
    else:
        raise ValueError(f"Unhandled tainted add types: {right} {left}")


init()
