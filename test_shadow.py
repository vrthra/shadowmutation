import pytest
import logging
logger = logging.getLogger(__name__)

from shadow import reinit, t_wrap, t_combine, t_wait_for_forks, t_get_killed, t_cond, t_assert, t_tuple

MODES = ['shadow_fork'] # , 'shadow_fork', 'shadow_cache', 'shadow_fork_cache']


def gen_killed(strong):
    return {
        'strong': set(strong),
    }


def get_killed():
    t_wait_for_forks()
    results = t_get_killed()
    return {
        'strong': set(results['strong']),
    }


#################################################
# shadow tests

@pytest.mark.parametrize("mode", MODES)
def test_reinit_t_assert(mode):
    for ii in range(1, 4):
        reinit(execution_mode=mode, no_atexit=True)
        tainted_int = t_combine({0: 0, ii: 1})

        t_assert(tainted_int == 0)
        assert get_killed() == gen_killed({ii})


@pytest.mark.parametrize("mode", MODES)
def test_split_stream_single_if(mode):
    reinit(execution_mode=mode, no_atexit=True)
    @t_wrap
    def func(tainted_int):
        if t_cond(tainted_int == 0):
            tainted_int += 1
        else:
            tainted_int -= 1
        return tainted_int

    
    t_assert(func(t_combine({0: 0, 1: 1})) == 1)
    assert get_killed() == gen_killed([1])


@pytest.mark.parametrize("mode", MODES)
def test_split_stream_double_if(mode):
    reinit(execution_mode=mode, no_atexit=True)
    @t_wrap
    def func(tainted_int):
        if t_cond(tainted_int <= 1):
            tainted_int += 1
            # 0: 1, 1: 2
            if t_cond(tainted_int == 1):
                tainted_int -= 1
                # 0: 0
            else:
                tainted_int += 1
                # 1: 3
        else:
            tainted_int -= 1
            # 2: 1, 3: 2
            if t_cond(tainted_int == 1):
                tainted_int -= 1
                # 2: 0
            else:
                tainted_int += 1
                # 3: 3
        return tainted_int

    t_assert(func(t_combine({0: 0, 1: 1, 2: 2, 3: 3})) == 0)
    t_get_killed()
    assert get_killed() == gen_killed([1, 3])


@pytest.mark.parametrize("mode", MODES)
def test_split_stream_nested_if_call(mode):
    reinit(execution_mode=mode, no_atexit=True)
    @t_wrap
    def inner(tainted_int):
        if t_cond(tainted_int == 1):
            tainted_int -= 1
        else:
            tainted_int += 1
        return tainted_int

    @t_wrap
    def func(tainted_int):
        if t_cond(tainted_int <= 1):
            tainted_int += 1
            tainted_int = inner(tainted_int)
        else:
            tainted_int -= 1
            tainted_int = inner(tainted_int)
        return tainted_int

    t_assert(func(t_combine({0: 0, 1: 1, 2: 2, 3: 3})) == 0)
    t_get_killed()
    assert get_killed() == gen_killed([1, 3])


@pytest.mark.parametrize("mode", MODES)
def test_wrap(mode):
    @t_wrap
    def simple(a, b):
        if t_cond(a == 1):
            return b + 1
        elif t_cond(t_combine({0: a == 2, 10: a >= 2})):
            return a + b
        else:
            return 0

    # reinit(execution_mode=mode, no_atexit=True)
    # assert simple(0, 1) == 0

    reinit(execution_mode=mode, no_atexit=True)
    t_assert(simple(t_combine({0: 0, 1: 1}), 0) == 0)
    assert get_killed() == gen_killed([1])

    reinit(execution_mode=mode, no_atexit=True)
    t_assert(simple(1, t_combine({0: 0, 1: 1})) == 1)
    assert get_killed() == gen_killed([1])

    reinit(execution_mode=mode, no_atexit=True)
    t_assert(simple(t_combine({0: 0, 1: 1}), t_combine({0: 0, 2: 1})) == 0)
    assert get_killed() == gen_killed([1])

    reinit(execution_mode=mode, no_atexit=True)
    t_assert(simple(2, t_combine({0: 1, 1: 2})) == 3)
    assert get_killed() == gen_killed([1])

    reinit(execution_mode=mode, no_atexit=True)
    t_assert(simple(3, 1) == 0)
    assert get_killed() == gen_killed([10])

    reinit(execution_mode=mode, no_atexit=True)
    t_assert(simple(t_combine({0: 3, 1: 1}), 1) == 0)
    assert get_killed() == gen_killed([1, 10])


@pytest.mark.parametrize("mode", MODES)
def test_recursive(mode):
    @t_wrap
    def rec(a):
        if t_cond(t_combine({0: a == 0, 1: a <= 2})):
            return 0

        res = rec(a - 1)
        res = res + a

        return res

    reinit(execution_mode=mode, no_atexit=True)
    t_assert(rec(t_combine({0: 5, 2: 6})) == 15)
    assert get_killed() == gen_killed([1, 2])


@pytest.mark.parametrize("mode", MODES)
def test_control_flow_return(mode):
    @t_wrap
    def fun(a):
        if t_cond(t_combine({0: a == 0, 1: a != 0})):
            return 0
        else:
            return 1

    reinit(execution_mode=mode, no_atexit=True)
    t_assert(fun(0) == 0)
    assert get_killed() == gen_killed([1])


@pytest.mark.parametrize("mode", MODES)
def test_non_mainline_divergence(mode):
    @t_wrap
    def fun(a):
        if t_cond(t_combine({0: a < 5, 1: a > 5})):
            return 0
        else:
            if t_cond(t_combine({0: a == 3, 2: a != 3})):
                return 1
            else:
                return 2

    reinit(execution_mode=mode, no_atexit=True)
    t_assert(fun(3) == 0)
    assert get_killed() == gen_killed([1])


@pytest.mark.parametrize("mode", MODES)
def test_return_influences_control_flow(mode):
    @t_wrap
    def fun_inner(a):
        # a = 0
        if t_cond(t_combine({0: a == 0, 1: a != 0})):
            # 0: return 1
            return 1
        else:
            # 1: return 0
            return 0

    @t_wrap
    def fun(a):
        res = fun_inner(a)
        logger.debug(f"{res}")
        # 0: 1, 1: 0
        if t_cond(t_combine({0: res == 1, 2: res != 1})):
            # 0: 1 -> return 0
            return 0
        else:
            # 1: 0, 2: 1 != 1 -> return 1
            return 1

    reinit(execution_mode=mode, no_atexit=True)
    res = fun(0)
    logger.debug(f"{res}")
    t_assert(res == 0)
    assert get_killed() == gen_killed([1, 2])


@pytest.mark.parametrize("mode", MODES)
def test_recursive_return_influences_control_flow(mode):

    @t_wrap
    def fun(a):
        a = t_combine({0: a - 1, 1: a - 2})
        if t_cond(t_combine({0: a <= 0, 2: a <= -5})):
            return a
        else:
            return a + fun(a)

    reinit(execution_mode=mode, no_atexit=True)
    res = fun(5)
    logger.debug(f"{res}")
    t_assert(res == 10)
    assert get_killed() == gen_killed([1, 2])


@pytest.mark.parametrize("mode", MODES)
def test_control_flow_based_on_called_function(mode):

    @t_wrap
    def inner(a):
        a = t_combine({0: a % 2, 2: a % 3})
        if t_cond(t_combine({0: a >= 1, 1: a != 0})):
            return 1
        else:
            return 0

    @t_wrap
    def fun(a):
        a = inner(a) - 1
        if t_cond(t_combine({0: a == 0, 3: a != 1})):
            return 1
        a = inner(a) - 1
        if t_cond(t_combine({0: a == 0, 4: a != 1})):
            return 2
        
        return 0

    reinit(execution_mode=mode, no_atexit=True)
    res = fun(3)
    logger.debug(f"{res}")
    t_assert(res == 1)
    assert get_killed() == gen_killed([2])


#################################################
# tests for tuple
# ['__add__', '__class__', '__class_getitem__', '__contains__', '__delattr__', '__dir__',
# '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__',
# '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__',
# '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
# '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__',
# 'count', 'index']


@pytest.mark.skip(reason="not implemented: need to update t_tuple")
@pytest.mark.parametrize("mode", MODES)
def test_tuple_eq_with_tint_elem(mode):
    reinit(execution_mode=mode, no_atexit=True)
    tainted_int = t_combine({0: 0, 1: 1})
    data = t_tuple((1, 2, 3, tainted_int))

    t_assert(data == (1, 2, 3, 0))
    assert get_killed() == gen_killed({1: True})

    reinit(execution_mode=mode, no_atexit=True)
    t_assert((1, 2, 3, 0) == data)
    assert get_killed() == gen_killed({1: True})




#################################################
# tests for list
# ['__add__', '__class__', '__class_getitem__', '__contains__', '__delattr__',
# '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
# '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__',
# '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__',
# '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
# '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__',
# '__str__', '__subclasshook__',
# 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']

@pytest.mark.skip(reason="not implemented: need to hook equal of list")
@pytest.mark.parametrize("mode", MODES)
def test_list_eq_with_tint_elem(mode):
    reinit(execution_mode=mode, no_atexit=True)
    tainted_int = t_combine({0: 0, 1: 1})
    data = [1, 2, 3, tainted_int]

    t_assert(data == [1, 2, 3, 0])
    assert get_killed()[0] == {1: True}


@pytest.mark.skip(reason="not implemented: list len dependent on tainted int")
@pytest.mark.parametrize("mode", MODES)
def test_list_mul_tint(mode):
    data = []
    tainted_int = t_combine({0: 0, 1: 1})

    # create a list where the length is dependent on the tainted int
    new_data = [1]*tainted_int
    data.extend(new_data)

    # this should cause a weakly killed
    # for val in data:
    #     assert val == 1

    # this should also cause a weakly killed
    result = sum(data)
    assert result == 0


@pytest.mark.skip(reason="not implemented: similar problems for pop, remove, index")
@pytest.mark.parametrize("mode", MODES)
def test_list_insert_tint(mode):
    data = [1, 2, 3]
    tainted_int = t_combine({0: 0, 1: 1})

    # insert data at pos defined by tainted int
    data.insert(tainted_int, 'a')

    for val in data:
        print(val)

    # TODO need to taint values at the involved positions
    # data[0] does not necessarily need to be tainted
    # data[1] does to be tainted
    assert data[0] == 'a'
    assert data[1] == 1




#################################################
# tests for set
# ['__and__', '__class__', '__class_getitem__', '__contains__', '__delattr__', '__dir__',
# '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__',
# '__iand__', '__init__', '__init_subclass__', '__ior__', '__isub__', '__iter__', '__ixor__',
# '__le__', '__len__', '__lt__', '__ne__', '__new__', '__or__', '__rand__', '__reduce__',
# '__reduce_ex__', '__repr__', '__ror__', '__rsub__', '__rxor__', '__setattr__', '__sizeof__',
# '__str__', '__sub__', '__subclasshook__', '__xor__',
# 'add', 'clear', 'copy', 'difference', 'difference_update', 'discard', 'intersection',
# 'intersection_update', 'isdisjoint', 'issubset', 'issuperset', 'pop', 'remove',
# 'symmetric_difference', 'symmetric_difference_update', 'union', 'update']




#################################################
# tests for dict
# ['__class__', '__class_getitem__', '__contains__', '__delattr__', '__delitem__', '__dir__',
# '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__',
# '__hash__', '__init__', '__init_subclass__', '__ior__', '__iter__', '__le__', '__len__',
# '__lt__', '__ne__', '__new__', '__or__', '__reduce__', '__reduce_ex__', '__repr__',
# '__reversed__', '__ror__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__',
# 'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values']

@pytest.mark.skip(reason="not implemented: tint not hashable")
@pytest.mark.parametrize("mode", MODES)
def test_dict_key_tainted(mode):
    data = {}
    tainted_int = t_combine({'0': 0, '1.1': 1})

    # tainted int is not hashable
    data[tainted_int] = 1

    # overwrite mainline value?
    # data[0] = 2

    assert data[tainted_int] == 1
