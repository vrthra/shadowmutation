import pytest

from shadow import reinit, t_get_killed, t_int, t_tuple, t_list, t_assert, t_cond


def gen_killed(strong, weak):
    return {
        'strong': strong,
        'weak': weak,
    }


#################################################
# shadow tests

def test_reinit_t_assert():
    for ii in range(3):
        reinit()
        tainted_int = t_int({'0': 0, f'{ii}.1': 1})

        t_assert(tainted_int == 0)
        assert t_get_killed() == gen_killed({f'{ii}.1': True}, {})




#################################################
# tests for tuple
# ['__add__', '__class__', '__class_getitem__', '__contains__', '__delattr__', '__dir__',
# '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__',
# '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__',
# '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
# '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__',
# 'count', 'index']

# @pytest.mark.skip(reason="not implemented: need to hook equal of tuple")
def test_tuple_eq_with_tint_elem():
    reinit()
    tainted_int = t_int({'0': 0, '1.1': 1})
    data = t_tuple((1, 2, 3, tainted_int))

    t_assert(data == (1, 2, 3, 0))
    assert t_get_killed() == gen_killed({'1.1': True}, {})

    reinit()
    t_assert((1, 2, 3, 0) == data)
    assert t_get_killed() == gen_killed({'1.1': True}, {})




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
def test_list_eq_with_tint_elem():
    reinit()
    tainted_int = t_int({'0': 0, '1.1': 1})
    data = [1, 2, 3, tainted_int]

    t_assert(data == [1, 2, 3, 0])
    assert t_get_killed()[0] == {'1.1': True}


@pytest.mark.skip(reason="not implemented: list len dependent on tainted int")
def test_list_mul_tint():
    data = []
    tainted_int = t_int({'0': 0, '1.1': 1})

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
def test_list_insert_tint():
    data = [1, 2, 3]
    tainted_int = t_int({'0': 0, '1.1': 1})

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
def test_dict_key_tainted():
    data = {}
    tainted_int = t_int({'0': 0, '1.1': 1})

    # tainted int is not hashable
    data[tainted_int] = 1

    # overwrite mainline value?
    # data[0] = 2

    assert data[tainted_int] == 1
