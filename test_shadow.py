from collections import defaultdict
from typing import Dict, List, Optional, Union
import os
from random import randint
import pytest
import logging
logger = logging.getLogger(__name__)

from shadow import reinit, t_final_exception_test, t_wrap, t_combine, t_wait_for_forks, t_get_killed, t_cond, t_assert, \
                   t_logical_path, t_seen_mutants, t_masked_mutants, ShadowVariable, t_class, t_active_mutants, t_sv
from lib.fork import get_forking_context

MODES = ['shadow', 'shadow_cache', 'shadow_fork_child', 'shadow_fork_parent', 'shadow_fork_cache'] # , 'shadow_fork', 'shadow_cache', 'shadow_fork_cache']
SPLIT_STREAM_MODES = ['split', 'modulo']

if os.environ.get("TEST_SKIP_MODES") is not None:
    modes_to_skip = os.environ.get("TEST_SKIP_MODES").split(',')
    logger.info(f"Skipping modes: {modes_to_skip}")
    for mts in modes_to_skip:
        if mts in MODES:
            MODES.remove(mts)
        elif mts in SPLIT_STREAM_MODES:
            SPLIT_STREAM_MODES.remove(mts)
        else:
            logger.warning(f"Unknown mode can't skip: {mts}")



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
    @t_wrap
    def func(tainted_int):
        if t_cond(tainted_int == 0):
            tainted_int += 1
        else:
            tainted_int -= 1
        return tainted_int

    
    reinit(execution_mode=mode, no_atexit=True)
    res = func(t_combine({0: 0, 1: 1}))
    t_assert(res == 1)

    assert get_killed() == gen_killed([1])


@pytest.mark.parametrize("mode", MODES)
def test_split_stream_double_if(mode):
    @t_wrap
    def func(tainted_int):
        if t_cond(tainted_int <= 1):
            tainted_int += 1
            # 0: 1, 1: 2, 4: 1, 5: 2
            if t_cond(tainted_int == 1):
                tainted_int -= 1
                # 0: 0, 4: 0
            else:
                tainted_int += 1
                # 1: 3, 5: 3
        else:
            tainted_int -= 1
            # 2: 1, 3: 2, 6: 1, 7: 2
            if t_cond(tainted_int == 1):
                tainted_int -= 1
                # 2: 0, 6: 0
            else:
                tainted_int += 1
                # 3: 3, 7: 3
        return tainted_int

    reinit(execution_mode=mode, no_atexit=True)
    res = func(t_combine({0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3}))
    t_assert(res == 0)
    assert get_killed() == gen_killed([1, 3, 5, 7])


@pytest.mark.parametrize("mode", MODES)
def test_split_stream_nested_if_call(mode):
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

    reinit(execution_mode=mode, no_atexit=True)
    t_assert(func(t_combine({0: 0, 1: 1, 2: 2, 3: 3})) == 0)
    assert get_killed() == gen_killed([1, 3])


@pytest.mark.parametrize("mode", MODES)
def test_wrap(mode):
    @t_wrap
    def simple(a, b):
        if t_cond(t_combine({0: a == 1})):
            return b + 1
        elif t_cond(t_combine({0: a == 2, 10: a >= 2})):
            return a + b
        else:
            return 0

    def do_test(a, b, res):
        act_res = simple(a, b)
        t_assert(act_res == res)

    # reinit(execution_mode=mode, no_atexit=True)
    # assert simple(0, 1) == 0

    reinit(execution_mode=mode, no_atexit=True)
    do_test(t_combine({0: 0, 1: 1}), 0, 0)
    assert get_killed() == gen_killed([1])

    reinit(execution_mode=mode, no_atexit=True)
    do_test(1, t_combine({0: 0, 1: 1}), 1)
    assert get_killed() == gen_killed([1])

    reinit(execution_mode=mode, no_atexit=True)
    do_test(t_combine({0: 0, 1: 1}), t_combine({0: 0, 2: 1}), 0)
    assert get_killed() == gen_killed([1])

    reinit(execution_mode=mode, no_atexit=True)
    do_test(2, t_combine({0: 1, 1: 2}), 3)
    assert get_killed() == gen_killed([1])

    reinit(execution_mode=mode, no_atexit=True)
    do_test(3, 1, 0)
    assert get_killed() == gen_killed([10])

    reinit(execution_mode=mode, no_atexit=True)
    do_test(t_combine({0: 3, 1: 1}), 1, 0)
    assert get_killed() == gen_killed([1, 10])


@pytest.mark.parametrize("mode", MODES)
def test_recursive(mode):
    @t_wrap
    def rec(a):
        if t_cond(t_combine({0: a <= 0, 1: a <= 2})):
            return 0

        res = rec(a - 1)
        res = res + a

        return res

    reinit(execution_mode=mode, no_atexit=True)
    res = rec(t_combine({0: 5, 2: 6}))
    t_assert(res == 15)
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
    res = fun(3)
    t_assert(res == 0)
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
        # 0: 1, 1: 0
        if t_cond(t_combine({0: res == 1, 2: res != 1})):
            # 0: 1 -> return 0
            return 0
        else:
            # 1: 0, 2: 1 != 1 -> return 1
            return 1

    reinit(execution_mode=mode, no_atexit=True)
    res = fun(0)
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
    t_assert(res == 1)
    assert get_killed() == gen_killed([2])


#################################################
# real-world tests

@pytest.mark.parametrize("mode", MODES)
def test_approx_exp(mode):
    @t_wrap
    def compute_exp(x: int, accuracy: int):
        extra_precision = 4
        accuracy_scaler = t_combine({(0): lambda : 10 ** accuracy, (1): lambda : 10 + accuracy, (2): lambda : 10 - accuracy, (3): lambda : 10 * accuracy, (4): lambda : 10 / accuracy, (5): lambda : 10 % accuracy, (6): lambda : 10 << accuracy, (7): lambda : 10 >> accuracy, (8): lambda : 10 | accuracy, (9): lambda : 10 ^ accuracy, (10): lambda : 10 & accuracy, (11): lambda : 10 // accuracy})
        extra_scaler = t_combine({(0): lambda : 10 ** extra_precision, (12): lambda : 10 + extra_precision, (13): lambda : 10 - extra_precision, (14): lambda : 10 * extra_precision, (16): lambda : 10 % extra_precision, (17): lambda : 10 << extra_precision, (18): lambda : 10 >> extra_precision, (19): lambda : 10 | extra_precision, (20): lambda : 10 ^ extra_precision, (21): lambda : 10 & extra_precision, (22): lambda : 10 // extra_precision})
        full_scaler = t_combine({(0): lambda : accuracy_scaler * extra_scaler, (23): lambda : accuracy_scaler + extra_scaler, (24): lambda : accuracy_scaler - extra_scaler, (25): lambda : accuracy_scaler / extra_scaler, (26): lambda : accuracy_scaler % extra_scaler, (27): lambda : accuracy_scaler << extra_scaler, (28): lambda : accuracy_scaler >> extra_scaler, (29): lambda : accuracy_scaler | extra_scaler, (30): lambda : accuracy_scaler ^ extra_scaler, (31): lambda : accuracy_scaler & extra_scaler, (32): lambda : accuracy_scaler // extra_scaler})
        sum_low = 0
        sum_high = 0
        term_low = full_scaler
        term_high = full_scaler
        floor_x = t_combine({(0): lambda : x // accuracy_scaler, (33): lambda : x + accuracy_scaler, (34): lambda : x - accuracy_scaler, (35): lambda : x * accuracy_scaler, (36): lambda : x / accuracy_scaler, (37): lambda : x % accuracy_scaler, (38): lambda : x << accuracy_scaler, (39): lambda : x >> accuracy_scaler, (40): lambda : x | accuracy_scaler, (41): lambda : x ^ accuracy_scaler, (42): lambda : x & accuracy_scaler})
        i = 0
        itr = 0
        while True:
            if t_cond(t_combine({(0): lambda : term_low <= 0, (43): lambda : term_low == 0, (44): lambda : term_low != 0, (45): lambda : term_low < 0, (46): lambda : term_low > 0, (47): lambda : term_low >= 0})):
                break
            if t_cond(t_combine({(0): lambda : itr >= 100, (48): lambda : itr == 100, (49): lambda : itr != 100, (50): lambda : itr < 100, (51): lambda : itr <= 100, (52): lambda : itr > 100})):
                break
            sum_low = t_combine({(0): lambda : sum_low + term_low, (53): lambda : sum_low - term_low, (54): lambda : sum_low * term_low, (55): lambda : sum_low / term_low, (56): lambda : sum_low % term_low, (57): lambda : sum_low << term_low, (58): lambda : sum_low >> term_low, (59): lambda : sum_low | term_low, (60): lambda : sum_low ^ term_low, (61): lambda : sum_low & term_low, (62): lambda : sum_low // term_low})
            sum_high = t_combine({(0): lambda : sum_high + term_high, (63): lambda : sum_high - term_high, (64): lambda : sum_high * term_high, (65): lambda : sum_high / term_high, (66): lambda : sum_high % term_high, (67): lambda : sum_high << term_high, (68): lambda : sum_high >> term_high, (69): lambda : sum_high | term_high, (70): lambda : sum_high ^ term_high, (71): lambda : sum_high & term_high, (72): lambda : sum_high // term_high})
            term_low = t_combine({(0): lambda : term_low * x, (73): lambda : term_low + x, (74): lambda : term_low - x, (75): lambda : term_low / x, (76): lambda : term_low % x, (77): lambda : term_low << x, (78): lambda : term_low >> x, (79): lambda : term_low | x, (80): lambda : term_low ^ x, (81): lambda : term_low & x, (82): lambda : term_low // x})
            term_low = t_combine({(0): lambda : term_low // accuracy_scaler, (83): lambda : term_low + accuracy_scaler, (84): lambda : term_low - accuracy_scaler, (85): lambda : term_low * accuracy_scaler, (86): lambda : term_low / accuracy_scaler, (87): lambda : term_low % accuracy_scaler, (88): lambda : term_low << accuracy_scaler, (89): lambda : term_low >> accuracy_scaler, (90): lambda : term_low | accuracy_scaler, (91): lambda : term_low ^ accuracy_scaler, (92): lambda : term_low & accuracy_scaler})
            high_accuracy_scaler = t_combine({(0): lambda : accuracy_scaler + 1, (93): lambda : accuracy_scaler - 1, (94): lambda : accuracy_scaler * 1, (95): lambda : accuracy_scaler / 1, (96): lambda : accuracy_scaler % 1, (97): lambda : accuracy_scaler << 1, (98): lambda : accuracy_scaler >> 1, (99): lambda : accuracy_scaler | 1, (100): lambda : accuracy_scaler ^ 1, (101): lambda : accuracy_scaler & 1, (102): lambda : accuracy_scaler // 1})
            term_high = t_combine({(0): lambda : term_high * x, (103): lambda : term_high + x, (104): lambda : term_high - x, (105): lambda : term_high / x, (106): lambda : term_high % x, (107): lambda : term_high << x, (108): lambda : term_high >> x, (109): lambda : term_high | x, (110): lambda : term_high ^ x, (111): lambda : term_high & x, (112): lambda : term_high // x})
            term_high = t_combine({(0): lambda : term_high // high_accuracy_scaler, (113): lambda : term_high + high_accuracy_scaler, (114): lambda : term_high - high_accuracy_scaler, (115): lambda : term_high * high_accuracy_scaler, (116): lambda : term_high / high_accuracy_scaler, (117): lambda : term_high % high_accuracy_scaler, (118): lambda : term_high << high_accuracy_scaler, (119): lambda : term_high >> high_accuracy_scaler, (120): lambda : term_high | high_accuracy_scaler, (121): lambda : term_high ^ high_accuracy_scaler, (122): lambda : term_high & high_accuracy_scaler})
            if t_cond(t_combine({(0): lambda : i > floor_x, (123): lambda : i == floor_x, (124): lambda : i != floor_x, (125): lambda : i < floor_x, (126): lambda : i <= floor_x, (127): lambda : i >= floor_x})):
                if t_cond(t_combine({(0): lambda : term_high < extra_scaler, (128): lambda : term_high == extra_scaler, (129): lambda : term_high != extra_scaler, (130): lambda : term_high <= extra_scaler, (131): lambda : term_high > extra_scaler, (132): lambda : term_high >= extra_scaler})):
                    sum_upper_bound = t_combine({(0): lambda : sum_high + term_high, (133): lambda : sum_high - term_high, (134): lambda : sum_high * term_high, (135): lambda : sum_high / term_high, (136): lambda : sum_high % term_high, (137): lambda : sum_high << term_high, (138): lambda : sum_high >> term_high, (139): lambda : sum_high | term_high, (140): lambda : sum_high ^ term_high, (141): lambda : sum_high & term_high, (142): lambda : sum_high // term_high})
                    temp: int = t_combine({(0): lambda : sum_low // extra_scaler, (143): lambda : sum_low + extra_scaler, (144): lambda : sum_low - extra_scaler, (145): lambda : sum_low * extra_scaler, (146): lambda : sum_low / extra_scaler, (147): lambda : sum_low % extra_scaler, (148): lambda : sum_low << extra_scaler, (149): lambda : sum_low >> extra_scaler, (150): lambda : sum_low | extra_scaler, (151): lambda : sum_low ^ extra_scaler, (152): lambda : sum_low & extra_scaler})
                    temp = round(temp)
                    upper: int = t_combine({(0): lambda : sum_upper_bound // extra_scaler, (153): lambda : sum_upper_bound + extra_scaler, (154): lambda : sum_upper_bound - extra_scaler, (155): lambda : sum_upper_bound * extra_scaler, (156): lambda : sum_upper_bound / extra_scaler, (157): lambda : sum_upper_bound % extra_scaler, (158): lambda : sum_upper_bound << extra_scaler, (159): lambda : sum_upper_bound >> extra_scaler, (160): lambda : sum_upper_bound | extra_scaler, (161): lambda : sum_upper_bound ^ extra_scaler, (162): lambda : sum_upper_bound & extra_scaler})
                    upper = round(upper)
                    if t_cond(t_combine({(0): lambda : upper == temp, (163): lambda : upper != temp, (164): lambda : upper < temp, (165): lambda : upper <= temp, (166): lambda : upper > temp, (167): lambda : upper >= temp})):
                        return temp
            i = t_combine({(0): lambda : i + 1, (168): lambda : i - 1, (169): lambda : i * 1, (171): lambda : i % 1, (172): lambda : i << 1, (173): lambda : i >> 1, (174): lambda : i | 1, (175): lambda : i ^ 1, (176): lambda : i & 1, (177): lambda : i // 1})
            term_low = t_combine({(0): lambda : term_low // i, (178): lambda : term_low + i, (179): lambda : term_low - i, (180): lambda : term_low * i, (181): lambda : term_low / i, (182): lambda : term_low % i, (183): lambda : term_low << i, (184): lambda : term_low >> i, (185): lambda : term_low | i, (186): lambda : term_low ^ i, (187): lambda : term_low & i})
            term_high = t_combine({(0): lambda : term_high // i, (188): lambda : term_high + i, (189): lambda : term_high - i, (190): lambda : term_high * i, (191): lambda : term_high / i, (192): lambda : term_high % i, (193): lambda : term_high << i, (194): lambda : term_high >> i, (195): lambda : term_high | i, (196): lambda : term_high ^ i, (197): lambda : term_high & i})
            term_high = t_combine({(0): lambda : term_high + 1, (198): lambda : term_high - 1, (199): lambda : term_high * 1, (200): lambda : term_high / 1, (201): lambda : term_high % 1, (202): lambda : term_high << 1, (203): lambda : term_high >> 1, (204): lambda : term_high | 1, (205): lambda : term_high ^ 1, (206): lambda : term_high & 1, (207): lambda : term_high // 1})
            itr = t_combine({(0): lambda : itr + 1, (208): lambda : itr - 1, (209): lambda : itr * 1, (211): lambda : itr % 1, (212): lambda : itr << 1, (213): lambda : itr >> 1, (214): lambda : itr | 1, (215): lambda : itr ^ 1, (216): lambda : itr & 1, (217): lambda : itr // 1})
        return None

    reinit(execution_mode=mode, no_atexit=True)
    res = compute_exp(3, 2)
    t_assert(res == 103)
    assert get_killed() == gen_killed([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 38, 40, 41, 44, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 96, 97, 98, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 122, 123, 125, 126, 128, 131, 132, 134, 135, 136, 141, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 164, 166, 168, 169, 171, 172, 173, 176, 177, 182, 183, 184, 187, 192, 193, 194, 197, 201, 202, 203, 206])


@pytest.mark.parametrize("mode", MODES)
def test_entropy(mode):
    @t_wrap
    def ln(val: float) ->float:
        c_log = t_combine({(0): lambda : 1 / 1000.0, (1): lambda : 1 + 1000.0, (2): lambda : 1 - 1000.0, (3): lambda : 1 * 1000.0, (4): lambda : 1 % 1000.0, (10): lambda : 1 // 1000.0})
        c_log = t_combine({(0): lambda : val ** c_log, (11): lambda : val + c_log, (12): lambda : val - c_log, (13): lambda : val * c_log, (14): lambda : val / c_log, (15): lambda : val % c_log, (21): lambda : val // c_log})
        c_log = t_combine({(0): lambda : c_log - 1, (22): lambda : c_log + 1, (23): lambda : c_log * 1, (24): lambda : c_log / 1, (25): lambda : c_log % 1, (31): lambda : c_log // 1})
        c_log = t_combine({(0): lambda : 1000.0 * c_log, (32): lambda : 1000.0 + c_log, (33): lambda : 1000.0 - c_log, (34): lambda : 1000.0 / c_log, (35): lambda : 1000.0 % c_log, (41): lambda : 1000.0 // c_log})
        return c_log


    @t_wrap
    def entropy(hist, l: int):
        c = t_combine({(0): lambda : hist[0] / l})
        c_log = ln(c)
        normalized = t_combine({(0): lambda : -c * c_log, (52): lambda : -c + c_log, (53): lambda : -c - c_log, (54): lambda : -c / c_log, (55): lambda : -c % c_log, (61): lambda : -c // c_log})
        res = normalized
        for v in hist[1:]:
            c = t_combine({(0): lambda : v / l, (62): lambda : v + l, (63): lambda : v - l, (64): lambda : v * l, (65): lambda : v % l, (66): lambda : v << l, (67): lambda : v >> l, (68): lambda : v | l, (69): lambda : v ^ l, (70): lambda : v & l, (71): lambda : v // l})
            c_log = ln(c)
            normalized = t_combine({(0): lambda : -c * c_log, (72): lambda : -c + c_log, (73): lambda : -c - c_log, (74): lambda : -c / c_log, (75): lambda : -c % c_log, (81): lambda : -c // c_log})
            res = t_combine({(0): lambda : res + normalized, (82): lambda : res - normalized, (83): lambda : res * normalized, (84): lambda : res / normalized, (85): lambda : res % normalized, (91): lambda : res // normalized})
        return res

    reinit(execution_mode=mode, no_atexit=True)
    source = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    res = entropy(source, len(source))
    expected = 3.421568195457525
    diff = abs(res - expected)
    rounded_diff = round(diff, 8)
    t_assert(rounded_diff == 0)
    assert get_killed() == gen_killed([1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 52, 53, 54, 55, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 81, 82, 83, 84, 85, 91])


@pytest.mark.parametrize("mode", MODES)
def test_euler(mode):
    @t_wrap
    def is_negligible(val: float) ->bool:
        return t_combine({(0): lambda : round(val, 3) == 0, (1): lambda : round(val, 3) != 0, (2): lambda : round(val, 3) < 0, (3): lambda : round(val, 3) <= 0, (4): lambda : round(val, 3) > 0, (5): lambda : round(val, 3) >= 0})


    @t_wrap
    def newton_cooling(h: float, y: float) ->float:
        new_y = t_combine({(0): lambda : y - 20, (6): lambda : y + 20, (7): lambda : y * 20, (8): lambda : y / 20, (9): lambda : y % 20, (15): lambda : y // 20})
        update = t_combine({(0): lambda : -0.07 * new_y, (16): lambda : -0.07 + new_y, (17): lambda : -0.07 - new_y, (18): lambda : -0.07 / new_y, (19): lambda : -0.07 % new_y, (25): lambda : -0.07 // new_y})
        update_y = t_combine({(0): lambda : h * update, (26): lambda : h + update, (27): lambda : h - update, (28): lambda : h / update, (29): lambda : h % update, (35): lambda : h // update})
        return update_y


    @t_wrap
    def euler(y0: float, a: float, b: int, h: int) ->float:
        t = a
        y = y0
        while True:
            if t_cond(t_combine({(0): lambda : t > b, (36): lambda : t == b, (37): lambda : t != b, (38): lambda : t < b, (39): lambda : t <= b, (40): lambda : t >= b})):
                break
            t = t_combine({(0): lambda : t + h, (41): lambda : t - h, (42): lambda : t * h, (43): lambda : t / h, (44): lambda : t % h, (50): lambda : t // h})
            update_y = newton_cooling(h, y)
            if t_cond(is_negligible(update_y)):
                break
            y = t_combine({(0): lambda : y + update_y, (51): lambda : y - update_y, (52): lambda : y * update_y, (53): lambda : y / update_y, (54): lambda : y % update_y, (60): lambda : y // update_y})
        return y


    reinit(execution_mode=mode, no_atexit=True)
    res = euler(100, 0, 100, 10)
    res = round(res, 3)
    t_assert(res == 20)
    assert get_killed() == gen_killed([1, 2, 3, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 35, 37, 38, 39, 51, 52, 53, 54, 60])


@pytest.mark.parametrize("mode", MODES)
def test_newton(mode):
    from math import sqrt

    @t_wrap
    def newton_method(number: float, number_iters: int=100) ->float:
        a = float(number)
        i = 0
        while True:
            if t_cond(t_combine({(0): lambda : i >= number_iters, (1): lambda : i == number_iters, (2): lambda : i != number_iters, (3): lambda : i < number_iters, (4): lambda : i <= number_iters, (5): lambda : i > number_iters})):
                break
            new_number = t_combine({(0): lambda : 0.5 * t_combine({(0): lambda : number + t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (16): lambda : number - t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (17): lambda : number * t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (18): lambda : number / t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (19): lambda : number % t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (25): lambda : number // t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number})}), (26): lambda : 0.5 + t_combine({(0): lambda : number + t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (16): lambda : number - t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (17): lambda : number * t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (18): lambda : number / t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (19): lambda : number % t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (25): lambda : number // t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number})}), (27): lambda : 0.5 - t_combine({(0): lambda : number + t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (16): lambda : number - t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (17): lambda : number * t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (18): lambda : number / t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (19): lambda : number % t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (25): lambda : number // t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number})}), (28): lambda : 0.5 / t_combine({(0): lambda : number + t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (16): lambda : number - t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (17): lambda : number * t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (18): lambda : number / t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (19): lambda : number % t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (25): lambda : number // t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number})}), (29): lambda : 0.5 % t_combine({(0): lambda : number + t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (16): lambda : number - t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (17): lambda : number * t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (18): lambda : number / t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (19): lambda : number % t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (25): lambda : number // t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number})}), (35): lambda : 0.5 // t_combine({(0): lambda : number + t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (16): lambda : number - t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (17): lambda : number * t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (18): lambda : number / t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (19): lambda : number % t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number}), (25): lambda : number // t_combine({(0): lambda : a / number, (6): lambda : a + number, (7): lambda : a - number, (8): lambda : a * number, (9): lambda : a % number, (15): lambda : a // number})})})
            diff = round(t_combine({(0): lambda : new_number - number, (36): lambda : new_number + number, (37): lambda : new_number * number, (38): lambda : new_number / number, (39): lambda : new_number % number, (45): lambda : new_number // number}), 8)
            if t_cond(t_combine({(0): lambda : diff == 0, (46): lambda : diff != 0, (47): lambda : diff < 0, (48): lambda : diff <= 0, (49): lambda : diff > 0, (50): lambda : diff >= 0})):
                number = new_number
                break
            number = new_number
            i = t_combine({(0): lambda : i + 1, (51): lambda : i - 1, (52): lambda : i * 1, (54): lambda : i % 1, (55): lambda : i << 1, (56): lambda : i >> 1, (57): lambda : i | 1, (58): lambda : i ^ 1, (59): lambda : i & 1, (60): lambda : i // 1})
        return number

    reinit(execution_mode=mode, no_atexit=True)
    val = 10
    number_iters = 10
    newt = newton_method(val, number_iters=number_iters)
    pyth = sqrt(val)
    diff = abs(newt - pyth)
    rounded_diff = round(diff, 8)
    t_assert(rounded_diff == 0)
    assert get_killed() == gen_killed([2, 3, 4, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 35, 45, 46, 47, 48])


@pytest.mark.parametrize("mode", MODES)
def test_prime(mode):
    @t_wrap
    def check_max_runtime(n: int) ->bool:
        if t_cond(t_combine({(0): lambda : n >= 100, (1): lambda : n == 100, (2): lambda : n != 100, (3): lambda : n < 100, (4): lambda : n <= 100, (5): lambda : n > 100})):
            return True
        return False


    @t_wrap
    def prime(input: int) ->bool:
        """
        >>> skips = ['2 : swap 0']
        >>> prime(3)
        True
        """
        if t_cond(t_combine({(0): lambda : input == 0, (6): lambda : input != 0, (7): lambda : input < 0, (8): lambda : input <= 0, (9): lambda : input > 0, (10): lambda : input >= 0})):
            return False
        if t_cond(t_combine({(0): lambda : input == 1, (11): lambda : input != 1, (12): lambda : input < 1, (13): lambda : input <= 1, (14): lambda : input > 1, (15): lambda : input >= 1})):
            return False
        if t_cond(t_combine({(0): lambda : input == 2, (16): lambda : input != 2, (17): lambda : input < 2, (18): lambda : input <= 2, (19): lambda : input > 2, (20): lambda : input >= 2})):
            return True
        ctr = 0
        n = 2
        while True:
            if t_cond(t_combine({(0): lambda : n >= input, (21): lambda : n == input, (22): lambda : n != input, (23): lambda : n < input, (24): lambda : n <= input, (25): lambda : n > input})):
                break
            modulo = t_combine({(0): lambda : input % n, (26): lambda : input + n, (27): lambda : input - n, (28): lambda : input * n, (29): lambda : input / n, (30): lambda : input << n, (31): lambda : input >> n, (32): lambda : input | n, (33): lambda : input ^ n, (34): lambda : input & n, (35): lambda : input // n})
            if t_cond(t_combine({(0): lambda : modulo == 0, (36): lambda : modulo != 0, (37): lambda : modulo < 0, (38): lambda : modulo <= 0, (39): lambda : modulo > 0, (40): lambda : modulo >= 0})):
                return False
            n = t_combine({(0): lambda : n + 1, (41): lambda : n - 1, (42): lambda : n * 1, (44): lambda : n % 1, (45): lambda : n << 1, (46): lambda : n >> 1, (47): lambda : n | 1, (48): lambda : n ^ 1, (49): lambda : n & 1, (50): lambda : n // 1})
            if t_cond(check_max_runtime(ctr)):
                break
            ctr = t_combine({(0): lambda : ctr + 1, (51): lambda : ctr - 1, (52): lambda : ctr * 1, (54): lambda : ctr % 1, (55): lambda : ctr << 1, (56): lambda : ctr >> 1, (57): lambda : ctr | 1, (58): lambda : ctr ^ 1, (59): lambda : ctr & 1, (60): lambda : ctr // 1})
        return True

    reinit(execution_mode=mode, no_atexit=True)
    t_assert(prime(5) == True)
    assert get_killed() == gen_killed([6, 9, 10, 11, 14, 15, 25, 31, 34, 36, 39, 40, 41, 44, 46, 49])


@t_class
class BankAccount:
    balance: int

    def __init__(self, initial_balance: int):
        self.balance = initial_balance

    def is_overdrawn(self) ->None:
        if t_cond(t_combine({(0): lambda : self.balance >= 100, (1): lambda : self.balance == 100, (2): lambda : self.balance != 100, (3): lambda : self.balance < 100, (4): lambda : self.balance <= 100, (5): lambda : self.balance > 100})):
            return False
        elif t_cond(t_combine({(0): lambda : self.balance >= 0, (6): lambda : self.balance == 0, (7): lambda : self.balance != 0, (8): lambda : self.balance < 0, (9): lambda : self.balance <= 0, (10): lambda : self.balance > 0})):
            return False
        elif t_cond(t_combine({(0): lambda : self.balance < -100, (11): lambda : self.balance == -100, (12): lambda : self.balance != -100, (13): lambda : self.balance <= -100, (14): lambda : self.balance > -100, (15): lambda : self.balance >= -100})):
            return True
        else:
            return True

    def deposit(self, amount: int) ->None:
        self.balance += amount

    def withdraw(self, amount: int) ->None:
        self.balance -= amount


@pytest.mark.parametrize("mode", MODES)
def test_bank_accounts(mode):

    @t_wrap
    def fun() -> None:
        my_account = BankAccount(10)
        t_assert(my_account.balance == 10)
        t_assert(my_account.is_overdrawn() == False)

        my_account.deposit(5)
        t_assert(my_account.balance == 15)
        t_assert(my_account.is_overdrawn() == False)

        my_account.withdraw(200)
        t_assert(my_account.balance == -185)
        t_assert(my_account.is_overdrawn() == True)


    reinit(execution_mode=mode, no_atexit=True)
    fun()
    assert get_killed() == gen_killed([2, 3, 4, 6, 7, 8, 9])


@t_class
class BankAccountMut:
    balance: int
    overdrawn: bool

    def __init__(self, initial_balance: int):
        self.balance = initial_balance
        self.overdrawn = False
        self.update_overdrawn()

    def update_overdrawn(self) ->None:
        if t_cond(t_combine({(0): lambda : self.balance >= 100, (1): lambda : self.balance == 100, (2): lambda : self.balance != 100, (3): lambda : self.balance < 100, (4): lambda : self.balance <= 100, (5): lambda : self.balance > 100})):
            print('not overdrawn')
            self.overdrawn = False
        elif t_cond(t_combine({(0): lambda : self.balance >= 0, (6): lambda : self.balance == 0, (7): lambda : self.balance != 0, (8): lambda : self.balance < 0, (9): lambda : self.balance <= 0, (10): lambda : self.balance > 0})):
            print('all good')
            self.overdrawn = False
        elif t_cond(t_combine({(0): lambda : self.balance < -100, (11): lambda : self.balance == -100, (12): lambda : self.balance != -100, (13): lambda : self.balance <= -100, (14): lambda : self.balance > -100, (15): lambda : self.balance >= -100})):
            print('very overdrawn')
            self.overdrawn = True
        else:
            self.overdrawn = True

    def deposit(self, amount: int) ->None:
        self.balance += amount
        self.update_overdrawn()

    def withdraw(self, amount: int) ->None:
        self.balance -= amount
        self.update_overdrawn()

    def is_overdrawn(self) ->bool:
        return self.overdrawn


@pytest.mark.parametrize("mode", MODES)
def test_bank_accounts_mut(mode):

    @t_wrap
    def fun() -> None:
        my_account = BankAccountMut(10)
        t_assert(my_account.balance == 10)
        t_assert(my_account.overdrawn == False)

        my_account.deposit(5)
        t_assert(my_account.balance == 15)
        t_assert(my_account.overdrawn == False)

        my_account.withdraw(200)
        t_assert(my_account.balance == -185)
        t_assert(my_account.overdrawn == True)


    reinit(execution_mode=mode, no_atexit=True)
    fun()
    assert get_killed() == gen_killed([2, 3, 4, 6, 7, 8, 9])


@pytest.mark.parametrize("mode", MODES)
def test_fact(mode) ->None:
    inc = t_sv([4, 2, 4, 2, 4, 6, 2, 6])

    @t_wrap
    def div(val: int, divisor: int) ->bool:
        modded = t_combine({(0): lambda : val % divisor, (1): lambda : val + divisor, (2): lambda : val - divisor, (3): lambda : val * divisor, (4): lambda : val / divisor, (5): lambda : val << divisor, (6): lambda : val >> divisor, (7): lambda : val | divisor, (8): lambda : val ^ divisor, (9): lambda : val & divisor, (10): lambda : val // divisor})
        return t_combine({(0): lambda : modded != 0, (11): lambda : modded == 0, (13): lambda : modded <= 0, (14): lambda : modded > 0, (15): lambda : modded >= 0})

    @t_wrap
    def factorize(n: int) ->List[int]:
        factors = t_sv([])
        while True:
            is_div = div(n, 2)
            if t_cond(is_div):
                break
            factors.append(2)
            n = t_combine({(0): lambda : n // 2, (22): lambda : n >> 2})
        while True:
            is_div = div(n, 3)
            if t_cond(is_div):
                break
            factors.append(3)
            n = t_combine({(0): lambda : n // 3, (26): lambda : n + 3, (27): lambda : n - 3, (28): lambda : n * 3, (30): lambda : n % 3, (31): lambda : n << 3, (32): lambda : n >> 3, (33): lambda : n | 3, (34): lambda : n ^ 3, (35): lambda : n & 3})
        while True:
            is_div = div(n, 5)
            if t_cond(is_div):
                break
            factors.append(5)
            n = t_combine({(0): lambda : n // 5, (36): lambda : n + 5, (37): lambda : n - 5, (38): lambda : n * 5, (40): lambda : n % 5, (41): lambda : n << 5, (42): lambda : n >> 5, (43): lambda : n | 5, (44): lambda : n ^ 5, (45): lambda : n & 5})
        k = 7
        i = 0
        while True:
            k_squared = t_combine({(0): lambda : k * k, (46): lambda : k + k, (50): lambda : k << k, (52): lambda : k | k, (54): lambda : k & k})
            if t_cond(t_combine({(0): lambda : k_squared > n, (57): lambda : k_squared != n, (58): lambda : k_squared < n, (59): lambda : k_squared <= n, (60): lambda : k_squared >= n})):
                break
            is_div = div(n, k)
            n_mod_k = t_combine({(0): lambda : n % k, (61): lambda : n + k, (62): lambda : n - k, (63): lambda : n * k, (64): lambda : n / k, (65): lambda : n << k, (66): lambda : n >> k, (67): lambda : n | k, (68): lambda : n ^ k, (69): lambda : n & k, (70): lambda : n // k})
            if t_cond(t_combine({(0): lambda : n_mod_k == 0, (71): lambda : n_mod_k != 0, (72): lambda : n_mod_k < 0, (73): lambda : n_mod_k <= 0, (74): lambda : n_mod_k > 0, (75): lambda : n_mod_k >= 0})):
                factors.append(k)
                n = t_combine({(0): lambda : n // k, (76): lambda : n + k, (77): lambda : n - k, (78): lambda : n * k, (80): lambda : n % k, (81): lambda : n << k, (82): lambda : n >> k, (83): lambda : n | k, (84): lambda : n ^ k, (85): lambda : n & k})
            else:
                k = t_combine({(0): lambda : k + inc[i], (87): lambda : k * inc[i], (90): lambda : k << inc[i], (91): lambda : k >> inc[i], (94): lambda : k & inc[i]})
                if t_cond(t_combine({(0): lambda : i < 7, (96): lambda : i == 7, (97): lambda : i != 7, (98): lambda : i <= 7, (99): lambda : i > 7, (100): lambda : i >= 7})):
                    i = t_combine({(0): lambda : i + 1, (101): lambda : i - 1, (102): lambda : i * 1, (104): lambda : i % 1, (105): lambda : i << 1, (106): lambda : i >> 1, (107): lambda : i | 1, (108): lambda : i ^ 1, (109): lambda : i & 1, (110): lambda : i // 1})
                else:
                    i = 1
        if t_cond(t_combine({(0): lambda : n > 1, (111): lambda : n == 1, (112): lambda : n != 1, (113): lambda : n < 1, (114): lambda : n <= 1, (115): lambda : n >= 1})):
            factors.append(n)
        return factors

    @t_wrap
    def do_it():
        res = factorize(3242)
        expected = t_sv([2, 1621])
        t_assert(res == expected)

    reinit(execution_mode=mode, no_atexit=True)
    do_it()
    assert get_killed() == gen_killed([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 22, 66, 71, 74, 75, 91, 94, 98, 111, 113, 114])


@pytest.mark.parametrize("mode", MODES)
def test_caesar(mode) ->None:
    @t_wrap
    def caesar(string: List[int], key: int, decode: bool=False) ->List[int]:
        if t_cond(decode):
            key = t_combine({(0): lambda : 26 - key, (1): lambda : 26 + key, (2): lambda : 26 * key, (4): lambda : 26 % key, (5): lambda : 26 << key, (6): lambda : 26 >> key, (7): lambda : 26 | key, (8): lambda : 26 ^ key, (9): lambda : 26 & key, (10): lambda : 26 // key})
        res = t_sv([])
        ii = 0
        for c in string:
            ii += 1
            if t_cond(t_combine({(0): lambda : c <= 64, (11): lambda : c == 64, (12): lambda : c != 64, (13): lambda : c < 64, (14): lambda : c > 64, (15): lambda : c >= 64})):
                pass
            elif t_cond(t_combine({(0): lambda : c >= 90, (16): lambda : c == 90, (17): lambda : c != 90, (18): lambda : c < 90, (19): lambda : c <= 90, (20): lambda : c > 90})):
                pass
            else:
                c = t_combine({(0): lambda : c - 65, (21): lambda : c + 65, (22): lambda : c * 65, (24): lambda : c % 65, (25): lambda : c << 65, (26): lambda : c >> 65, (27): lambda : c | 65, (28): lambda : c ^ 65, (29): lambda : c & 65, (30): lambda : c // 65})
                c = t_combine({(0): lambda : c + key, (31): lambda : c - key, (32): lambda : c * key, (34): lambda : c % key, (35): lambda : c << key, (36): lambda : c >> key, (37): lambda : c | key, (38): lambda : c ^ key, (39): lambda : c & key, (40): lambda : c // key})
                c = t_combine({(0): lambda : c % 26, (41): lambda : c + 26, (42): lambda : c - 26, (43): lambda : c * 26, (45): lambda : c << 26, (46): lambda : c >> 26, (47): lambda : c | 26, (48): lambda : c ^ 26, (49): lambda : c & 26, (50): lambda : c // 26})
                c = t_combine({(0): lambda : c + 65, (51): lambda : c - 65, (52): lambda : c * 65, (54): lambda : c % 65, (55): lambda : c << 65, (56): lambda : c >> 65, (57): lambda : c | 65, (58): lambda : c ^ 65, (59): lambda : c & 65, (60): lambda : c // 65})
            res.append(c)
        return res

    reinit(execution_mode=mode, no_atexit=True)
    msg = 'The quick brown fox jumped over the lazy dogs'
    input = t_sv([ord(c) for c in msg])
    enc = caesar(input, 11)
    dec = caesar(enc, 11, decode=True)
    t_assert(input == dec)
    assert get_killed() == gen_killed([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 18, 19, 22, 25, 26, 27, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60])


@pytest.mark.parametrize("mode", MODES)
def test_tonelli_shanks(mode) ->None:
    @t_wrap
    def pow(base: int, exp: int, mod: int) ->int:
        res: int = t_combine({(0): lambda : base ** exp, (1): lambda : base + exp, (2): lambda : base - exp, (3): lambda : base * exp, (5): lambda : base % exp, (6): lambda : base << exp, (7): lambda : base >> exp, (8): lambda : base | exp, (9): lambda : base ^ exp, (10): lambda : base & exp, (11): lambda : base // exp})
        res = t_combine({(0): lambda : res % mod, (12): lambda : res + mod, (13): lambda : res - mod, (14): lambda : res * mod, (16): lambda : res << mod, (17): lambda : res >> mod, (18): lambda : res | mod, (19): lambda : res ^ mod, (20): lambda : res & mod, (21): lambda : res // mod})
        return res


    @t_wrap
    def legendre_symbol(a: int, p: int) ->int:
        """
        Legendre symbol
        Define if a is a quadratic residue modulo odd prime
        http://en.wikipedia.org/wiki/Legendre_symbol
        """
        ls = t_combine({(0): lambda : p - 1, (22): lambda : p + 1, (23): lambda : p * 1, (25): lambda : p % 1, (26): lambda : p << 1, (27): lambda : p >> 1, (28): lambda : p | 1, (29): lambda : p ^ 1, (30): lambda : p & 1, (31): lambda : p // 1})
        ls = t_combine({(0): lambda : ls // 2, (32): lambda : ls + 2, (33): lambda : ls - 2, (34): lambda : ls * 2, (36): lambda : ls % 2, (37): lambda : ls << 2, (38): lambda : ls >> 2, (39): lambda : ls | 2, (40): lambda : ls ^ 2, (41): lambda : ls & 2})
        ls = pow(a, ls, p)
        p_less = t_combine({(0): lambda : p - 1, (42): lambda : p + 1, (43): lambda : p * 1, (44): lambda : p / 1, (45): lambda : p % 1, (46): lambda : p << 1, (47): lambda : p >> 1, (48): lambda : p | 1, (49): lambda : p ^ 1, (50): lambda : p & 1, (51): lambda : p // 1})
        if t_cond(ls == p_less):
            return -1
        return ls


    @t_wrap
    def prime_mod_sqrt(a: int, p: int) ->list[int]:
        """
        Square root modulo prime number
        Solve the equation
            x^2 = a mod p
        and return list of x solution
        http://en.wikipedia.org/wiki/Tonelli-Shanks_algorithm
        """
        a = t_combine({(0): lambda : a % p, (52): lambda : a + p, (53): lambda : a - p, (54): lambda : a * p, (56): lambda : a << p, (57): lambda : a >> p, (58): lambda : a | p, (59): lambda : a ^ p, (60): lambda : a & p, (61): lambda : a // p})
        if t_cond(a == 0):
            return t_sv([0])
        if t_cond(p == 2):
            return t_sv([a])
        leg_sym = legendre_symbol(a, p)
        if t_cond(t_combine({(0): lambda : leg_sym != 1, (62): lambda : leg_sym == 1, (63): lambda : leg_sym < 1, (64): lambda : leg_sym <= 1, (65): lambda : leg_sym > 1, (66): lambda : leg_sym >= 1})):
            return t_sv([])
        p_mod = t_combine({(0): lambda : p % 4, (67): lambda : p + 4, (68): lambda : p - 4, (69): lambda : p * 4, (70): lambda : p / 4, (71): lambda : p << 4, (72): lambda : p >> 4, (73): lambda : p | 4, (74): lambda : p ^ 4, (75): lambda : p & 4, (76): lambda : p // 4})
        if t_cond(p_mod == 3):
            x = t_combine({(0): lambda : p + 1, (77): lambda : p - 1, (78): lambda : p * 1, (80): lambda : p % 1, (81): lambda : p << 1, (82): lambda : p >> 1, (83): lambda : p | 1, (84): lambda : p ^ 1, (85): lambda : p & 1, (86): lambda : p // 1})
            x = t_combine({(0): lambda : x // 4, (87): lambda : x + 4, (88): lambda : x - 4, (89): lambda : x * 4, (91): lambda : x % 4, (92): lambda : x << 4, (93): lambda : x >> 4, (94): lambda : x | 4, (95): lambda : x ^ 4, (96): lambda : x & 4})
            x = pow(a, x, p)
            return t_sv([x, t_combine({(0): lambda : p - x, (97): lambda : p + x, (98): lambda : p * x, (100): lambda : p % x, (101): lambda : p << x, (102): lambda : p >> x, (103): lambda : p | x, (104): lambda : p ^ x, (105): lambda : p & x, (106): lambda : p // x})])
        q = t_combine({(0): lambda : p - 1, (107): lambda : p + 1, (108): lambda : p * 1, (110): lambda : p % 1, (111): lambda : p << 1, (112): lambda : p >> 1, (113): lambda : p | 1, (114): lambda : p ^ 1, (115): lambda : p & 1, (116): lambda : p // 1})
        s = t_combine({(0): lambda : 0, (118): lambda : 0 + 1, (119): lambda : 0 * 2})
        max_iter = t_combine({(0): lambda : 10, (122): lambda : 10 + 1, (123): lambda : 10 * 2})
        while True:
            if t_cond(max_iter <= 0):
                break
            q_mod = t_combine({(0): lambda : q % 2, (125): lambda : q + 2, (126): lambda : q - 2, (127): lambda : q * 2, (128): lambda : q / 2, (129): lambda : q << 2, (130): lambda : q >> 2, (131): lambda : q | 2, (132): lambda : q ^ 2, (133): lambda : q & 2, (134): lambda : q // 2})
            if t_cond(q_mod != 0):
                break
            s = t_combine({(0): lambda : s + 1, (135): lambda : s - 1, (136): lambda : s * 1, (138): lambda : s % 1, (139): lambda : s << 1, (140): lambda : s >> 1, (141): lambda : s | 1, (142): lambda : s ^ 1, (143): lambda : s & 1, (144): lambda : s // 1})
            q = t_combine({(0): lambda : q // 2, (145): lambda : q + 2, (146): lambda : q - 2, (147): lambda : q * 2, (149): lambda : q % 2, (151): lambda : q >> 2, (152): lambda : q | 2, (153): lambda : q ^ 2, (154): lambda : q & 2})
            max_iter = t_combine({(0): lambda : max_iter - 1, (155): lambda : max_iter + 1, (156): lambda : max_iter * 1, (158): lambda : max_iter % 1, (159): lambda : max_iter << 1, (160): lambda : max_iter >> 1, (161): lambda : max_iter | 1, (162): lambda : max_iter ^ 1, (163): lambda : max_iter & 1, (164): lambda : max_iter // 1})
        z = t_combine({(0): lambda : 1, (166): lambda : 1 + 1, (167): lambda : 1 * 2})
        max_iter = t_combine({(0): lambda : 10, (170): lambda : 10 + 1, (171): lambda : 10 * 2, (172): lambda : not 10})
        while True:
            if t_cond(max_iter <= 0):
                break
            leg_sym = legendre_symbol(z, p)
            if t_cond(t_combine({(0): lambda : leg_sym == -1, (173): lambda : leg_sym != -1, (174): lambda : leg_sym < -1, (175): lambda : leg_sym <= -1, (176): lambda : leg_sym > -1, (177): lambda : leg_sym >= -1})):
                break
            z = t_combine({(0): lambda : z + 1, (178): lambda : z - 1, (179): lambda : z * 1, (181): lambda : z % 1, (182): lambda : z << 1, (183): lambda : z >> 1, (184): lambda : z | 1, (185): lambda : z ^ 1, (186): lambda : z & 1, (187): lambda : z // 1})
            max_iter = t_combine({(0): lambda : max_iter - 1, (188): lambda : max_iter + 1, (189): lambda : max_iter * 1, (191): lambda : max_iter % 1, (192): lambda : max_iter << 1, (193): lambda : max_iter >> 1, (194): lambda : max_iter | 1, (195): lambda : max_iter ^ 1, (196): lambda : max_iter & 1, (197): lambda : max_iter // 1})
        c = pow(z, q, p)
        x = t_combine({(0): lambda : q + 1, (198): lambda : q - 1, (199): lambda : q * 1, (201): lambda : q % 1, (202): lambda : q << 1, (203): lambda : q >> 1, (204): lambda : q | 1, (205): lambda : q ^ 1, (206): lambda : q & 1, (207): lambda : q // 1})
        x = t_combine({(0): lambda : x // 2, (208): lambda : x + 2, (209): lambda : x - 2, (210): lambda : x * 2, (212): lambda : x % 2, (213): lambda : x << 2, (214): lambda : x >> 2, (215): lambda : x | 2, (216): lambda : x ^ 2, (217): lambda : x & 2})
        x = pow(a, x, p)
        t = pow(a, q, p)
        m = t_combine({(0): lambda : s, (219): lambda : s + 1, (220): lambda : s * 2})
        max_iter_outer = t_combine({(0): lambda : 10, (223): lambda : 10 + 1, (224): lambda : 10 * 2})
        while True:
            if t_cond(max_iter_outer <= 0):
                break
            if t_cond(t == 1):
                break
            i = t_combine({(0): lambda : 0, (227): lambda : 0 + 1, (228): lambda : 0 * 2})
            e = t_combine({(0): lambda : 2, (231): lambda : 2 + 1, (232): lambda : 2 * 2})
            i = t_combine({(0): lambda : 1, (234): lambda : 1 != 1, (235): lambda : 1 + 1, (236): lambda : 1 * 2, (237): lambda : not 1})
            max_iter_inner = t_combine({(0): lambda : 10, (239): lambda : 10 + 1, (240): lambda : 10 * 2})
            while True:
                if t_cond(max_iter_inner <= 0):
                    break
                if t_cond(i > m):
                    break
                pp = pow(t, e, p)
                if t_cond(t_combine({(0): lambda : pp == 1, (242): lambda : pp != 1, (243): lambda : pp < 1, (244): lambda : pp <= 1, (245): lambda : pp > 1, (246): lambda : pp >= 1})):
                    break
                e = t_combine({(0): lambda : e * 2, (247): lambda : e + 2, (248): lambda : e - 2, (250): lambda : e % 2, (251): lambda : e << 2, (252): lambda : e >> 2, (253): lambda : e | 2, (254): lambda : e ^ 2, (255): lambda : e & 2, (256): lambda : e // 2})
                i = t_combine({(0): lambda : i + 1, (257): lambda : i - 1, (258): lambda : i * 1, (260): lambda : i % 1, (261): lambda : i << 1, (262): lambda : i >> 1, (263): lambda : i | 1, (264): lambda : i ^ 1, (265): lambda : i & 1, (266): lambda : i // 1})
                max_iter_inner = t_combine({(0): lambda : max_iter_inner - 1, (267): lambda : max_iter_inner + 1, (268): lambda : max_iter_inner * 1, (270): lambda : max_iter_inner % 1, (271): lambda : max_iter_inner << 1, (272): lambda : max_iter_inner >> 1, (273): lambda : max_iter_inner | 1, (274): lambda : max_iter_inner ^ 1, (275): lambda : max_iter_inner & 1, (276): lambda : max_iter_inner // 1})
            b = t_combine({(0): lambda : m - i, (277): lambda : m + i, (278): lambda : m * i, (280): lambda : m % i, (281): lambda : m << i, (282): lambda : m >> i, (283): lambda : m | i, (284): lambda : m ^ i, (285): lambda : m & i, (286): lambda : m // i})
            b = t_combine({(0): lambda : b - 1, (287): lambda : b + 1, (288): lambda : b * 1, (290): lambda : b % 1, (291): lambda : b << 1, (292): lambda : b >> 1, (293): lambda : b | 1, (294): lambda : b ^ 1, (295): lambda : b & 1, (296): lambda : b // 1})
            b = t_combine({(0): lambda : 2 ** b, (297): lambda : 2 + b, (298): lambda : 2 - b, (299): lambda : 2 * b, (301): lambda : 2 % b, (302): lambda : 2 << b, (303): lambda : 2 >> b, (304): lambda : 2 | b, (305): lambda : 2 ^ b, (306): lambda : 2 & b, (307): lambda : 2 // b})
            b = pow(c, b, p)
            x = t_combine({(0): lambda : x * b, (308): lambda : x + b, (309): lambda : x - b, (311): lambda : x % b, (312): lambda : x << b, (313): lambda : x >> b, (314): lambda : x | b, (315): lambda : x ^ b, (316): lambda : x & b, (317): lambda : x // b})
            x = t_combine({(0): lambda : x % p, (318): lambda : x + p, (319): lambda : x - p, (320): lambda : x * p, (322): lambda : x << p, (323): lambda : x >> p, (324): lambda : x | p, (325): lambda : x ^ p, (326): lambda : x & p, (327): lambda : x // p})
            t = t_combine({(0): lambda : t * b, (328): lambda : t + b, (329): lambda : t - b, (331): lambda : t % b, (332): lambda : t << b, (333): lambda : t >> b, (334): lambda : t | b, (335): lambda : t ^ b, (336): lambda : t & b, (337): lambda : t // b})
            t = t_combine({(0): lambda : t * b, (338): lambda : t + b, (339): lambda : t - b, (341): lambda : t % b, (342): lambda : t << b, (343): lambda : t >> b, (344): lambda : t | b, (345): lambda : t ^ b, (346): lambda : t & b, (347): lambda : t // b})
            t = t_combine({(0): lambda : t % p, (348): lambda : t + p, (349): lambda : t - p, (350): lambda : t * p, (352): lambda : t << p, (353): lambda : t >> p, (354): lambda : t | p, (355): lambda : t ^ p, (356): lambda : t & p, (357): lambda : t // p})
            c = t_combine({(0): lambda : b * b, (358): lambda : b + b, (359): lambda : b - b, (361): lambda : b % b, (362): lambda : b << b, (363): lambda : b >> b, (364): lambda : b | b, (365): lambda : b ^ b, (366): lambda : b & b, (367): lambda : b // b})
            c = t_combine({(0): lambda : c % p, (368): lambda : c + p, (369): lambda : c - p, (370): lambda : c * p, (372): lambda : c << p, (373): lambda : c >> p, (374): lambda : c | p, (375): lambda : c ^ p, (376): lambda : c & p, (377): lambda : c // p})
            m = t_combine({(0): lambda : i, (378): lambda : i != 1, (379): lambda : i + 1, (380): lambda : i * 2, (381): lambda : not i})
            max_iter_outer = t_combine({(0): lambda : max_iter_outer - 1, (382): lambda : max_iter_outer + 1, (383): lambda : max_iter_outer * 1, (385): lambda : max_iter_outer % 1, (386): lambda : max_iter_outer << 1, (387): lambda : max_iter_outer >> 1, (388): lambda : max_iter_outer | 1, (389): lambda : max_iter_outer ^ 1, (390): lambda : max_iter_outer & 1, (391): lambda : max_iter_outer // 1})
        return t_sv([x, p - x])


    @t_wrap
    def do_it() ->None:
        res = prime_mod_sqrt(5, 41)
        t_assert(res == t_sv([28, 13]))

    reinit(execution_mode=mode, no_atexit=True)
    do_it()
    assert get_killed() == gen_killed([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 30, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 54, 56, 57, 58, 59, 60, 61, 62, 64, 66, 107, 108, 110, 111, 112, 113, 115, 116, 118, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 149, 151, 152, 153, 154, 158, 163, 172, 173, 174, 176, 177, 178, 179, 181, 182, 183, 184, 185, 186, 187, 191, 196, 198, 199, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212, 213, 214, 215, 216, 217, 219, 220, 231, 232, 234, 235, 236, 237, 242, 243, 245, 246, 253, 255, 256, 257, 258, 260, 262, 263, 264, 265, 266, 277, 278, 280, 281, 282, 283, 284, 285, 286, 287, 288, 291, 293, 295, 296, 297, 298, 299, 301, 302, 303, 304, 305, 306, 307, 308, 309, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 322, 323, 324, 325, 326, 327, 328, 329, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 352, 353, 354, 355, 356, 357, 358, 359, 361, 362, 363, 364, 365, 366, 367, 370, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 385, 390])


#################################################
# real-world tests for split stream variants


@pytest.mark.skipif(os.environ.get("TEST_SKIP_SPLIT_MODES") is not None, reason="Skip split variant tests.")
@pytest.mark.parametrize("mode", SPLIT_STREAM_MODES)
def test_approx_exp_split_stream(mode):
    from typing import Optional

    def compute_exp(x: int, accuracy: int) ->Optional[int]:
        extra_precision = 4
        accuracy_scaler = t_combine({(0): lambda : 10 ** accuracy, (1): lambda : 10 + accuracy, (2): lambda : 10 - accuracy, (3): lambda : 10 * accuracy, (4): lambda : 10 / accuracy, (5): lambda : 10 % accuracy, (6): lambda : 10 << accuracy, (7): lambda : 10 >> accuracy, (8): lambda : 10 | accuracy, (9): lambda : 10 ^ accuracy, (10): lambda : 10 & accuracy, (11): lambda : 10 // accuracy})
        extra_scaler = t_combine({(0): lambda : 10 ** extra_precision, (12): lambda : 10 + extra_precision, (13): lambda : 10 - extra_precision, (14): lambda : 10 * extra_precision, (16): lambda : 10 % extra_precision, (17): lambda : 10 << extra_precision, (18): lambda : 10 >> extra_precision, (19): lambda : 10 | extra_precision, (20): lambda : 10 ^ extra_precision, (21): lambda : 10 & extra_precision, (22): lambda : 10 // extra_precision})
        full_scaler = t_combine({(0): lambda : accuracy_scaler * extra_scaler, (23): lambda : accuracy_scaler + extra_scaler, (24): lambda : accuracy_scaler - extra_scaler, (25): lambda : accuracy_scaler / extra_scaler, (26): lambda : accuracy_scaler % extra_scaler, (27): lambda : accuracy_scaler << extra_scaler, (28): lambda : accuracy_scaler >> extra_scaler, (29): lambda : accuracy_scaler | extra_scaler, (30): lambda : accuracy_scaler ^ extra_scaler, (31): lambda : accuracy_scaler & extra_scaler, (32): lambda : accuracy_scaler // extra_scaler})
        sum_low = 0
        sum_high = 0
        term_low = full_scaler
        term_high = full_scaler
        floor_x = t_combine({(0): lambda : x // accuracy_scaler, (33): lambda : x + accuracy_scaler, (34): lambda : x - accuracy_scaler, (35): lambda : x * accuracy_scaler, (36): lambda : x / accuracy_scaler, (37): lambda : x % accuracy_scaler, (38): lambda : x << accuracy_scaler, (39): lambda : x >> accuracy_scaler, (40): lambda : x | accuracy_scaler, (41): lambda : x ^ accuracy_scaler, (42): lambda : x & accuracy_scaler})
        i = 0
        itr = 0
        while True:
            if t_combine({(0): lambda : term_low <= 0, (43): lambda : term_low == 0, (44): lambda : term_low != 0, (45): lambda : term_low < 0, (46): lambda : term_low > 0, (47): lambda : term_low >= 0}):
                break
            if t_combine({(0): lambda : itr >= 100, (48): lambda : itr == 100, (49): lambda : itr != 100, (50): lambda : itr < 100, (51): lambda : itr <= 100, (52): lambda : itr > 100}):
                break
            sum_low = t_combine({(0): lambda : sum_low + term_low, (53): lambda : sum_low - term_low, (54): lambda : sum_low * term_low, (55): lambda : sum_low / term_low, (56): lambda : sum_low % term_low, (57): lambda : sum_low << term_low, (58): lambda : sum_low >> term_low, (59): lambda : sum_low | term_low, (60): lambda : sum_low ^ term_low, (61): lambda : sum_low & term_low, (62): lambda : sum_low // term_low})
            sum_high = t_combine({(0): lambda : sum_high + term_high, (63): lambda : sum_high - term_high, (64): lambda : sum_high * term_high, (65): lambda : sum_high / term_high, (66): lambda : sum_high % term_high, (67): lambda : sum_high << term_high, (68): lambda : sum_high >> term_high, (69): lambda : sum_high | term_high, (70): lambda : sum_high ^ term_high, (71): lambda : sum_high & term_high, (72): lambda : sum_high // term_high})
            term_low = t_combine({(0): lambda : term_low * x, (73): lambda : term_low + x, (74): lambda : term_low - x, (75): lambda : term_low / x, (76): lambda : term_low % x, (77): lambda : term_low << x, (78): lambda : term_low >> x, (79): lambda : term_low | x, (80): lambda : term_low ^ x, (81): lambda : term_low & x, (82): lambda : term_low // x})
            term_low = t_combine({(0): lambda : term_low // accuracy_scaler, (83): lambda : term_low + accuracy_scaler, (84): lambda : term_low - accuracy_scaler, (85): lambda : term_low * accuracy_scaler, (86): lambda : term_low / accuracy_scaler, (87): lambda : term_low % accuracy_scaler, (88): lambda : term_low << accuracy_scaler, (89): lambda : term_low >> accuracy_scaler, (90): lambda : term_low | accuracy_scaler, (91): lambda : term_low ^ accuracy_scaler, (92): lambda : term_low & accuracy_scaler})
            high_accuracy_scaler = t_combine({(0): lambda : accuracy_scaler + 1, (93): lambda : accuracy_scaler - 1, (94): lambda : accuracy_scaler * 1, (95): lambda : accuracy_scaler / 1, (96): lambda : accuracy_scaler % 1, (97): lambda : accuracy_scaler << 1, (98): lambda : accuracy_scaler >> 1, (99): lambda : accuracy_scaler | 1, (100): lambda : accuracy_scaler ^ 1, (101): lambda : accuracy_scaler & 1, (102): lambda : accuracy_scaler // 1})
            term_high = t_combine({(0): lambda : term_high * x, (103): lambda : term_high + x, (104): lambda : term_high - x, (105): lambda : term_high / x, (106): lambda : term_high % x, (107): lambda : term_high << x, (108): lambda : term_high >> x, (109): lambda : term_high | x, (110): lambda : term_high ^ x, (111): lambda : term_high & x, (112): lambda : term_high // x})
            term_high = t_combine({(0): lambda : term_high // high_accuracy_scaler, (113): lambda : term_high + high_accuracy_scaler, (114): lambda : term_high - high_accuracy_scaler, (115): lambda : term_high * high_accuracy_scaler, (116): lambda : term_high / high_accuracy_scaler, (117): lambda : term_high % high_accuracy_scaler, (118): lambda : term_high << high_accuracy_scaler, (119): lambda : term_high >> high_accuracy_scaler, (120): lambda : term_high | high_accuracy_scaler, (121): lambda : term_high ^ high_accuracy_scaler, (122): lambda : term_high & high_accuracy_scaler})
            if t_combine({(0): lambda : i > floor_x, (123): lambda : i == floor_x, (124): lambda : i != floor_x, (125): lambda : i < floor_x, (126): lambda : i <= floor_x, (127): lambda : i >= floor_x}):
                if t_combine({(0): lambda : term_high < extra_scaler, (128): lambda : term_high == extra_scaler, (129): lambda : term_high != extra_scaler, (130): lambda : term_high <= extra_scaler, (131): lambda : term_high > extra_scaler, (132): lambda : term_high >= extra_scaler}):
                    sum_upper_bound = t_combine({(0): lambda : sum_high + term_high, (133): lambda : sum_high - term_high, (134): lambda : sum_high * term_high, (135): lambda : sum_high / term_high, (136): lambda : sum_high % term_high, (137): lambda : sum_high << term_high, (138): lambda : sum_high >> term_high, (139): lambda : sum_high | term_high, (140): lambda : sum_high ^ term_high, (141): lambda : sum_high & term_high, (142): lambda : sum_high // term_high})
                    temp: int = t_combine({(0): lambda : sum_low // extra_scaler, (143): lambda : sum_low + extra_scaler, (144): lambda : sum_low - extra_scaler, (145): lambda : sum_low * extra_scaler, (146): lambda : sum_low / extra_scaler, (147): lambda : sum_low % extra_scaler, (148): lambda : sum_low << extra_scaler, (149): lambda : sum_low >> extra_scaler, (150): lambda : sum_low | extra_scaler, (151): lambda : sum_low ^ extra_scaler, (152): lambda : sum_low & extra_scaler})
                    temp = round(temp)
                    upper: int = t_combine({(0): lambda : sum_upper_bound // extra_scaler, (153): lambda : sum_upper_bound + extra_scaler, (154): lambda : sum_upper_bound - extra_scaler, (155): lambda : sum_upper_bound * extra_scaler, (156): lambda : sum_upper_bound / extra_scaler, (157): lambda : sum_upper_bound % extra_scaler, (158): lambda : sum_upper_bound << extra_scaler, (159): lambda : sum_upper_bound >> extra_scaler, (160): lambda : sum_upper_bound | extra_scaler, (161): lambda : sum_upper_bound ^ extra_scaler, (162): lambda : sum_upper_bound & extra_scaler})
                    upper = round(upper)
                    if t_combine({(0): lambda : upper == temp, (163): lambda : upper != temp, (164): lambda : upper < temp, (165): lambda : upper <= temp, (166): lambda : upper > temp, (167): lambda : upper >= temp}):
                        return temp
            i = t_combine({(0): lambda : i + 1, (168): lambda : i - 1, (169): lambda : i * 1, (171): lambda : i % 1, (172): lambda : i << 1, (173): lambda : i >> 1, (174): lambda : i | 1, (175): lambda : i ^ 1, (176): lambda : i & 1, (177): lambda : i // 1})
            term_low = t_combine({(0): lambda : term_low // i, (178): lambda : term_low + i, (179): lambda : term_low - i, (180): lambda : term_low * i, (181): lambda : term_low / i, (182): lambda : term_low % i, (183): lambda : term_low << i, (184): lambda : term_low >> i, (185): lambda : term_low | i, (186): lambda : term_low ^ i, (187): lambda : term_low & i})
            term_high = t_combine({(0): lambda : term_high // i, (188): lambda : term_high + i, (189): lambda : term_high - i, (190): lambda : term_high * i, (191): lambda : term_high / i, (192): lambda : term_high % i, (193): lambda : term_high << i, (194): lambda : term_high >> i, (195): lambda : term_high | i, (196): lambda : term_high ^ i, (197): lambda : term_high & i})
            term_high = t_combine({(0): lambda : term_high + 1, (198): lambda : term_high - 1, (199): lambda : term_high * 1, (200): lambda : term_high / 1, (201): lambda : term_high % 1, (202): lambda : term_high << 1, (203): lambda : term_high >> 1, (204): lambda : term_high | 1, (205): lambda : term_high ^ 1, (206): lambda : term_high & 1, (207): lambda : term_high // 1})
            itr = t_combine({(0): lambda : itr + 1, (208): lambda : itr - 1, (209): lambda : itr * 1, (211): lambda : itr % 1, (212): lambda : itr << 1, (213): lambda : itr >> 1, (214): lambda : itr | 1, (215): lambda : itr ^ 1, (216): lambda : itr & 1, (217): lambda : itr // 1})
        return None


    def test_approx_exp() ->None:
        res = compute_exp(3, 2)
        print(res)
        t_assert(res == 103)

    reinit(execution_mode=mode, no_atexit=True)

    try:
        test_approx_exp()
    except Exception as e:
        t_final_exception_test()

    t_wait_for_forks()
    killed = get_killed()
    print(killed)
    assert killed == gen_killed([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 38, 40, 41, 44, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 96, 97, 98, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 122, 123, 125, 126, 128, 131, 132, 134, 135, 136, 141, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 164, 166, 168, 169, 171, 172, 173, 176, 177, 182, 183, 184, 187, 192, 193, 194, 197, 201, 202, 203, 206])
    


#################################################
# tests for tuple


@pytest.mark.parametrize("mode", MODES)
def test_tuple_eq_with_tint_elem(mode):
    @t_wrap
    def fun():
        tainted_int = t_combine({0: 0, 1: 1})
        data = ShadowVariable((1, 2, 3, tainted_int), False)
        return data

    reinit(execution_mode=mode, no_atexit=True)
    res = fun()
    t_assert(res == (1, 2, 3, 0))
    assert get_killed() == gen_killed([1])

    reinit(execution_mode=mode, no_atexit=True)
    res = fun()
    t_assert((1, 2, 3, 0) == res)
    assert get_killed() == gen_killed([1])


#################################################
# tests for list


@pytest.mark.parametrize("mode", MODES)
def test_list_eq_with_tint_elem(mode):
    @t_wrap
    def fun():
        tainted_int = t_combine({0: 0, 1: 1})
        data = [1, 2, 3, tainted_int]
        return data

    reinit(execution_mode=mode, no_atexit=True)
    data = fun()
    t_assert(data == [1, 2, 3, 0])
    assert get_killed() == gen_killed([1])


@pytest.mark.parametrize("mode", MODES)
def test_list_mul_tint(mode):
    @t_wrap
    def fun():
        # data = ShadowVariable([], False)
        tainted_int = t_combine({0: 0, 1: 2})

        # create a list where the length is dependent on the tainted int
        data = ShadowVariable([], False)
        new_data = ShadowVariable([1], False)*tainted_int
        data.extend(new_data)
        res = 0
        ii = 0
        while True:
            if t_cond(ii > data.__len__() - 1):
                break
            add = data[ii]
            res += add
            ii += 1
        return res

    reinit(execution_mode=mode, no_atexit=True)
    result = fun()
    t_assert(result == 0)
    assert get_killed() == gen_killed([1])


@pytest.mark.parametrize("mode", MODES)
def test_list_insert_tint(mode):
    reinit(execution_mode=mode, no_atexit=True)
    data = ShadowVariable([1, 2, 3], False)
    tainted_int = t_combine({0: 0, 1: 1})

    # insert data at pos defined by tainted int
    data.insert(tainted_int, 'a')

    t_assert(data[0] == 'a')
    t_assert(data[1] == 1)
    assert get_killed() == gen_killed([1])




#################################################
# tests for set


@pytest.mark.parametrize("mode", MODES)
def test_set_add_tint(mode):
    reinit(execution_mode=mode, no_atexit=True)
    data = ShadowVariable(set([1, 2, 3]), False)
    tainted_int = t_combine({0: 1, 1: 4})

    # insert data at pos defined by tainted int
    data.add(tainted_int)

    t_assert(data.__len__() == 3)
    assert get_killed() == gen_killed([1])


@pytest.mark.parametrize("mode", MODES)
def test_set_init_with_tint(mode):
    reinit(execution_mode=mode, no_atexit=True)
    tainted_int = t_combine({0: 1, 1: 4})
    data = (1, 2, 3, tainted_int)
    data = set(data)
    data = ShadowVariable(data, False)

    res = data.__len__() == 3
    t_assert(res)
    assert get_killed() == gen_killed([1])


#################################################
# tests for dict


@pytest.mark.parametrize("mode", MODES)
def test_dict_key_tainted(mode):
    reinit(execution_mode=mode, no_atexit=True)

    data = ShadowVariable({}, False)
    tainted_int = t_combine({0: 0, 1: 1})
    data[tainted_int] = tainted_int

    for key in data:
        t_assert(key == 0)
        t_assert(data[key] == 0)

    assert get_killed() == gen_killed([1])


@pytest.mark.parametrize("mode", MODES)
def test_dict_init_tainted(mode):
    reinit(execution_mode=mode, no_atexit=True)

    def gen_val(disallowed: Optional[set[int]]=None) -> Union[int, ShadowVariable]:
        is_sv = bool(randint(0, 1))

        if is_sv:
            vals = {}
            wanted_len = randint(1, 3)
            while True:
                if 0 not in vals:
                    path = 0
                else:
                    path = randint(0, 10)
                if path not in vals:
                    val = randint(0, 100)
                    if disallowed is not None:
                        if val in disallowed:
                            continue
                        else:
                            disallowed.add(val)
                    vals[path] = val
                if len(vals) >= wanted_len:
                    break
            return ShadowVariable(vals, from_mapping=True)
        else:
            while True:
                val = randint(0, 100)
                if disallowed is not None:
                    if val in disallowed:
                        continue
                    else:
                        disallowed.add(val)
                return val

    for _ in range(100):
        data = {}
        disallowed: set[int] = set()
        for _ in range(randint(1, 3)):
            key = gen_val(disallowed)
            val = gen_val()
            data[key] = val
        sv = ShadowVariable(data, from_mapping=False)



#################################################
# tests for class

@t_class
class InitWithTaintCls():
    def __init__(self, val):
        self.val = val


@pytest.mark.parametrize("mode", MODES)
def test_class_init_with_taint(mode):

    @t_wrap
    def func(tainted_int):
        t = InitWithTaintCls(tainted_int)
        return t
    
    reinit(execution_mode=mode, no_atexit=True)
    res = func(t_combine({0: 0, 1: 1}))
    t_assert(res.val == 0)
    assert get_killed() == gen_killed([1])


@t_class
class UpdateWithTaintCls():
    def __init__(self):
        self.val = 0



@pytest.mark.parametrize("mode", MODES)
def test_class_attr_update_with_taint(mode):
    @t_wrap
    def func(tainted_int):
        t = UpdateWithTaintCls()
        t.val += tainted_int
        return t
    
    reinit(execution_mode=mode, no_atexit=True)
    res = func(t_combine({0: 0, 1: 1}))
    t_assert(res.val == 0)
    assert get_killed() == gen_killed([1])


@t_class
class CondAttrAccess():
    def __init__(self):
        self.val = 0


@pytest.mark.parametrize("mode", MODES)
def test_class_cond_attr_access(mode):

    @t_wrap
    def func(tainted_int):
        t = CondAttrAccess()
        if t_cond(tainted_int != 0):
            t.val += 1
        return t
    
    reinit(execution_mode=mode, no_atexit=True)
    res = func(t_combine({0: 0, 1: 1}))
    t_assert(res.val == 0)
    assert get_killed() == gen_killed([1])

# TODO mutation changes control flow as well as value, followed by fork where companion mutation 

# TODO recursive wrapped object init

# TODO multiple assigns (with SV values / forks and both) to the same member variable during init

# TODO test that SV objects can not contain other SV objects

# TODO nested function calls in __init__

# TODO internal references in wrapped object (cycles)

# TODO mutation in class function

# TODO function call same function for each shadow version should execute unified

# TODO attribute access/function call raises exception for some shadow versions

# TODO attribute access/function call raises exception for mainline

# TODO test calling non-wrapped functions (they don't contain mutations)


#################################################
# tests for external code

# external function needs to be called with untainted values