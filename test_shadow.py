import pytest
import logging
logger = logging.getLogger(__name__)

from shadow import reinit, t_final_exception_test, t_wrap, t_combine, t_wait_for_forks, t_get_killed, t_cond, t_assert, \
                   t_logical_path, t_seen_mutants, t_masked_mutants, ShadowVariable

MODES = ['shadow_fork'] # , 'shadow_fork', 'shadow_cache', 'shadow_fork_cache']
SPLIT_STREAM_MODES = ['split', 'modulo'] # , 'shadow_fork', 'shadow_cache', 'shadow_fork_cache']


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
    t_assert(func(t_combine({0: 0, 1: 1})) == 1)
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
    t_get_killed()
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
    t_get_killed()
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
    res = simple(3, 1)
    t_assert(res == 0)
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


#################################################
# real-world tests for split stream variants

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

    test_approx_exp()

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



#################################################
# tests for class
@pytest.mark.parametrize("mode", MODES)
def test_class_attr_access(mode):
    class Test():
        def __init__(self):
            self.val = 0

    @t_wrap
    def func(tainted_int):
        t = Test()
        t.val += tainted_int
        return t
    
    reinit(execution_mode=mode, no_atexit=True)
    res = func(t_combine({0: 0, 1: 1}))
    logger.debug(f"{res} {res.val}")
    t_assert(res.val == 0)
    assert get_killed() == gen_killed([1])


@pytest.mark.skip()
@pytest.mark.parametrize("mode", MODES)
def test_class_cond_attr_access(mode):
    class Test():
        def __init__(self):
            self.val = 0

    @t_wrap
    def func(tainted_int):
        t = Test()
        if t_cond(tainted_int != 0):
            t.val += 1
        return t
    
    reinit(execution_mode=mode, no_atexit=True)
    res = func(t_combine({0: 0, 1: 1}))
    logger.debug(f"{res} {res.val}")
    t_assert(res.val == 0)
    assert get_killed() == gen_killed([1])


@pytest.mark.parametrize("mode", MODES)
def test_class_init_taint(mode):
    class Test():
        def __init__(self, val):
            self.val = val

    @t_wrap
    def func(tainted_int):
        t = Test(tainted_int)
        return t
    
    reinit(execution_mode=mode, no_atexit=True)
    res = func(t_combine({0: 0, 1: 1}))
    logger.debug(f"{res} {res.val}")
    t_assert(res.val == 0)
    assert get_killed() == gen_killed([1])


@pytest.mark.parametrize("mode", MODES)
def test_class_wrap(mode):

    def taint(orig_class):
        # TODO this needs to be inserted by ast_mutator
        # TODO move to shadow.py
        orig_new = orig_class.__new__
        try:
            orig_deepcopy = orig_class.__getattr__('__deepcopy__')
        except AttributeError:
            orig_deepcopy = None

        def wrap_new(cls, *args, **kwargs):
            new = orig_new(cls)
            new._init = False
            obj = ShadowVariable(new, False, True)
            obj.__init__(*args, **kwargs)
            new._init = True
            return obj

        def deepcopy(*args, **kwargs):
            logger.debug("hi")
            raise NotImplementedError()

        orig_class._orig_new = orig_new
        orig_class.__new__ = wrap_new
        # orig_class.__deepcopy__ = deepcopy


        # cls_proxy = partial(proxy_function, orig_class)
        # for func in dir(orig_class):
        #     if func in ['__bool__']:
        #         setattr(orig_class, func, losing_taint)
        #         continue
        #     if func in [
        #         '_shadow',
        #         '__new__', '__init__', '__class__', '__dict__', '__getattribute__', '__repr__'
        #     ]:
        #         continue
        #     orig_func = getattr(orig_class, func)
        #     # logging.debug("%s %s", orig_class, func)
        #     setattr(orig_class, func, cls_proxy(func, orig_func))
        return orig_class

    @taint
    class Test():
        # def __new__(cls, *args, **kwargs):
        #     new = super().__new__(cls)
        #     obj = ShadowVariable(new, False)
        #     obj._wrapped_init(*args, **kwargs)
        #     return obj

        def __init__(self, val):
            self.val = val

    @t_wrap
    def func(tainted_int):
        t = Test(tainted_int)
        return t
    
    reinit(execution_mode=mode, no_atexit=True)
    res = func(t_combine({0: 0, 1: 1}))
    logger.debug(f"{res} {res.val}")
    t_assert(res.val == 0)
    assert get_killed() == gen_killed([1])

# TODO recursive wrapped object init

# TODO internal references in wrapped object (cycles)

# TODO mutation in class function

# TODO mutation in class method

# TODO init with vanilla values

# TODO init taking shadow values

# TODO function call same function for each shadow version should execute unified

# TODO function call different functions

# TODO attribute access fails for some shadow versions

# TODO attribute access fails for mainline


#################################################
# tests for external code

# external function needs to be called with untainted values