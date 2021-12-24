from typing import Tuple, List, Optional


def ln(val: float) -> float:
    # c_log = 1000.0 * ((c ** (1/1000.0)) - 1)
    c_log = 1/1000.0
    c_log = val ** c_log
    c_log = c_log - 1
    c_log = 1000.0 * c_log
    return c_log

 
def entropy(hist: List[int], l: int) -> Optional[float]:
    c = hist[0] / l

    c_log = ln(c)

    normalized = -c * c_log
    res = normalized
    for v in hist[1:]:
        c = v / l
        c_log = ln(c)
        normalized = -c * c_log
        # print(c, normalized)
        res = res + normalized
    return res
 
 
def test_entropy() -> None:
    source = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    res = entropy(source, len(source))
    if res is not None:
        # print(res)
        expected = 3.421568195457525
        diff = abs(res - expected)
        rounded_diff = round(diff, 8)
        assert rounded_diff == 0
    else:
        assert False

test_entropy()