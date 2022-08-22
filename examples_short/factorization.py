from typing import List


inc = [4, 2, 4, 2, 4, 6, 2, 6]


def div(val: int, divisor: int) -> bool:
    modded = val % divisor
    return modded != 0


def factorize(n: int) -> List[int]:
    factors = []
    while True:
        is_div = div(n, 2)
        if is_div:
            break
        factors.append(2)
        n = n // 2
    while True:
        is_div = div(n, 3)
        if is_div:
            break
        factors.append(3)
        n = n // 3
    while True:
        is_div = div(n, 5)
        if is_div:
            break
        factors.append(5)
        n = n // 5
    k = 7
    i = 0
    while True:
        k_squared = k * k
        if k_squared > n:
            break
        is_div = div(n, k)
        n_mod_k = n % k
        if n_mod_k == 0:
            factors.append(k)
            n = n // k
        else:
            k = k + inc[i]
            if i < 7:
                i = i + 1
            else:
                i = 1
    if n > 1:
        factors.append(n)
    return factors


def test_fact() -> None:
    res = factorize(3242)
    expected = [2, 1621]
    assert res == expected


test_fact()