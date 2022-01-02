from typing import List


def div(val: int, divisor: int) -> bool:
    # print(val, divisor, val % divisor)
    return (val % divisor) != 0


# calculate the sum of the factors using wheel factorization
def factorize(n: int) -> int:
    factors = 0
    while True:
        is_div = div(n, 2)
        if is_div:
            break
        factors = factors + 2
        n = n // 2
    while True:
        is_div = div(n, 3)
        if is_div:
            break
        factors = factors + 3
        n = n // 3
    while True:
        is_div = div(n, 5)
        if is_div:
            break
        factors = factors + 5
        n = n // 5
    k = 7
    i = 0
    while True:
        # print(k, n)
        k_squared = k * k
        if k_squared > n:
            break
        is_div = div(n, k)
        n_mod = n % k
        if n_mod == 0:
            factors = factors + k
            n = n // k
        else:
            if i == 0:
                k = k + 4
            if i == 1:
                k = k + 2
            if i == 2:
                k = k + 4
            if i == 3:
                k = k + 2
            if i == 4:
                k = k + 4
            if i == 5:
                k = k + 6
            if i == 6:
                k = k + 2
            if i == 7:
                k = k + 6

            if i < 7:
                i = i + 1
            else:
                i = 1
    if n > 1:
        factors = factors + n
    return factors


def test_fact() -> None:
    res = factorize(3242)
    # print(res)
    expected = 2 + 1621 # [2, 1621]
    assert res == expected


test_fact()