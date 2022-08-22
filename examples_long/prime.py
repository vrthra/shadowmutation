

def check_max_runtime(n: int) -> bool:
    if n >= 100:
        return True
    return False


def prime(input: int) -> bool:
    """
    >>> skips = ['2 : swap 0']
    >>> prime(3)
    True
    """
    if input == 0:
        return False
    if input == 1:
        return False
    if input == 2:
        return True
    ctr = 0
    n = 2
    while True:
        if n >= input:
            break
        modulo = input % n
        if modulo == 0:
            return False
        n = n + 1
        if check_max_runtime(ctr):
            break
        ctr = ctr + 1
    return True


def test_simple() -> None:
    # assert prime(3) == True
    # assert prime(4) == False
    assert prime(5) == True
    # assert prime(6) == False
    # assert prime(7) == True
    # assert prime(8) == False
    # assert prime(9) == False
    # assert prime(10) == False


test_simple()