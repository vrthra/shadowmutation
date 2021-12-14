

def prime(input: int) -> bool:
    """
    >>> skips = ['2 : swap 0']
    >>> prime(3)
    True
    """
    for n in range(2, input):
        if input % n == 0:
            return False
    return True


def test_simple() -> None:
    assert prime(3) == True
    assert prime(4) == False
    assert prime(5) == True
    assert prime(6) == False
    assert prime(7) == True
    assert prime(8) == False
    assert prime(9) == False
    assert prime(10) == False


test_simple()