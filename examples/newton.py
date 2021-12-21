from math import sqrt

def newton_method(number: float, number_iters: int = 100) -> float:

    a = float(number) 

    i = 0
    while True:
        if i >= number_iters:
            break

        number = 0.5 * (number + a / number) 

        i += 1

    return number

def test_newton_method() -> None:
    val = 10
    number_iters = 10
    newt = newton_method(val, number_iters=number_iters)
    pyth = sqrt(val)
    diff = abs(newt - pyth)
    rounded_diff = round(diff, 8)
    assert rounded_diff == 0, f"{diff}, {rounded_diff}"

test_newton_method()