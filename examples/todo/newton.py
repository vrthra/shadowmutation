from math import sqrt

def newton_method(number: float, number_iters: int = 100) -> float:

    a = float(number) 

    for i in range(number_iters): 

        number = 0.5 * (number + a / number) 

    return number

def test_newton_method() -> None:
    newt = newton_method(10)
    pyth = sqrt(10)
    diff = abs(newt - pyth)
    rounded_diff = round(diff, 8)
    assert rounded_diff == 0, f"{diff}, {rounded_diff}"

test_newton_method()