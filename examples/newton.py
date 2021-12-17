from math import sqrt

def newton_method(number: float, number_iters: int = 100) -> float:

    a = float(number) 

    for i in range(number_iters): 

        number = 0.5 * (number + a / number) 

    return number

def test_newton_method() -> None:
    for i in range(1, 100):
        newt = newton_method(i)
        pyth = sqrt(i)
        diff = abs(newt - pyth)
        rounded_diff = round(diff, 8)
        assert rounded_diff == 0, f"{diff}, {rounded_diff}"

test_newton_method()