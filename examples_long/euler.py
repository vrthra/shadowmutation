
def is_negligible(val: float) -> bool:
    return round(val, 3) == 0


def newton_cooling(h: float, y: float) -> float:
    new_y = y - 20
    update = -0.07 * new_y
    update_y = h * update
    return update_y


def euler(y0: float, a: float, b: int, h: int) -> float:
    t = a
    y = y0
    while True:
        # if t > b
        if t > b:
            break

        t = t + h
        update_y = newton_cooling(h, y)
        if is_negligible(update_y):
            break

        y = y + update_y
    return y

 
def test_euler() -> None:
    res = euler(100,0,100,10)
    res = round(res, 3)
    assert res == 20

test_euler()