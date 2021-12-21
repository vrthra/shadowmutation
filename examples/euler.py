from typing import Callable

def euler(y0: float, a: float, b: int, h: int) -> float:
    t: float = a
    y: float = y0
    while True:
        if (b - t) < 0:
            break
        # print("%6.3f %6.3f" % (t, y))
        t += h
        cooling_factor = (y - 20)
        newton_cooling = -0.07 * cooling_factor
        update_y = h * newton_cooling
        y += update_y
    return y
 
# def newtoncooling(time: float, temp: float) -> float:
# 	return 
 
def test_euler() -> None:
    res = euler(100,0,100,10)
    res = round(res, 3)
    assert res == 20

test_euler()