from typing import Callable

def euler(y0: float, a: float, b: int, h: int) -> float:
    iter_ctr = 0
    t = a
    y = y0
    while True:
        # if t > b
        if (b - t) < 0:
            break
        if iter_ctr > 100:
            break
        # print("%6.3f %6.3f" % (t, y))
        t = t + h
        cooling_factor = y - 20
        newton_cooling = -0.07 * cooling_factor
        update_y = h * newton_cooling
        y = y + update_y
        iter_ctr = iter_ctr + 1
    return y
 
# def newtoncooling(time: float, temp: float) -> float:
# 	return 
 
def test_euler() -> None:
    res = euler(100,0,100,10)
    res = round(res, 3)
    assert res == 20

test_euler()