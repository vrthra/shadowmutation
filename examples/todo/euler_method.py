from typing import Callable

def euler(f: Callable[[float, float], float], y0: int, a: int, b: int, h: int) -> None:
    t: float = float(a)
    y: float = float(y0)
    while t <= b:
        print("%6.3f %6.3f" % (t, y))
        t += h
        y += h * f(t, y)
 
def newtoncooling(time: float, temp: float) -> float:
	return -0.07 * (temp - 20)
 
euler(newtoncooling,100,0,100,10)