from typing import Tuple, List
import math
 
def hist(source: str) -> Tuple[int, dict[str, int]]:
    hist = {}; l = 0;
    for e in source:
        l += 1
        if e not in hist:
            hist[e] = 0
        hist[e] += 1
    return (l,hist)
 
def entropy(hist: dict[str, int], l: int) -> float:
    elist = []
    for v in hist.values():
        c = v / l
        elist.append(-c * math.log(c ,2))
    return sum(elist)
 
def printHist(h: dict[str, int]) -> None:
    flip = lambda x : (x[1], x[0])
    h_sorted: List[Tuple[str, int]] = sorted(h.items(), key = flip)
    print('Sym\thi\tfi\tInf')
    for (k,v) in h_sorted:
        print('%s\t%f\t%f\t%f'%(k,v,v/l,-math.log(v/l, 2)))
 
 
 
source = "1223334444"
(l,h) = hist(source)
print('.[Results].')
print('Length',l)
print('Entropy:', entropy(h, l))
printHist(h)