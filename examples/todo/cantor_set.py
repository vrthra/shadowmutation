from typing import List, Union, Any

WIDTH = 81
HEIGHT = 5
 
results: List[Any] = []
lines: List[str] = []
def cantor(start: int, len: int, index: int) -> None:
    results.append((start, len, index))
    seg = len / 3
    if seg == 0:
        return None
    for it in range(HEIGHT-index):
        i = index + it
        for jt in range(int(seg)):
            j = start + seg + jt
            pos = i * WIDTH + j
            lines[int(pos)] = ' '
    cantor(int(start),           int(seg), index + 1)
    cantor(int(start + seg * 2), int(seg), index + 1)
    return None
 
lines = ['*'] * (WIDTH*HEIGHT)
cantor(0, WIDTH, 1)
 
for i in range(HEIGHT):
    beg = WIDTH * i
    print(''.join(lines[beg : beg+WIDTH]))

print(results)

[(0, 81, 1), (0, 27, 2), (0, 9, 3), (0, 3, 4), (0, 1, 5), (0, 0, 6), (0, 0, 6), (2, 1, 5), (2, 0, 6), (2, 0, 6),
 (6, 3, 4), (6, 1, 5), (6, 0, 6), (6, 0, 6), (8, 1, 5), (8, 0, 6), (8, 0, 6), (18, 9, 3), (18, 3, 4), (18, 1, 5),
 (18, 0, 6), (18, 0, 6), (20, 1, 5), (20, 0, 6), (20, 0, 6), (24, 3, 4), (24, 1, 5), (24, 0, 6), (24, 0, 6),
 (26, 1, 5), (26, 0, 6), (26, 0, 6), (54, 27, 2), (54, 9, 3), (54, 3, 4), (54, 1, 5), (54, 0, 6), (54, 0, 6),
 (56, 1, 5), (56, 0, 6), (56, 0, 6), (60, 3, 4), (60, 1, 5), (60, 0, 6), (60, 0, 6), (62, 1, 5), (62, 0, 6),
 (62, 0, 6), (72, 9, 3), (72, 3, 4), (72, 1, 5), (72, 0, 6), (72, 0, 6), (74, 1, 5), (74, 0, 6), (74, 0, 6),
 (78, 3, 4), (78, 1, 5), (78, 0, 6), (78, 0, 6), (80, 1, 5), (80, 0, 6), (80, 0, 6)
]