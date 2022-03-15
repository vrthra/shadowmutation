

from functools import reduce
import json
from pathlib import Path

MODES = [r'\mT', r'\mSS', r'\mMS', r'\mETNfNm', r'\mETNf', r'\mETFcNm', r'\mETFpNm', r'\mET']

hline = r"\hline"

def prog_cmd(name: str) -> str:
    name = name.replace('_', '')
    return fr"\pgm{name}"


def print_table(data, key, val_format):
    print(key)
    print("Program", end=' ')
    for mm in MODES:
        print(f'& {mm}', end=' ')
    print(r'\\')
    print(hline)

    for prog, prog_data in data.items():
        print(prog_cmd(prog), end=' ')
        prog_subj_count = prog_data[key]
        assert len(prog_subj_count) == len(MODES)
        for val in prog_subj_count:
            print(f'& {val:{val_format}}', end=' ')
        print(r'\\')

    print(hline)
    mean_data = []
    for _ in range(len(MODES)):
        mean_data.append([])

    for prog_data in data.values():
        for ii in range(len(MODES)):
            mean_data[ii].append(prog_data[key][ii])

    # print(mean_data)

    max_val = max(sum(dd) for dd in mean_data)
    # print(max_val)
    print(fr"Mean $T\times$", end=' ')
    for dd in mean_data:
        print(f"& {sum(dd) / max_val:0.2f}", end=' ')
    print(r'\\')

    max_geo_val = max(reduce(lambda a, b: a * b, dd) for dd in mean_data)**(1/len(mean_data))
    # print(max_geo_val)
    print(fr"Geo Mean $T\times$", end=' ')
    for dd in mean_data:
        print(f"& {(reduce(lambda a, b: a * b, dd)**(1/len(mean_data))) / max_geo_val:.2f}", end=' ')
    print(r'\\')





def main() -> None:
    data = {}
    for res in Path('tmp').glob('*/res.json'):
        subj = res.parent.name
        with open(res, 'rt') as f:
            data[subj] = json.load(f)

    print("="*80)
    print_table(data, 'subj_count', ',d')
    print("="*80)
    print_table(data, 'tool_count', ',d')
    print("="*80)
    print_table(data, 'runtime', '.2f')
    print("="*80)


if __name__ == "__main__":
    main()