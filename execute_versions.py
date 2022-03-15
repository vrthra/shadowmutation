import os
import argparse
import json
import tempfile
import time
from copy import deepcopy
from subprocess import CompletedProcess, run, PIPE, STDOUT
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def run_it(path: Path, trace: bool, mode: Optional[str]=None, logical_path: Optional[str]=None, result_file: Optional[Path]=None, should_not_print: bool=False, timeout: Optional[int]=None) -> CompletedProcess[bytes]:
    # print(path)
    env = deepcopy(os.environ)

    if logical_path is not None:
        env['LOGICAL_PATH'] = str(logical_path)

    if result_file is not None:
        env['RESULT_FILE'] = str(result_file)

    if mode is not None:
        env['EXECUTION_MODE'] = mode

    if trace:
        env['TRACE'] = "1"
    else:
        pass

    env['GATHER_ATEXIT'] = '1'

    res = run(['python3', path], stdout=PIPE, stderr=STDOUT, env=env, timeout=timeout)
    if res.returncode != 0 and not should_not_print:
        print(f"{res.args} => {res.returncode}")
        print(res.stdout.decode())
    return res


def get_res_inner(path: Path, mode: str, trace: bool, should_not_print: bool=False, timeout: Optional[int]=None) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(mode='rt') as f:
        res_path = Path(f.name)
        start = time.time()
        run_res = run_it(path, trace, mode=mode, result_file=res_path, should_not_print=should_not_print, timeout=timeout)
        end = time.time()

        try:
            with open(res_path, 'rt') as f:
                res: dict[str, Any] = json.load(f)
                res['exit_code'] = run_res.returncode
                res['runtime'] = end - start
                return res
        except FileNotFoundError:
            return {'strong': ['error'], 'execution_mode': mode, 'exit_code': run_res.returncode, 'out': run_res.stdout.decode()}


def get_res(path: Path, mode: str, should_not_print: bool=False, timeout: Optional[int]=None) -> dict[str, Any]:
    trace_res = get_res_inner(path, mode, True, should_not_print, timeout)
    time_res = get_res_inner(path, mode, False, should_not_print, timeout)

    combined_res = {}
    for kk in ['execution_mode', 'strong', 'exit_code']:
        assert trace_res[kk] == time_res[kk], f"\n{trace_res}\n{time_res}"
        combined_res[kk] = trace_res[kk]

    combined_res['subject_count'] = trace_res['subject_count']
    combined_res['subject_count_lines'] = trace_res['subject_count_lines']
    combined_res['tool_count'] = trace_res['tool_count']
    combined_res['tool_count_lines'] = trace_res['tool_count_lines']

    combined_res['runtime'] = time_res['runtime']
    return combined_res


def extract_data(data: dict[str, Any]) -> Any:
    def get_sorted(data, key):
        return sorted(data[key])

    def get_mode(data):
        return data['execution_mode']

    def subj_count(data):
        return data['subject_count']

    def tool_count(data):
        return data['tool_count']

    def subject_line_counts(data):
        return {d[0][1]: d[1] for d in data['subject_count_lines']}

    def tool_line_counts(data):
        return {f"{d[0][0]}:{d[0][1]}": d[1] for d in data['tool_count_lines']}
    
    def runtime(data):
        return data['runtime']

    if data.get('strong') == 'error':
        print(data)
        raise ValueError(data)

    return get_sorted(data, 'strong'), get_mode(data), subj_count(data), tool_count(data), subject_line_counts(data), tool_line_counts(data), float(runtime(data))


def tool_lines_to_file(path, trad_tool_line, ss_tool_line, mod_tool_line, s_tool_line, sc_tool_line, sf_c_tool_line, sf_p_tool_line, sfc_tool_line):
    all_lines = trad_tool_line.keys() | ss_tool_line.keys() | mod_tool_line.keys() | s_tool_line.keys() | sc_tool_line.keys() | sf_c_tool_line.keys() | sf_p_tool_line.keys() | sfc_tool_line.keys()
    trad_c_total = 0
    ss_c_total = 0
    modulo_c_total = 0
    shadow_c_total = 0
    shadow_cache_c_total = 0
    sf_c_total = 0
    sf_p_total = 0
    sfc_c_total = 0
    with open(path, 'wt') as f:
        f.write("trad_tool_line, ss_tool_line, mod_tool_line, s_tool_line, sc_tool_line, sf_c_tool_line, sf_p_tool_line, sfc_tool_line\n")
        for ll in sorted(all_lines):
            trad_c = trad_tool_line.get(ll, 0)
            ss_c = ss_tool_line.get(ll, 0)
            modulo_c = mod_tool_line.get(ll, 0)
            shadow_c = s_tool_line.get(ll, 0)
            shadow_cache_c = sc_tool_line.get(ll, 0)
            sf_c = sf_c_tool_line.get(ll, 0)
            sf_p = sf_p_tool_line.get(ll, 0)
            sfc_c = sfc_tool_line.get(ll, 0)

            trad_c_total += trad_c
            ss_c_total += ss_c
            modulo_c_total += modulo_c
            shadow_c_total += shadow_c
            shadow_cache_c_total += shadow_cache_c
            sf_c_total += sf_c
            sf_p_total += sf_p
            sfc_c_total += sfc_c

            f.write(f"{ll:>30}: {trad_c:10} {ss_c:10} {modulo_c:10} {shadow_c:10} {shadow_cache_c:10} {sf_c:10} {sf_p:10} {sfc_c:10}\n")

    print("Tool lines sum:")
    print(trad_c_total, ss_c_total, modulo_c_total, shadow_c_total, shadow_cache_c_total, sf_c_total, sf_p_total, sfc_c_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help="Path to ast_mutator result dir.")
    args = parser.parse_args()

    run_it(Path(args.dir)/'original.py', False)


    mut_ids = []
    trad_subj_count = 0
    trad_tool_count = 0
    trad_runtime = 0
    trad_subj_line = defaultdict(int)
    trad_tool_line = defaultdict(int)
    traditional_results = {'killed': [], 'alive': [], 'timeout': []}
    for path in sorted(list(Path(args.dir).glob("traditional_*.py"))):
        res = get_res(path, None, should_not_print=True, timeout=10)
        mut_id = int(path.stem[len('traditional_'):])
        mut_ids.append(mut_id)
        trad_subj_count += res['subject_count']
        trad_tool_count += res['tool_count']
        trad_runtime += res['runtime']
        for k, v in res['subject_count_lines']:
            trad_subj_line[k[1]] += v
        for k, v in res['tool_count_lines']:
            trad_tool_line[k[1]] += v
        if res['exit_code'] != 0:
            traditional_results['killed'].append(mut_id)
        else:
            traditional_results['alive'].append(mut_id)
    trad_killed = sorted(traditional_results['killed'])
    print("Comparing results:")
    print(trad_killed, "TRADITIONAL", trad_subj_count, trad_tool_count, f"{trad_runtime:.2f}")


    ss_killed, ss_mode, ss_subj_count, ss_tool_count, ss_subj_line, ss_tool_line, ss_runtime = \
        extract_data(get_res(Path(args.dir)/"split_stream.py",     'split'))
    print(ss_killed, ss_mode, ss_subj_count, ss_tool_count, f"{ss_runtime:.2f}")

    mod_killed, modulo_mode, mod_subj_count, mod_tool_count, mod_subj_line, mod_tool_line, mod_runtime = \
        extract_data(get_res(Path(args.dir)/"split_stream.py",     'modulo'))
    print(mod_killed, modulo_mode, mod_subj_count, mod_tool_count, f"{mod_runtime:.2f}")

    s_killed, shadow_mode, s_subj_count, s_tool_count, s_subj_line, s_tool_line, s_runtime = \
        extract_data(get_res(Path(args.dir)/"shadow_execution.py", 'shadow'))
    print(s_killed, shadow_mode, s_subj_count, s_tool_count, f"{s_runtime:.2f}")

    sc_killed, shadow_cache_mode, sc_subj_count, sc_tool_count, sc_subj_line, sc_tool_line, sc_runtime = \
        extract_data(get_res(Path(args.dir)/"shadow_execution.py", 'shadow_cache'))
    print(sc_killed, shadow_cache_mode, sc_subj_count, sc_tool_count, f"{sc_runtime:.2f}")

    sf_c_killed, sf_c_mode, sf_c_subj_count, sf_c_tool_count, sf_c_subj_line, sf_c_tool_line, sf_c_runtime = \
        extract_data(get_res(Path(args.dir)/"shadow_execution.py", 'shadow_fork_child'))
    print(sf_c_killed, sf_c_mode, sf_c_subj_count, sf_c_tool_count, f"{sf_c_runtime:.2f}")

    sf_p_killed, sf_p_mode, sf_p_subj_count, sf_p_tool_count, sf_p_subj_line, sf_p_tool_line, sf_p_runtime = \
        extract_data(get_res(Path(args.dir)/"shadow_execution.py", 'shadow_fork_parent'))
    print(sf_p_killed, sf_p_mode, sf_p_subj_count, sf_p_tool_count, f"{sf_p_runtime:.2f}")

    sfc_killed, sfc_mode, sfc_subj_count, sfc_tool_count, sfc_subj_line, sfc_tool_line, sfc_runtime = \
        extract_data(get_res(Path(args.dir)/"shadow_execution.py", 'shadow_fork_cache'))
    print(sfc_killed, sfc_mode, sfc_subj_count, sfc_tool_count, f"{sfc_runtime:.2f}")


    all_lines = trad_subj_line.keys() | ss_subj_line.keys() | mod_subj_line.keys() | sf_c_subj_line.keys()
    trad_c_total = 0
    ss_c_total = 0
    modulo_c_total = 0
    shadow_c_total = 0
    shadow_cache_c_total = 0
    sf_c_total = 0
    sf_p_total = 0
    sfc_c_total = 0
    for ll in sorted(all_lines):
        trad_c = trad_subj_line.get(ll, 0)
        ss_c = ss_subj_line.get(ll, 0)
        modulo_c = mod_subj_line.get(ll, 0)
        shadow_c = s_subj_line.get(ll, 0)
        shadow_cache_c = sc_subj_line.get(ll, 0)
        sf_c = sf_c_subj_line.get(ll, 0)
        sf_p = sf_p_subj_line.get(ll, 0)
        sfc_c = sfc_subj_line.get(ll, 0)

        trad_c_total += trad_c
        ss_c_total += ss_c
        modulo_c_total += modulo_c
        shadow_c_total += shadow_c
        shadow_cache_c_total += shadow_cache_c
        sf_c_total += sf_c
        sf_p_total += sf_p
        sfc_c_total += sfc_c

        print(f"{ll:4}: {trad_c:10} {ss_c:10} {modulo_c:10} {shadow_c:10} {shadow_cache_c:10} {sf_c:10} {sf_p:10} {sfc_c:10}")


    tool_lines_to_file(Path(args.dir)/'tool_lines.txt', trad_tool_line, ss_tool_line, mod_tool_line, s_tool_line, sc_tool_line, sf_c_tool_line, sf_p_tool_line, sfc_tool_line)


    assert trad_killed == ss_killed
    assert trad_killed == mod_killed
    assert trad_killed == s_killed
    assert trad_killed == sc_killed
    assert trad_killed == sf_c_killed
    assert trad_killed == sf_p_killed
    assert trad_killed == sfc_killed

    assert ss_mode == "SPLIT_STREAM"
    assert modulo_mode == "MODULO_EQV"
    assert shadow_mode == "SHADOW"
    assert shadow_cache_mode == "SHADOW_CACHE"
    assert sf_c_mode == "SHADOW_FORK_CHILD"
    assert sf_p_mode == "SHADOW_FORK_PARENT"
    assert sfc_mode == "SHADOW_FORK_CACHE"

    assert trad_c_total == trad_subj_count
    assert ss_c_total == ss_subj_count
    assert modulo_c_total == mod_subj_count
    assert shadow_c_total == s_subj_count
    assert shadow_cache_c_total == sc_subj_count
    assert sf_c_total == sf_c_subj_count, f"{sf_c_total}, {sf_c_subj_count}"
    assert sf_p_total == sf_p_subj_count, f"{sf_p_total}, {sf_p_subj_count}"
    assert sfc_c_total == sfc_subj_count

    data = {
        'mode':       ['traditional',   'split_stream', 'modulo_eqv',    'shadow',     'shadow_cache', 'shadow_fork_child', 'shadow_fork_parent', 'shadow_fork_cache'],
        'killed':     [trad_killed,      ss_killed,      mod_killed,      s_killed,     sc_killed,      sf_c_killed,         sf_p_killed,          sfc_killed],
        'subj_count': [trad_subj_count,  ss_subj_count,  mod_subj_count,  s_subj_count, sc_subj_count,  sf_c_subj_count,     sf_p_subj_count,      sfc_subj_count],
        'tool_count': [trad_tool_count,  ss_tool_count,  mod_tool_count,  s_tool_count, sc_tool_count,  sf_c_tool_count,     sf_p_tool_count,      sfc_tool_count],
        'runtime':    [trad_runtime,     ss_runtime,     mod_runtime,     s_runtime,    sc_runtime,     sf_c_runtime,        sf_p_runtime,         sfc_runtime],
        'subj_dict':  [trad_subj_line,   ss_subj_line,   mod_subj_line,   s_subj_line,  sc_subj_line,   sf_c_subj_line,      sf_p_subj_line,       sfc_subj_line],
        'tool_dict':  [trad_tool_line,   ss_tool_line,   mod_tool_line,   s_tool_line,  sc_tool_line,   sf_c_tool_line,      sf_p_tool_line,       sfc_tool_line],
    }

    with open(Path(args.dir)/'res.json', 'wt') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
