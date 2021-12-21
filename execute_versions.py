import os
import argparse
import json
from copy import deepcopy
from subprocess import run, PIPE, STDOUT
from collections import defaultdict
from pathlib import Path


def run_it(path, mode=None, logical_path=None, result_file=None, should_not_print=False):
    env = deepcopy(os.environ)
    if logical_path is not None:
        env['LOGICAL_PATH'] = str(logical_path)
    if result_file is not None:
        env['RESULT_FILE'] = str(result_file)
    if mode is not None:
        env['EXECUTION_MODE'] = mode
    env['TRACE'] = "1"
    env['GATHER_ATEXIT'] = '1'
    res = run(['python3', path], stdout=PIPE, stderr=STDOUT, env=env)
    if res.returncode != 0 and not should_not_print:
        print(f"{res.args} => {res.returncode}")
        print(res.stdout.decode())
    return res


def get_res(path, mode, should_not_print=False):
    res_path = Path('res.json')
    res_path.unlink(missing_ok=True)
    run_res = run_it(path, mode=mode, result_file=res_path, should_not_print=should_not_print)
    try:
        with open(res_path, 'rt') as f:
            res = json.load(f)
            res['exit_code'] = run_res.returncode
            # print(res)
            return res
    except FileNotFoundError:
        return {'strong': ['error'], 'execution_mode': mode, 'exit_code': run_res.returncode}


def extract_data(data):
    def get_sorted(data, key):
        return sorted(data[key])

    def get_mode(data):
        return data['execution_mode']

    def subj_count(data):
        return data['subject_count']

    def tool_count(data):
        return data['tool_count']

    def line_counts(data):
        return {d[0][1]: d[1] for d in data['subject_count_lines']}

    return get_sorted(data, 'strong'), get_mode(data), subj_count(data), tool_count(data), line_counts(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help="Path to ast_mutator result dir.")
    args = parser.parse_args()

    run_it(Path(args.dir)/'original.py')


    mut_ids = []
    subject_ctr = 0
    tool_ctr = 0
    subject_line_ctr = defaultdict(int)
    traditional_results = {'killed': [], 'alive': []}
    for path in sorted(list(Path(args.dir).glob("traditional_*.py"))):
        res = get_res(path, None, should_not_print=True)
        mut_id = int(path.stem[len('traditional_'):])
        mut_ids.append(mut_id)
        subject_ctr += res['subject_count']
        tool_ctr += res['tool_count']
        for k, v in res['subject_count_lines']:
            subject_line_ctr[k[1]] += v
        if res['exit_code'] != 0:
            traditional_results['killed'].append(mut_id)
        else:
            traditional_results['alive'].append(mut_id)
    trad_killed = sorted(traditional_results['killed'])
    print("Comparing results:")
    print(trad_killed, "TRADITIONAL", subject_ctr, tool_ctr)


    split_stream_killed, split_stream_mode, split_stream_subj_count, split_stream_tool_count, ss_line_count = \
        extract_data(get_res(Path(args.dir)/"split_stream.py",     'split'))
    print(split_stream_killed, split_stream_mode, split_stream_subj_count, split_stream_tool_count)

    modulo_killed, modulo_mode, modulo_subj_count, modulo_tool_count, modulo_line_count = \
        extract_data(get_res(Path(args.dir)/"split_stream.py",     'modulo'))
    print(modulo_killed, modulo_mode, modulo_subj_count, modulo_tool_count)

    shadow_killed, shadow_mode, shadow_subj_count, shadow_tool_count, shadow_line_count = \
        extract_data(get_res(Path(args.dir)/"shadow_execution.py", 'shadow'))
    print(shadow_killed, shadow_mode, shadow_subj_count, shadow_tool_count)

    sf_killed, sf_mode, sf_subj_count, sf_tool_count, sf_line_count = \
        extract_data(get_res(Path(args.dir)/"shadow_execution.py", 'shadow_fork'))
    print(sf_killed, sf_mode, sf_subj_count, sf_tool_count)


    all_lines = subject_line_ctr.keys() | ss_line_count.keys() | modulo_line_count.keys() | shadow_line_count.keys()
    trad_c_total = 0
    ss_c_total = 0
    modulo_c_total = 0
    shadow_c_total = 0
    sf_c_total = 0
    for ll in sorted(all_lines):
        trad_c = subject_line_ctr.get(ll, 0)
        ss_c = ss_line_count.get(ll, 0)
        modulo_c = modulo_line_count.get(ll, 0)
        shadow_c = shadow_line_count.get(ll, 0)
        sf_c = sf_line_count.get(ll, 0)

        trad_c_total += trad_c
        ss_c_total += ss_c
        modulo_c_total += modulo_c
        shadow_c_total += shadow_c
        sf_c_total += sf_c

        print(f"{ll}: {trad_c:10} {ss_c:10} {modulo_c:10} {shadow_c:10} {sf_c:10}")


    assert trad_killed == split_stream_killed
    assert trad_killed == modulo_killed
    assert trad_killed == shadow_killed

    assert split_stream_mode == "SPLIT_STREAM"
    assert modulo_mode == "MODULO_EQV"
    assert shadow_mode == "SHADOW"

    assert trad_c_total == subject_ctr
    assert ss_c_total == split_stream_subj_count
    assert modulo_c_total == modulo_subj_count
    assert shadow_c_total == shadow_subj_count
    assert sf_c_total == sf_subj_count




if __name__ == "__main__":
    main()