import os
import argparse
import json
from copy import deepcopy
from subprocess import run, PIPE, STDOUT
from pathlib import Path


def run_it(path, mode=None, logical_path=None, result_file=None, should_print=False):
    env = deepcopy(os.environ)
    if logical_path is not None:
        env['LOGICAL_PATH'] = str(logical_path)
    if result_file is not None:
        env['RESULT_FILE'] = str(result_file)
    if mode is not None:
        env['EXECUTION_MODE'] = mode
    env['TRACE'] = "1"
    res = run(['python3', path], stdout=PIPE, stderr=STDOUT, env=env)
    if res.returncode != 0 or should_print:
        print(f"{res.args} => {res.returncode}")
        print(res.stdout.decode())
    return res


def get_res(path, mode):
    res_path = Path('res.json')
    res_path.unlink(missing_ok=True)
    run_res = run_it(path, mode=mode, result_file=res_path)
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

    return get_sorted(data, 'strong'), get_mode(data), subj_count(data), tool_count(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help="Path to ast_mutator result dir.")
    args = parser.parse_args()

    run_it(Path(args.dir)/'original.py')


    mut_ids = []
    subject_ctr = 0
    tool_ctr = 0
    traditional_results = {'killed': [], 'alive': []}
    for path in sorted(list(Path(args.dir).glob("traditional_*.py"))):
        res = get_res(path, None)
        mut_id = int(path.stem[len('traditional_'):])
        mut_ids.append(mut_id)
        # print(res)
        subject_ctr += res['subject_count']
        tool_ctr += res['tool_count']
        if res['exit_code'] != 0:
            traditional_results['killed'].append(mut_id)
        else:
            traditional_results['alive'].append(mut_id)
    trad_killed = sorted(traditional_results['killed'])
    print("Comparing results:")
    print(trad_killed, "TRADITIONAL", subject_ctr, tool_ctr)


    split_stream_killed, split_stream_mode, split_stream_subj_count, split_stream_tool_count = \
        extract_data(get_res(Path(args.dir)/"split_stream.py",     'split'))
    print(split_stream_killed, split_stream_mode, split_stream_subj_count, split_stream_tool_count)

    modulo_killed, modulo_mode, modulo_subj_count, modulo_tool_count = \
        extract_data(get_res(Path(args.dir)/"split_stream.py",     'modulo'))
    print(modulo_killed, modulo_mode, modulo_subj_count, modulo_tool_count)

    shadow_killed, shadow_mode, shadow_subj_count, shadow_tool_count = \
        extract_data(get_res(Path(args.dir)/"shadow_execution.py", 'shadow'))
    print(shadow_killed, shadow_mode, shadow_subj_count, shadow_tool_count)


    assert trad_killed == split_stream_killed
    assert trad_killed == modulo_killed
    assert trad_killed == shadow_killed

    assert split_stream_mode == "SPLIT_STREAM"
    assert modulo_mode == "MODULO_EQV"
    assert shadow_mode == "SHADOW"




if __name__ == "__main__":
    main()