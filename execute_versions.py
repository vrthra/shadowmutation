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
    res = run(['python3', path], stdout=PIPE, stderr=STDOUT, env=env)
    if should_print:
        print(f"{res.args} => {res.returncode}")
        print(res.stdout.decode())
    return res


def get_res(path, mode):
    res_path = Path('res.json')
    res_path.unlink(missing_ok=True)
    run_it(path, mode=mode, result_file=res_path)
    try:
        with open(res_path, 'rt') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'strong': ['error'], 'execution_mode': mode}


def shadow_res(path, mut_ids):
    strongly_killed = set()
    mut_ids = set(mut_ids)
    while mut_ids:
        m_id = mut_ids.pop()
        res_path = Path('res.json')
        res_path.unlink(missing_ok=True)
        run_it(path, mode='shadow', logical_path=m_id, result_file=res_path)
        try:
            with open(res_path, 'rt') as f:
                results = json.load(f)
        except FileNotFoundError:
            return {'strong': ['error'], 'execution_mode': 'shadow'}
        assert results['execution_mode'] == 'SHADOW'
        strongly_killed |= set(results['strong'])
        mut_ids -= strongly_killed
    return {'strong': list(strongly_killed), 'execution_mode': 'SHADOW'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help="Path to ast_mutator result dir.")
    args = parser.parse_args()

    run_it(Path(args.dir)/'original.py')


    mut_ids = []
    traditional_results = {'killed': [], 'alive': []}
    for path in sorted(list(Path(args.dir).glob("traditional_*.py"))):
        res = run_it(path)
        mut_id = int(path.stem[len('traditional_'):])
        mut_ids.append(mut_id)
        if res.returncode != 0:
            traditional_results['killed'].append(mut_id)
        else:
            traditional_results['alive'].append(mut_id)


    split_stream_results = get_res(Path(args.dir)/"split_stream.py",     'split')
    modulo_results =       get_res(Path(args.dir)/"split_stream.py",     'modulo')
    shadow_results =       get_res(Path(args.dir)/"shadow_execution.py", 'shadow')

    def get_sorted(data, key):
        return sorted(data[key])

    def get_mode(data):
        return data['execution_mode']

    trad_killed = get_sorted(traditional_results, 'killed')
    split_stream_killed = get_sorted(split_stream_results, 'strong')
    modulo_killed = get_sorted(modulo_results, 'strong')
    shadow_killed = get_sorted(shadow_results, 'strong')

    split_stream_mode = get_mode(split_stream_results)
    modulo_mode = get_mode(modulo_results)
    shadow_mode = get_mode(shadow_results)

    print("Traditional results: {traditional_results}")
    print("Comparing results:")
    print(trad_killed, "TRADITIONAL")
    print(split_stream_killed, split_stream_mode)
    print(modulo_killed, modulo_mode)
    print(shadow_killed, shadow_mode)

    assert trad_killed == split_stream_killed
    assert trad_killed == modulo_killed
    assert trad_killed == shadow_killed

    assert split_stream_mode == "SPLIT_STREAM"
    assert modulo_mode == "MODULO_EQV"
    assert shadow_mode == "SHADOW"




if __name__ == "__main__":
    main()