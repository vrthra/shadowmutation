import os
import argparse
import json
from copy import deepcopy
from subprocess import run, PIPE, STDOUT
from pathlib import Path


def run_it(path, mode=None, result_file=None, should_print=False):
    print(path)
    env = deepcopy(os.environ)
    if result_file is not None:
        env['RESULT_FILE'] = str(result_file)
    if mode is not None:
        env['EXECUTION_MODE'] = mode
    res = run(['python3', path], stdout=PIPE, stderr=STDOUT, env=env)
    print(f"{res.args} => {res.returncode}")
    if should_print:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help="Path to ast_mutator result dir.")
    args = parser.parse_args()

    run_it(Path(args.dir)/'original.py')

    traditional_results = {'killed': [], 'alive': []}
    for path in sorted(list(Path(args.dir).glob("traditional_*.py"))):
        res = run_it(path)
        mut_id = int(path.stem[len('traditional_'):])
        if res.returncode != 0:
            traditional_results['killed'].append(mut_id)
        else:
            traditional_results['alive'].append(mut_id)



    split_stream_results = get_res(Path(args.dir)/"split_stream.py",     'split')
    modulo_results =       get_res(Path(args.dir)/"split_stream.py",     'modulo')
    shadow_results =       get_res(Path(args.dir)/"shadow_execution.py", 'shadow')
    shadow_fork_results =  get_res(Path(args.dir)/"shadow_execution.py", 'shadow_fork')

    print(traditional_results)
    print(sorted(traditional_results['killed']))
    print(sorted(split_stream_results['strong']), split_stream_results['execution_mode'])
    print(sorted(modulo_results['strong']), modulo_results['execution_mode'])
    print(sorted(shadow_results['strong']), shadow_results['execution_mode'])
    print(sorted(shadow_fork_results['strong']), shadow_fork_results['execution_mode'])




if __name__ == "__main__":
    main()