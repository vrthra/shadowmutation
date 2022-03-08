# Code for handling the different forking modes.

import os
import sys
import time
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable, TypeVar, Tuple, Union
import pickle
from json.decoder import JSONDecodeError

from lib.path import get_logical_path, get_masked_mutants, get_seen_mutants, set_logical_path, t_get_killed
from .mode import get_execution_mode
from .line_counter import add_subject_counter, add_subject_counter_dict, add_tool_counter, get_counter_results, reset_lines
from .shadow_variable import ShadowVariable, set_new_no_init, unset_new_no_init, get_active_shadow

import logging
logger = logging.getLogger(__name__)


class Forker():
    def __init__(self) -> None:
        self.is_parent = True
        self.sync_dir = Path(mkdtemp())
        (self.sync_dir/'paths').mkdir()
        (self.sync_dir/'forks').mkdir()
        (self.sync_dir/'results').mkdir()

    def __del__(self) -> None:
        if self.is_parent:
            (self.sync_dir/'paths').rmdir()
            (self.sync_dir/'forks').rmdir()
            (self.sync_dir/'results').rmdir()
            self.sync_dir.rmdir()

    def my_pid(self) -> int:
        return os.getpid()

    def maybe_fork(self, path: int) -> bool:

        # Only fork once, from then on follow that path.
        path_file = self.sync_dir.joinpath('paths', str(path))
        if path_file.is_file():
            return False
        path_file.touch()

        # Try to fork
        forked_pid = os.fork()
        # logger.debug(f"Forking for path: {path} got pid: {forked_pid}")
        if forked_pid == -1:
            # Error during forking. Not much we can do.
            raise ValueError(f"Could not fork for path: {path}!")
        elif forked_pid != 0:
            # We are in parent, record the child pid and path.
            try:
                os.waitpid(forked_pid, 0)
            except ChildProcessError as e:
                if e.errno != 10:
                    logger.debug(f"{e}")
            path_file.write_text(str(forked_pid))
            # logger.debug(f"Waiting for {forked_pid} ended.")
            return False
        else:
            # Update that this is the child.
            self.is_parent = False

            # Update which path child is supposed to follow
            set_logical_path(path)
            forked_pid = self.my_pid()
            # fork_file = self.sync_dir.joinpath('forks', str(forked_pid)) # this is used to indicate when child can start
            # while not fork_file.is_file():
            #     # Wait until parent finishes.
            #     time.sleep(.1)
            # fork_file.unlink()

            # logger.debug(f"Child starting for path: {path}, with pid: {forked_pid}")
            reset_lines()
            return True

    def child_end(self, fork_res: Any=None) -> Any:
        assert not self.is_parent
        pid = self.my_pid()
        path = get_logical_path()
        # logger.debug(f"Child with pid: {pid} and path: {path} has reached sync point.")
        res_path = self.sync_dir/'results'/str(pid)
        # logger.debug(f"Writing results to: {res_path}")
        with open(res_path, 'wb') as f:
            results = t_get_killed()
            for res in ['strong', 'masked'] + ['seen'] if get_execution_mode().is_shadow_variant() else ['active']:
                results[res] = list(results[res])
            results['pid'] = pid
            results['path'] = path
            results = {**results, **get_counter_results()}
            # results['subject_count'] = SUBJECT_COUNTER
            # results['subject_count_lines'] = {'::'.join(str(a) for a in k): v for k, v in SUBJECT_COUNTER_DICT.items()}
            # results['tool_count'] = TOOL_COUNTER
            if fork_res is None:
                results['fork_res'] = None
            else:
                return_val, args, kwargs = fork_res
                assert type(return_val) != dict

                results_args = {}
                for ii, arg in enumerate(args):
                    if type(arg) == ShadowVariable:
                        results_args[ii] = arg

                results_kwargs = {}
                for kk, val in kwargs.items():
                    if type(val) == ShadowVariable:
                        results_kwargs[kk] = val

                results['fork_res'] = (return_val, results_args, results_kwargs)
            # logger.debug(f"child results to write: {results}")
            pickle.dump(results, f)

        # exit the child immediately, this might cause problems for programs actually using multiprocessing
        # but this is a prototype
        # logger.debug(f"Child with {pid} ended.")
        os._exit(0)

    def wait_for_forks(self, fork_res: Union[dict[int, Any], None]=None) -> list[Any]:
        global SUBJECT_COUNTER
        global TOOL_COUNTER
        # if child, write results and exit
        if not self.is_parent:
            self.child_end(fork_res)

        # wait for all child processes to end
        if fork_res is not None:
            return_val, _, _ = fork_res
        else:
            return_val = None

        combined_fork_res = [get_active_shadow(return_val, get_seen_mutants(), get_masked_mutants())]
        all_results = t_get_killed()
        while True:
            is_done = True
            for path_file in (self.sync_dir/'paths').glob("*"):
                is_done = False
                try:
                    child_pid = int(path_file.read_text())
                except ValueError:
                    continue

                # sync_pid_go_file = (self.sync_dir/'forks').joinpath(str(child_pid))
                # logger.debug(f"Waiting for pid: {child_pid} {sync_pid_go_file}")

                # Signal that child can start.
                # sync_pid_go_file.touch()
                execution_mode = get_execution_mode()
                while True:
                    time.sleep(.01)

                    try:
                        os.waitpid(child_pid, 0)
                    except ChildProcessError as e:
                        if e.errno != 10:
                            logger.debug(f"{e}")

                    result_file = self.sync_dir/'results'/str(child_pid)
                    if result_file.is_file():
                        with open(result_file, 'rb') as f:
                            try:
                                set_new_no_init()
                                child_results = pickle.load(f)
                            except JSONDecodeError:
                                # Child has not yet written the results.
                                continue
                            finally:
                                unset_new_no_init()

                        for res in ['strong', 'masked'] + ['seen'] if execution_mode.is_shadow_variant() else ['active']:
                            child_results[res] = set(child_results[res])

                        if execution_mode.is_split_stream_variant():
                            for res in ['strong']:
                                add_res = child_results[res] & child_results['active']
                                all_results[res] |= add_res
                        else:
                            assert not (child_results['strong'] - child_results['seen']), f"{child_results}"

                        all_results['strong'] |= child_results['strong']

                        # logger.debug(f"child results: {child_results}")
                        add_subject_counter(child_results['subject_count'])
                        add_tool_counter(child_results['tool_count'])
                        add_subject_counter_dict(child_results['subject_count_lines'])

                        combined_fork_res.append(child_results)

                        path_file.unlink()
                        result_file.unlink()
                        break
            
            if is_done:
                break
        return combined_fork_res

    
_FORKING_CONTEXT: Union[Forker, None] = None


def reinit_forking_context() -> None:
    global _FORKING_CONTEXT
    if get_execution_mode().should_start_forker():
        # logger.debug("Initializing forker")
        _FORKING_CONTEXT = Forker()
    else:
        _FORKING_CONTEXT = None


def get_forking_context() -> Union[Forker, None]:
    return _FORKING_CONTEXT


def new_forking_context() -> Forker:
    global _FORKING_CONTEXT
    _FORKING_CONTEXT = Forker()
    return _FORKING_CONTEXT


def set_forking_context(forker: Forker) -> None:
    global _FORKING_CONTEXT
    _FORKING_CONTEXT = forker
