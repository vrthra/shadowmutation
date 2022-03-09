# Code for handling the different forking modes.

import os
import sys
import time
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable, Optional, TypeVar, Tuple, Union
import pickle
from json.decoder import JSONDecodeError
from lib.constants import MAINLINE

from lib.path import get_logical_path, get_masked_mutants, get_seen_mutants, merge_strongly_killed, set_logical_path, t_get_killed
from .mode import get_execution_mode
from .line_counter import add_subject_counter, add_subject_counter_dict, add_tool_counter, get_counter_results, reset_lines
from .shadow_variable import ShadowVariable, set_new_no_init, unset_new_no_init, get_active_shadow

import logging
logger = logging.getLogger(__name__)


def act_on_logging_handlers(callback: Callable[..., Any]) -> None:
    c: Optional[logging.Logger] = logger
    while c:
        if c.handlers:
            for h in c.handlers:
                try:
                    callback(h)
                except:
                    pass
            break
        if not c.propagate:
            break
        else:
            c = c.parent


_FORK_STYLE = "child_first"  # "parent_first" / "child_first"


def get_child_results(result_file: Path) -> Union[list[dict[str, Any]], None]:
    if result_file.is_file():
        execution_mode = get_execution_mode()

        with open(result_file, 'rb') as f:
            try:
                set_new_no_init()
                childrens_results: list[dict[str, Any]] = pickle.load(f)
            except JSONDecodeError:
                # Child has not yet written the results.
                return None
            finally:
                unset_new_no_init()

        # Convert results back into original types.
        for child_res in childrens_results:

            for res in ['strong', 'masked'] + ['seen'] if execution_mode.is_shadow_variant() else ['active']:
                child_res[res] = set(child_res[res])

            if not execution_mode.is_split_stream_variant():
                assert not (child_res['strong'] - child_res['seen']), f"{child_res}"

        return childrens_results

    return None


def merge_child_results(combined_fork_res: list[dict[str, Any]], childrens_results: list[dict[str, Any]]) -> None:
    for child_results in childrens_results:
        merge_strongly_killed(child_results['strong'])
        add_subject_counter(child_results['subject_count'])
        add_tool_counter(child_results['tool_count'])
        add_subject_counter_dict(child_results['subject_count_lines'])

        combined_fork_res.append(child_results)


class Forker():
    def __init__(self) -> None:
        self.started_paths = {}  # What paths have been forked off.
        self.parent_result_dir = None  # Where to write results.
        self.result_dir = None  # Where to get results of forked children.
        self._new_sync_dir()

    def __del__(self) -> None:
        (self.sync_dir/'results').rmdir()
        self.sync_dir.rmdir()

    def _new_sync_dir(self) -> None:
        self.sync_dir = Path(mkdtemp())

        self.parent_result_dir = self.result_dir
        self.result_dir = self.sync_dir/'results'
        self.result_dir.mkdir()

    def fork_child_first(self, path: int) -> bool:

        # Try to fork
        forked_pid = os.fork()

        if forked_pid == -1:
            # Error during forking. Not much we can do.
            raise ValueError(f"Could not fork for path: {path}!")
        elif forked_pid != 0:
            # We are in parent, record the child pid and path.
            self.started_paths[path] = forked_pid
            try:
                os.waitpid(forked_pid, 0)
            except ChildProcessError as e:
                if e.errno != 10:
                    logger.debug(f"{e}")

            return False
        else:
            # Clear path as the child has not started any paths.
            self.started_paths.clear()
            self._new_sync_dir()

            # Update which path child is supposed to follow
            set_logical_path(path)

            reset_lines()
            return True

    def maybe_fork(self, path: int) -> bool:
        # Only fork once, from then on follow that path.
        if path in self.started_paths:
            return False

        if _FORK_STYLE == "child_first":
            return self.fork_child_first(path)
        elif _FORK_STYLE == "parent_first":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown fork style: {_FORK_STYLE}")

    def _gather_results(self, fork_res: Any=None) -> dict[str, Any]:
        results = t_get_killed()
        for res in ['strong', 'masked'] + ['seen'] if get_execution_mode().is_shadow_variant() else ['active']:
            results[res] = list(results[res])
        results['pid'] = os.getpid()
        results['path'] = get_logical_path()
        results = {**results, **get_counter_results()}
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
        return results

    def wait_for_children_results(self) -> list[dict[str, Any]]:
        combined_fork_res: list[dict[str, Any]] = []
        while self.started_paths:
            path, child_pid = self.started_paths.popitem()

            while True:
                time.sleep(.01)

                try:
                    os.waitpid(child_pid, 0)
                except ChildProcessError as e:
                    if e.errno != 10:
                        logger.debug(f"{e}")

                result_file = self.result_dir/str(path)
                child_results = get_child_results(result_file)
                if child_results is not None:

                    merge_child_results(combined_fork_res, child_results)

                    result_file.unlink()
                    break
            
            # if is_done:
            #     break

        return combined_fork_res

    def child_end(self, fork_res: Any=None) -> Any:
        path = get_logical_path()
        assert path != MAINLINE
        assert self.parent_result_dir is not None

        if self.started_paths:
            child_results = self.wait_for_children_results()
        else:
            child_results = []

        results = self._gather_results(fork_res)

        res_path = self.parent_result_dir/str(path)
        with open(res_path, 'wb') as f:
            pickle.dump([results, *child_results], f)

        # OOS: exit the child immediately, this might cause problems for programs actually using multiprocessing
        # but this is a prototype
        act_on_logging_handlers(lambda h: h.flush())
        os._exit(0)

    def wait_for_forks(self, fork_res: Union[dict[int, Any], None]=None) -> tuple[Optional[dict[int, Any]], list[dict[str, Any]]]:
        global SUBJECT_COUNTER
        global TOOL_COUNTER

        # if child, write results and exit
        if self.parent_result_dir is not None:
            self.child_end(fork_res)

        if fork_res is not None:
            return_val, _, _ = fork_res
        else:
            return_val = None

        mainline_res = get_active_shadow(return_val, get_seen_mutants(), get_masked_mutants())

        # wait for all child processes to end
        combined_fork_res = self.wait_for_children_results()

        return mainline_res, combined_fork_res

    
_FORKING_CONTEXT: Union[Forker, None] = None


def reinit_forking_context() -> None:
    global _FORKING_CONTEXT
    if get_execution_mode().should_start_forker():
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
