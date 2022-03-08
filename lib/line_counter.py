# Counting the lines executed in tooling and subject code.

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Tuple, Union

OLD_TRACE = sys.gettrace()

LAST_TRACED_LINE: Union[Tuple[str, int], None] = None
SUBJECT_COUNTER = 0
TOOL_COUNTER = 0
SUBJECT_COUNTER_DICT: Dict[Tuple[str, int], int] = {}

IGNORE_FILES = set([
    "__init__.py",
    "decoder.py",
    "encoder.py",
    "threading.py",
    "genericpath.py",
    "posixpath.py",
    "types.py",
    "enum.py",
    "copy.py",
    "abc.py",
    "os.py",
    "re.py",
    "copyreg.py",
    "warnings.py",
    "sre_compile.py",
    "sre_parse.py",
    "functools.py",
    "tempfile.py",
    "random.py",
    "pathlib.py",
    "codecs.py",
    "fnmatch.py",
    "typing.py",
    "_collections_abc.py",
    "_weakrefset.py",
    "_bootlocale.py",
    "<frozen importlib._bootstrap>",
    "<string>",
])


TOOL_FILES = set([
    "shadow.py",
    "cache.py",
    "fork.py",
    "func_wrap.py",
    "line_counter.py",
    "mode.py",
    "path.py",
    "shadow_variable.py",
])


def reset_lines() -> None:
    global SUBJECT_COUNTER
    global SUBJECT_COUNTER_DICT
    global TOOL_COUNTER
    SUBJECT_COUNTER = 0
    SUBJECT_COUNTER_DICT = defaultdict(int)
    TOOL_COUNTER = 0


def trace_func(frame: Any, event: Any, arg: Any) -> Any:
    global LAST_TRACED_LINE
    global SUBJECT_COUNTER
    global SUBJECT_COUNTER_DICT
    global TOOL_COUNTER
    global CACHE_PATH

    fname = frame.f_code.co_filename
    fname_sub = Path(fname)
    fname_sub_name = fname_sub.name

    if fname_sub_name in IGNORE_FILES:
        # logger.debug(f"ignored: {fname_sub.name} {frame.f_code.co_name} {frame.f_code.co_firstlineno}")
        return trace_func

    # frame.f_trace_opcodes = True

    # logger.debug(f"{dir(frame)}")
    # logger.debug(f"{frame}")
    if event != 'line':
        return trace_func

    if frame.f_code.co_name in [
        "tool_line_counting", "subject_line_counting", "t_gather_results", "disable_line_counting"
    ]:
        return trace_func

    is_subject_file = fname_sub.parent.parent.parent.name == "shadowmutation" and \
        fname_sub.parent.parent.name == "tmp"
    is_tool_file = fname_sub_name in TOOL_FILES
    assert not (is_subject_file and is_tool_file)
    if not (is_subject_file or is_tool_file):
        assert False, f"Unknown file: {fname}, add it to the top of shadow.py"


    if is_tool_file:
        # logger.debug(f"tool: {fname_sub.name} {frame.f_code.co_name} {frame.f_code.co_firstlineno}")
        TOOL_COUNTER += 1
    else:
        cur_line: Tuple[str, int] = (fname_sub.name, frame.f_lineno)
        # logger.debug(f"{cur_line} {LAST_TRACED_LINE}")
        if cur_line == LAST_TRACED_LINE:
            return trace_func
        LAST_TRACED_LINE = cur_line

        # logger.debug(f"subject: {fname_sub.name} {frame.f_code.co_name} {frame.f_lineno}")
        SUBJECT_COUNTER += 1
        SUBJECT_COUNTER_DICT[cur_line] += 1

    return trace_func


def reinit_trace() -> None:
    if os.environ.get("TRACE", "0") == "1":
        sys.settrace(trace_func)
        reset_lines()


def disable_line_counting() -> None:
    sys.settrace(OLD_TRACE)


def get_counter_results() -> dict:
    res = {}
    res['subject_count'] = SUBJECT_COUNTER
    res['subject_count_lines'] = sorted(SUBJECT_COUNTER_DICT.items(), key=lambda x: x[0])
    res['tool_count'] = TOOL_COUNTER
    return res


def add_subject_counter(count: int) -> None:
    global SUBJECT_COUNTER
    SUBJECT_COUNTER += count


def add_subject_counter_dict(counter_dict: list[tuple[str, int]]) -> None:
    global SUBJECT_COUNTER_DICT
    for k, v in counter_dict:
        file, line = k
        key: tuple[str, int] = (file, int(line))
        SUBJECT_COUNTER_DICT[key] += v


def add_tool_counter(count: int) -> None:
    global TOOL_COUNTER
    TOOL_COUNTER += count
