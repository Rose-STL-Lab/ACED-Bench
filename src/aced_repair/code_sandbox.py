#!/usr/bin/env python3
"""Runs the agent's analysis code in a separate process over its experiment CSVs.

The code runs in a fresh temporary folder that holds only copies of this episode's CSVs, and a Python audit
hook refuses file and directory access outside that folder and the Python installation (the world files, which
hold the answers, are not reachable from agent code). Pandas prints wide tables in full, and
pd.read_csv('experiment_1_csv') / pd.read_csv('experiment_1.csv') both read the right file.

The worker also clears the environment to a small allowlist, refuses ctypes lookups of libc/Python entry points that
open files, run programs, open sockets or read the environment, refuses /proc/<pid>/{environ,mem,pagemap}, and refuses
UDP sends and binds, so agent code cannot reach the world file (which holds the answers) below the audit hook.
"""
from __future__ import annotations

import multiprocessing as mp
import re
from typing import Any, Dict, Optional, Tuple

CODE_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 10000
_MAX_VAR_PICKLE_BYTES = 200_000
_MAX_NDARRAY_ELEMS = 200_000
_FENCE_RE = re.compile(r"^```(?:python)?\s*|\s*```$", re.MULTILINE)
# environment variables agent code may see (everything else, e.g. API keys, is removed in the worker)
_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "HOME", "TMPDIR", "PYTHONHASHSEED", "OMP_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MPLCONFIGDIR", "PYTHONPATH")
_PROC_SECRET_RE = re.compile(r"^/proc/(?:\d+|self|thread-self)(?:/task/\d+)?/(?:environ|mem|pagemap)$")
# libc / Python C-API entry points that reach files, programs, sockets or the environment (full-name match);
# libraries that use ctypes for other symbols (numpy, threadpoolctl / sklearn BLAS lookups) are unaffected
_DENY_SYM = re.compile(r"^_*(?:open\w*|fopen\w*|freopen\w*|fdopen\w*|"
                       r"creat\w*|syscall|name_to_handle_at|open_by_handle_at|opendir|fdopendir|scandir|"
                       r"dlopen|dlmopen|dlsym|dlvsym|system|popen|exec\w*|fork|vfork|clone|clone3|posix_spawnp?|"
                       r"socket|socketpair|connect|sendto|sendmsg|bind|getenv|secure_getenv|environ|"
                       r"PyRun_\w*|Py_RunMain|PyImport_\w*|PySys_\w*|_Py_\w*)$")


def _worker(code: str, csv_map: Dict[str, str], carried: Dict[str, Any], result_q, wd: str) -> None:
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aced_mplconfig")
    import posix
    keep = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    try:
        posix.environ.clear()            # the startup snapshot
    except Exception:
        pass
    import contextlib
    import io
    import pickle
    import shutil
    import sys
    import traceback

    import numpy as _np
    import pandas as _pd
    try:
        import scipy.stats as _stats
    except Exception:
        _stats = None

    # a private working folder (created and removed by the parent) with copies of this episode's CSVs only
    local: Dict[str, str] = {}
    for var_name, path in csv_map.items():
        base = os.path.basename(path)
        dst = os.path.join(wd, base)
        try:
            shutil.copyfile(path, dst)
            local[var_name] = dst
        except Exception:
            pass
    os.chdir(wd)
    _pd.set_option("display.width", 250)
    _pd.set_option("display.max_columns", 80)
    _pd.set_option("display.max_rows", 200)
    _pd.set_option("display.max_colwidth", 60)

    _orig_read_csv = _pd.read_csv
    alias = {}
    for var_name, dst in local.items():
        alias[var_name] = dst                                   # experiment_1_csv
        alias[os.path.basename(dst)] = dst                      # experiment_1.csv
        alias[var_name[:-4] + "_df" if var_name.endswith("_csv") else var_name] = dst
        alias[(var_name[:-4] + "_df" if var_name.endswith("_csv") else var_name) + ".csv"] = dst   # experiment_1_df.csv

    def _read_csv(filepath_or_buffer, *a, **k):
        if isinstance(filepath_or_buffer, str) and os.path.basename(filepath_or_buffer) in alias:
            filepath_or_buffer = alias[os.path.basename(filepath_or_buffer)]
        return _orig_read_csv(filepath_or_buffer, *a, **k)
    _pd.read_csv = _read_csv

    ns: Dict[str, Any] = {"pd": _pd, "np": _np, "stats": _stats}
    for var_name, dst in local.items():
        ns[var_name] = dst
        df_name = var_name[:-4] + "_df" if var_name.endswith("_csv") else var_name + "_df"
        try:
            ns[df_name] = _orig_read_csv(dst)
        except Exception:
            pass
    # names we provide (not carried variables: a reassigned carried variable must be saved again)
    injected = set(ns.keys())
    ns.update(carried)

    # file-system guard: agent code may read/write only inside wd (plus the Python installation for imports)
    allowed = [os.path.realpath(wd)]
    for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(os.__file__), "/dev/null",
              "/proc/self", "/sys/devices/system/cpu", "/tmp/aced_mplconfig", "/usr/share/zoneinfo",
              "/etc/localtime"}:
        if p:
            allowed.append(os.path.realpath(p))

    def _ok(path) -> bool:
        if isinstance(path, int):
            return True
        try:
            rp = os.path.realpath(os.fsdecode(path))
        except Exception:
            return False
        return any(rp == a or rp.startswith(a.rstrip("/") + "/") for a in allowed)

    def _guard(event, args):
        if event == "open" and args and not _ok(args[0]):
            raise PermissionError(f"file access outside the analysis folder is not allowed: {args[0]}")
        if event in ("os.listdir", "os.scandir", "os.chdir", "os.remove", "os.rename", "shutil.rmtree",
                     "os.mkdir", "os.rmdir") and args and args[0] not in (None, "", ".") and not _ok(args[0]):
            raise PermissionError(f"file access outside the analysis folder is not allowed: {args[0]}")
        if event in ("subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "socket.connect",
                     "socket.sendto", "socket.sendmsg", "socket.bind"):
            raise PermissionError(f"{event} is not allowed in analysis code")
        if event in ("ctypes.dlsym", "ctypes.dlsym/handle"):
            name = next((os.fsdecode(a) for a in (args or ()) if isinstance(a, (str, bytes))), "")
            if _DENY_SYM.match(name):
                raise PermissionError("this ctypes call is not allowed in analysis code")
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            try:
                raw = os.fsdecode(args[0])
                rp = os.path.realpath(raw)
            except Exception:
                raw = rp = ""
            if _PROC_SECRET_RE.match(rp) or _PROC_SECRET_RE.match(raw):
                raise PermissionError(f"file access outside the analysis folder is not allowed: {args[0]}")
    sys.addaudithook(_guard)

    buf = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
    except Exception:
        error = traceback.format_exc()

    out = buf.getvalue()
    if error:
        # no host paths in what the agent sees
        error = re.sub(r'File "[^"]*?/(?:site-packages|lib/python3\.\d+)/', 'File "<python>/', error)
        error = re.sub(r'File "[^"]*code_sandbox\.py"', 'File "<sandbox>"', error)
        error = error.replace(wd, "<analysis folder>")
        out += f"\n[PYTHON ERROR]\n{error}"

    new_vars: Dict[str, Any] = {}
    for k, v in ns.items():
        if k in injected or k.startswith("__"):
            continue
        if isinstance(v, (_pd.DataFrame, _pd.Series)):
            continue
        if isinstance(v, _np.ndarray) and v.size > _MAX_NDARRAY_ELEMS:
            continue
        try:
            if len(pickle.dumps(v)) <= _MAX_VAR_PICKLE_BYTES:
                new_vars[k] = v
        except Exception:
            continue
    result_q.put((out, new_vars))


def run_code(code: str, csv_map: Dict[str, str], carried: Optional[Dict[str, Any]] = None,
             *, timeout: int = CODE_TIMEOUT_SECONDS,
             max_chars: int = MAX_OUTPUT_CHARS) -> Tuple[str, Dict[str, Any]]:
    import shutil
    import tempfile
    code = _FENCE_RE.sub("", code).replace("```", "").strip()
    carried = carried or {}
    wd = tempfile.mkdtemp(prefix="aced_code_")
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(code, csv_map, carried, result_q, wd))
    proc.start()
    import queue as _queue
    try:
        try:
            out, new_vars = result_q.get(timeout=timeout)
        except _queue.Empty:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                return (f"[TIMEOUT: execution exceeded {timeout}s]", {})
            return ("[NO OUTPUT: worker died without returning a result]", {})
        except Exception:
            proc.join(timeout=5)
            return ("[NO OUTPUT: worker died without returning a result]", {})
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    if len(out) > max_chars:
        # keep a Python error that follows long output
        i = out.find("\n[PYTHON ERROR]\n")
        err = out[i:] if i >= 0 else ""
        body = out[:i] if i >= 0 else out
        out = (body[:max_chars] + f"\n[OUTPUT CUT: your code printed {len(body)} characters; only the first "
               f"{max_chars} are shown. Print less (e.g. summaries or .head()) to see everything.]" + err)
    return (out, new_vars)
