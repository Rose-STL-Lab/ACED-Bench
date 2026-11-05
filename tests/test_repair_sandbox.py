"""The analysis-code sandbox: normal analysis works; files outside the analysis folder, subprocesses, ctypes file
access, the environment, /proc secrets and UDP are refused; small variables carry over between code turns."""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "aced_repair"))
from code_sandbox import run_code  # noqa: E402

WORLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "repair", "test", "index.json")
CASES = [  # (name, code, expected substring in the output)
    ("describe wide", "print(experiment_1_df.describe())", "max"),
    ("read_csv bare name", "df=pd.read_csv('experiment_1_csv'); print(df.shape)", "(50, 15)"),
    ("read_csv file name", "df=pd.read_csv('experiment_1.csv'); print(df.shape)", "(50, 15)"),
    ("read_csv variable", "df=pd.read_csv(experiment_1_csv); print(df.shape)", "(50, 15)"),
    ("statsmodels", "import statsmodels.api as sm\nX=sm.add_constant(experiment_1_df[['Temp']])\n"
                    "print('r2', sm.OLS(experiment_1_df.Yield,X).fit().rsquared>-1)", "r2 True"),
    ("statsmodels formula", "import statsmodels.formula.api as smf\n"
                            "print(smf.ols('Yield ~ Temp', data=experiment_1_df).fit().params.shape)", "(2,)"),
    ("sklearn", "from sklearn.linear_model import LinearRegression\n"
                "print(LinearRegression().fit(experiment_1_df[['Temp']],experiment_1_df.Yield).coef_.shape)", "(1,)"),
    ("listdir .", "import os; print(sorted(os.listdir('.')))", "experiment_1.csv"),
    ("write+read local", "open('notes.txt','w').write('x'); print(open('notes.txt').read())", "x"),
    ("numeric idioms", "print(round(float(((experiment_1_df-experiment_1_df.mean())/experiment_1_df.std())"
                       ".abs().max().max()),0)>0)", "True"),
    ("environment scrubbed", "import os; print(sorted(k for k in os.environ if 'KEY' in k or 'TOKEN' in k or 'SECRET' in k))",
     "[]"),
    # refused
    ("listdir parent", "import os; print(os.listdir('..'))", "PermissionError"),
    ("open world index", f"print(open({WORLD!r}).read()[:40])", "PermissionError"),
    ("pandas read world", f"print(pd.read_json({WORLD!r}).shape)", "PermissionError"),
    ("glob home is empty", "import glob; print(glob.glob('/home/*/*'))", "[]"),
    ("subprocess", "import subprocess; print(subprocess.run(['ls','/'],capture_output=True).stdout[:30])", "not allowed"),
    ("os.system", "import os; os.system('ls /')", "not allowed"),
    ("ctypes libc fopen", f"import ctypes; libc=ctypes.CDLL(None); libc.fopen.restype=ctypes.c_void_p; "
                          f"print(libc.fopen({WORLD!r}.encode(), b'r'))", "not allowed"),
    ("ctypes pythonapi", "import ctypes; f=ctypes.pythonapi.fopen; print(f)", "not allowed"),
    *[(f"ctypes {sym}", f"import ctypes; f=getattr(ctypes.CDLL(None), {sym!r}); print(f({WORLD!r}.encode(), 0))",
       "not allowed") for sym in ("__open64_2", "__openat64_2", "open64", "__open_2", "creat")],
    ("/proc/self/environ", "print(open('/proc/self/environ','rb').read()[:20])", "PermissionError"),
    ("udp send", "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(b'x', ('127.0.0.1', 9))",
     "not allowed"),
]


@pytest.fixture(scope="module")
def csv_map():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "experiment_1.csv")
    rng = np.random.default_rng(0)
    cols = {"unit_id": [1000 + i for i in range(50)], "Yield": rng.random(50), "Temp": rng.random(50)}
    cols.update({f"ColumnNumber{i}": rng.random(50) for i in range(12)})
    pd.DataFrame(cols).to_csv(p, index=False)
    return {"experiment_1_csv": p}


@pytest.mark.parametrize("name,code,expect", CASES, ids=[c[0] for c in CASES])
def test_sandbox_case(csv_map, name, code, expect):
    out, _ = run_code(code, csv_map)
    assert expect in out, out


def test_carried_variables_and_cleanup(csv_map, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))   # the sandbox's working folders go here
    carried = {}
    for code in ("best = 5", "best = 7", "print('best', best)"):
        out, new_vars = run_code(code, csv_map, carried)
        carried.update(new_vars)
    assert "best 7" in out
    assert not [x for x in os.listdir(tmp_path) if x.startswith("aced_code_")]
