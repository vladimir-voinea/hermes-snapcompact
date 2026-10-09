"""Minimal runner for the snapcompact test suite (venv has no pytest by design).

Usage: venv/bin/python plugins/context_engine/snapcompact/tests/run_tests.py
"""

import inspect
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import test_snapcompact as ts  # noqa: E402


def make_engine():
    eng = ts.SnapCompactEngine()
    eng.update_model(ts.MODEL, ts.CTX, provider="glm53-flash")
    eng._vision = True
    return eng


FIXTURES = {"engine": make_engine}


def main() -> int:
    tests = [(n, f) for n, f in inspect.getmembers(ts, inspect.isfunction)
             if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        kwargs = {}
        try:
            for pname in inspect.signature(fn).parameters:
                if pname in FIXTURES:
                    kwargs[pname] = FIXTURES[pname]()
                else:
                    raise RuntimeError(f"no fixture for parameter {pname!r}")
            fn(**kwargs)
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            failed.append(name)
            print(f"FAIL {name}")
            traceback.print_exc(limit=6)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
