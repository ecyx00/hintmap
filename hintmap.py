"""Thin shim: the code lives in hm/ (see README Layout). Re-exports what bench/swe.py uses; main() keeps the hook fast path off the argparse/cli import."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from hm.retrieve import retrieve  # noqa: F401  (bench/swe.py)

def main(argv=None):
    if argv is None and sys.argv[1:2] == ["hook"] and len(sys.argv) == 3 and sys.argv[2] in ("pretooluse-agent", "session-start", "subagent-start"):
        from hm.common import argparse; from hm.hook import cmd_hook
        return cmd_hook(argparse.Namespace(event=sys.argv[2]))  # hook fast path: no parser construction
    if argv is None and sys.argv[1:3] == ["hook", "pretooluse-bash"] and len(sys.argv) == 3:
        from hm.jit import cmd_jit_hook; return cmd_jit_hook()
    from hm.cli import main as m; return m(argv)
