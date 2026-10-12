"""hintmap.cli: see README Module ownership."""
import array, datetime, contextlib, functools, hashlib, heapq, itertools, json, math, os, re, signal, sqlite3, struct, sys, time
from hm.common import HOME, SELF, argparse, glob, shutil, subprocess
from hm.graph import cmd_graph_editcheck, cmd_graph_impact, cmd_graph_tests
from hm.hook import cmd_bench_hook, cmd_graph_replay, cmd_hook
from hm.report import cmd_report
from hm.jit import cmd_def, cmd_jit_hook, cmd_refs, cmd_show
from hm.nav import cmd_callees, cmd_changed, cmd_map, cmd_q, cmd_search
from hm.index import build_cmd, cmd_graph_audit, cmd_graph_init, cmd_graph_model, cmd_graph_stats
from hm.selftest import selftest

def du(p):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs if os.path.exists(os.path.join(d, f)))
def cmd_doctor(a):
    """Health report: graph registry, hook log, disk usage. Writes nothing unless --fix: then drops registry roots that vanished."""
    home = HOME(); mb = lambda n: f"{n / 1e6:.1f} MB"; warn = 0; fix = []  # fix: (kind, path)
    print(f"home: {home}  ({mb(du(home)) if os.path.isdir(home) else 'missing'})")
    try: reg = [r for r in open(os.path.join(home, "graphs")).read().splitlines() if r]
    except OSError: reg = []
    for r in reg:
        g = os.path.join(r, ".hintmap"); gone = not os.path.isdir(r); warn += gone; gone and fix.append(("unreg", r))
        print(f"  graph {r}  {'VANISHED root (remove from ~/.hintmap/graphs)' if gone else mb(du(g)) if os.path.isdir(g) else 'no .hintmap'}")
    lg = []
    try: lg = open(os.path.join(home, "injections.jsonl")).read().splitlines()[-20:]
    except OSError: pass
    rows = []
    for l in lg:
        try: rows.append(json.loads(l))
        except ValueError: pass
    ms = sorted(r.get("ms", 0) for r in rows); sk = {}
    for r in rows: sk[r.get("skipped") or "ok"] = sk.get(r.get("skipped") or "ok", 0) + 1
    print(f"  hook log (last {len(rows)}): " + (f"p95={ms[min(len(ms) - 1, math.ceil(.95 * len(ms)) - 1)]} ms, " + ", ".join(f"{k}={v}" for k, v in sorted(sk.items())) if rows else "empty"))
    print(f"doctor: {'OK' if not warn else f'{warn} warning(s)'}")
    if not a.fix: return print(f"  ({len(fix)} fixable: run `hintmap doctor --fix`)" if fix else "") or 0
    for kind, p in fix:
        keep = [r for r in open(os.path.join(home, "graphs")).read().splitlines() if r and r != p]; open(os.path.join(home, "graphs"), "w").write("".join(r + "\n" for r in keep)); print(f"fixed: dropped vanished root from the registry: {p}")
    print(f"doctor --fix: {len(fix)} change(s)" if fix else "doctor --fix: nothing to fix")
PKGS = ["tree-sitter", "tree-sitter-python", "tree-sitter-typescript", "tree-sitter-javascript", "tree-sitter-swift", "tree-sitter-haskell", "tree-sitter-zig", "tree-sitter-c", "tree-sitter-cpp", "tree-sitter-c-sharp",
        "tree-sitter-java", "tree-sitter-go", "tree-sitter-rust", "tree-sitter-ruby", "tree-sitter-php", "tree-sitter-kotlin", "tree-sitter-scala", "tree-sitter-lua", "tree-sitter-bash", "tree-sitter-elixir", "numpy"]
def venv_make(vd):
    """(Re)build the graph venv at its final path via uv (venvs are path-bound); the old one is parked and deleted only after the import check passes. Keeps pinned versions."""
    pins = []
    for di in glob.glob(os.path.join(vd, "lib", "python*", "site-packages", "*.dist-info")):
        n, v = os.path.basename(di)[:-10].rsplit("-", 1); n = n.replace("_", "-")
        if n in PKGS: pins.append(f"{n}=={v}")
    uv = shutil.which("uv") or sys.exit("hintmap: uv not found (needed to build the venv)"); park = vd + ".old"
    shutil.rmtree(park, ignore_errors=True)
    if os.path.isdir(vd): os.rename(vd, park)
    py = os.path.join(vd, "bin", "python")
    try:
        subprocess.run([uv, "venv", "-q", vd], check=True); subprocess.run([uv, "pip", "install", "-q", "-p", py, *pins, *(p for p in PKGS if not any(x.startswith(p + "==") for x in pins))], check=True)  # pinned versions of what was installed, plus newly listed packages
        subprocess.run([py, "-c", "import tree_sitter, numpy, tree_sitter_python"], check=True)
    except (subprocess.CalledProcessError, OSError):
        shutil.rmtree(vd, ignore_errors=True)
        if os.path.isdir(park): os.rename(park, vd)
        sys.exit("hintmap: venv build/import check failed; old venv restored")
    shutil.rmtree(park, ignore_errors=True); print(f"venv: ok {vd}")
def cmd_venv(a): venv_make(os.path.join(HOME(), "venv"))
def main(argv=None):
    p = argparse.ArgumentParser(prog="hintmap"); p.add_argument("--selftest", action="store_true")
    sp = p.add_subparsers(dest="cmd")
    def c(name, fn, *args, **kw):
        s = sp.add_parser(name); s.set_defaults(fn=fn)
        for flag, opts in args: s.add_argument(flag, **opts)
    s = sp.add_parser("hook"); s.add_argument("event", choices=["pretooluse-agent", "pretooluse-bash", "session-start", "subagent-start"])
    c("venv", cmd_venv); c("doctor", lambda *_: 0, ("--fix", dict(action="store_true")))
    s = sp.add_parser("bench-hook"); s.add_argument("root"); s.add_argument("--n", type=int, default=20); s.add_argument("--bin", default=SELF); s.add_argument("--stale", action="store_true", help="age the walk stamp before every call (hook calls further apart than the TTL)"); s.set_defaults(gfn=cmd_bench_hook)
    g = sp.add_parser("graph"); gs = g.add_subparsers(dest="gcmd", required=True)
    for n, f in (("init", cmd_graph_init), ("build", build_cmd), ("stats", cmd_graph_stats), ("audit", cmd_graph_audit), ("replay", cmd_graph_replay), ("model", cmd_graph_model)):
        s = gs.add_parser(n); s.add_argument("path", nargs="?"); s.set_defaults(gfn=f)
        if n == "build": s.add_argument("--full", action="store_true"); s.add_argument("--files", nargs="+", help="re-index exactly these relpaths (hook fast path)")
        if n == "init": s.add_argument("--paths", help="index scope, space/comma separated (default .)"); s.add_argument("--max-files", type=int, help="auto-index guard: log `too-big` and exit when the scope holds more files")
        if n == "model": s.add_argument("--dim", type=int, default=256); s.add_argument("--src")
        if n == "replay": s.add_argument("--join", action="store_true"); s.add_argument("--count", action="store_true")
        if n == "audit": s.add_argument("--n", type=int, default=50); s.add_argument("--tier", choices=["T1", "T2"])
    for n, f, arg in (("impact", cmd_graph_impact, "target"), ("tests-for", cmd_graph_tests, "target"), ("edit-check", cmd_graph_editcheck, "symbol")):  # top-level aliases of the graph readers (run from inside the repo)
        s = sp.add_parser(n); s.add_argument(arg); s.set_defaults(gfn=f)
        if n == "impact": s.add_argument("--depth", type=int, default=3); s.add_argument("--json", action="store_true"); s.add_argument("--limit", type=int, default=40)
        if n == "tests-for": s.add_argument("--depth", type=int, default=3); s.add_argument("--json", action="store_true")
        if n == "edit-check": s.add_argument("--limit", type=int, default=30); s.add_argument("--keep", action="store_true")
    for n, f, arg in (("def", cmd_def, "symbol"), ("refs", cmd_refs, "symbol"), ("show", cmd_show, "target")):  # just-in-time lookups (hm/jit.py)
        s = sp.add_parser(n); s.add_argument(arg); s.set_defaults(gfn=f)
        if n == "refs": s.add_argument("--limit", type=int, default=30); s.add_argument("--all", action="store_true")
        if n == "show": s.add_argument("--max", type=int, default=150)
    s = sp.add_parser("search"); s.add_argument("text", nargs="+"); s.add_argument("--k", type=int, default=8); s.add_argument("--terms", action="store_true", help="term search: exact sub-token/stem matches on symbol names, code lines, paths"); s.set_defaults(gfn=cmd_search)  # navigation (hm/nav.py)
    s = sp.add_parser("map"); s.add_argument("target", nargs="?", default="."); s.add_argument("--budget", type=int, default=1500, help="tokens (about 4 chars each)"); s.set_defaults(gfn=cmd_map)
    s = sp.add_parser("callees"); s.add_argument("symbol"); s.add_argument("--limit", type=int, default=40); s.set_defaults(gfn=cmd_callees)
    s = sp.add_parser("changed"); s.add_argument("rev", nargs="?"); s.add_argument("--limit", type=int, default=60); s.set_defaults(gfn=cmd_changed)
    s = sp.add_parser("q"); s.add_argument("text", nargs="+"); s.set_defaults(gfn=cmd_q)
    s = sp.add_parser("report"); s.add_argument("--since", help="ISO date/time; default: the Agent-hook A/B window start"); s.add_argument("--jit-only", action="store_true"); s.set_defaults(gfn=cmd_report)
    a = p.parse_args(argv)
    if a.selftest: return selftest()
    if not a.cmd: p.print_help(); return 1
    if a.cmd == "hook": return cmd_jit_hook() if a.event == "pretooluse-bash" else cmd_hook(a)
    if a.cmd == "doctor": return cmd_doctor(a)
    if a.cmd == "venv": return a.fn(a)
    return a.gfn(a)
if __name__ == "__main__":
    sys.exit(main())
