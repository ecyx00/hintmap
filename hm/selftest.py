"""hintmap.selftest: see README Module ownership."""
import array, datetime, contextlib, functools, hashlib, heapq, itertools, json, math, os, re, signal, sqlite3, struct, sys, time
from hm.common import HOLDOUT_PCT, POTION, SELF, codes_path, fts_match, glob, gopen, gpath, h, norm, query_of, rrf, sh, shutil, subprocess, subqueries, tempfile
from hm.graph import graph_block
from hm.hook import hook_root, nfiles, prompt_root, scratch
from hm.index import GSCHEMA, VENVPY, paths_of, write_codes
from hm.retrieve import path_score, pathparts, retrieve

import hm.common, hm.index, hm.retrieve
_M = (hm.common, hm.index, hm.retrieve)
def setg(**kw):  # the selftest patches module globals; the names are imported by value, so set them in every module that has them
    for m in _M:
        for k, v in kw.items():
            if hasattr(m, k): setattr(m, k, v)

def gdb_files(root, p): return gopen(root, True).execute("SELECT 1 FROM files WHERE path=?", (p,)).fetchone() is not None
def selftest():
    me = SELF
    tmp = tempfile.mkdtemp(prefix=".hm-selftest-", dir=os.path.expanduser("~"))  # under $HOME: the system temp dir (/private/var) is a scratch prefix the hook refuses
    os.environ["HINTMAP_AGENT_TYPES"] = "coder,deep,worker"; repo = os.path.join(tmp, "repo"); os.makedirs(repo)
    env = dict(os.environ, HINTMAP_HOME=os.path.join(tmp, "home"), HINTMAP_NO_AUTOINDEX="1", HINTMAP_NO_HEAVY_LOCK="1")  # tests must not wait on the machine-wide heavy lock; auto-index has its own case below
    def run(*args):
        r = subprocess.run([sys.executable, me, *args], cwd=repo, env=env, capture_output=True, text=True)
        assert r.returncode == 0, (args, r.stderr); return r.stdout
    def snap(): return sorted((d, f) for d, _, fs in os.walk(repo) if ".git" not in d for f in fs)
    g = lambda *c: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *c], cwd=repo, check=True, capture_output=True)
    g("init", "-q")
    open(os.path.join(repo, "a.py"), "w").write("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
    open(os.path.join(repo, "b.py"), "w").write("def other():\n    return 3\n")
    g("add", "."); g("commit", "-qm", "init"); before = snap()
    # hook checks: DB bytes untouched, no prompt text logged, no semble anywhere on PATH
    henv = dict(env, PATH="/usr/bin:/bin", HINTMAP_HOLDOUT_PCT="40")
    def hook(ev, stdin, **x):
        t = time.time(); r = subprocess.run([sys.executable, me, "hook", ev], cwd=repo, env=dict(henv, **x), input=stdin, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr; return r.stdout, time.time() - t
    ids = [i for i in map(str, range(99)) if int(hashlib.sha1(i.encode()).hexdigest()[:4], 16) % 100 < 40]
    hold, live = ids[0], next(i for i in map(str, range(99)) if i not in ids)
    ti = {"description": "d", "prompt": "SECRETPROMPT make foo return one", "subagent_type": "coder", "model": "haiku"}
    ev = lambda tid, **k: json.dumps({"cwd": repo, "tool_name": "Agent", "tool_use_id": tid, "session_id": "s", "tool_input": {**ti, **k}})
    assert hook("pretooluse-agent", ev(live, subagent_type="scout"))[0] == ""
    assert hook("pretooluse-agent", json.dumps({"cwd": repo, "tool_name": "Bash", "tool_input": {"subagent_type": "coder"}}))[0] == ""
    assert hook("pretooluse-agent", ev(live))[0] == "", "unregistered root: no block, nothing indexed"
    assert hook("pretooluse-agent", ev(hold))[0] == ""
    assert hook("pretooluse-agent", "garbage{")[0] == ""
    assert hook("pretooluse-agent", ev(live), HINTMAP_HOOK_DISABLED="1")[0] == ""
    lg = open(os.path.join(tmp, "home", "injections.jsonl")).read()
    assert '"arm": "holdout"' in lg and "SECRETPROMPT" not in lg and len(lg.splitlines()) == 2, lg
    ng = os.path.join(tmp, "nogit"); os.makedirs(os.path.join(ng, "sub")); open(os.path.join(ng, "pyproject.toml"), "w").close()
    nm = os.path.join(tmp, "nomarker"); os.makedirs(nm)
    ev2 = lambda cwd: json.dumps({"cwd": cwd, "tool_name": "Agent", "tool_use_id": live, "tool_input": ti})
    assert hook("pretooluse-agent", ev2(os.path.join(ng, "sub")))[0] == "" and json.loads(open(os.path.join(tmp, "home", "injections.jsonl")).read().splitlines()[-1])["root"] == os.path.realpath(ng) and hook("pretooluse-agent", ev2(nm))[0] == ""
    sp_ = os.path.join(tmp, "my repo"); os.makedirs(sp_); open(os.path.join(sp_, "package.json"), "w").close(); open(os.path.join(sp_, "x.py"), "w").write("x")
    pr = lambda p: json.dumps({"cwd": nm, "tool_name": "Agent", "tool_use_id": live, "tool_input": {**ti, "prompt": p}})
    assert hook("pretooluse-agent", pr(f"In {sp_}/x.py fix foo"))[0] == "" and json.loads(open(os.path.join(tmp, "home", "injections.jsonl")).read().splitlines()[-1])["root_from"] == "prompt" and hook("pretooluse-agent", pr("fix /nonexistent/x.py"))[0] == ""
    assert query_of(f"In {sp_}/x.py fix foo") == "In x fix foo", query_of(f"In {sp_}/x.py fix foo")
    brief = ("Context line about the task that is long enough to matter here, please read it carefully before touching anything. " * 3 + "\n- Fix compute_totals in api/app.py so the VAT sum is right\n- Update processingStore to handle batch results\n- Add a pytest for the endpoint")
    assert len(subqueries(brief)) >= 2 and subqueries("short brief") == ["short brief"], subqueries(brief)
    h = lambda p: (p, 1, 2, "")
    assert [x[0] for x in rrf([[h("a"), h("b"), h("c")], [h("c"), h("b")], [h("c"), h("d")]])] == ["c", "b", "a", "d"]
    # ---- 1a / 1c / 2: no git on PATH ----
    fz = os.path.join(os.path.realpath(tmp), "fz"); os.makedirs(fz); open(os.path.join(fz, "pyproject.toml"), "w").close(); fa = os.path.join(fz, "a.py")
    open(fa, "w").write("def foo():\n    return 1\n"); fhome = os.path.join(tmp, "fhome"); fenv = dict(env, PATH="/nonexistent", HINTMAP_HOME=fhome)
    def fr(*args, cwd=fz, stdin=None):
        r = subprocess.run([sys.executable, me, *args], cwd=cwd, env=fenv, capture_output=True, text=True, input=stdin); assert r.returncode == 0, (args, r.stderr); return r.stdout
    # ---- doctor --fix: vanished registry root dropped ----
    xh = os.path.join(tmp, "xhome"); os.makedirs(xh); open(os.path.join(xh, "graphs"), "w").write(f"/nonexistent/vanished\n{fz}\n")
    xd = lambda *a: subprocess.run([sys.executable, me, "doctor", *a], cwd=fz, env=dict(env, HINTMAP_HOME=xh), capture_output=True, text=True).stdout
    assert "1 fixable" in xd() and "VANISHED" in xd() and "dropped vanished root" in xd("--fix") and open(os.path.join(xh, "graphs")).read() == fz + "\n" and "nothing to fix" in xd("--fix")
    assert fr("hook", "session-start", stdin=json.dumps({"cwd": fz})) == "", "session-start emits no context"
    # ---- retriever: normalizer ----
    assert set(norm("processingStore")) >= {"processing", "store", "processingstore"} and set(norm("processing_store")) >= {"processingstore"} and norm("HTTPServer") == ["http", "server", "httpserver"]
    assert "api" in norm("src/api/app.py")
    assert norm("Işık Çağrı Öğle Şükrü Gül ve the") == ["isik", "cagri", "ogle", "sukru", "gul"], norm("Işık Çağrı Öğle Şükrü Gül ve the")
    assert fts_match(["export", "pdf"]) == '"export"* OR "pdf"'
    # ---- graph: plain dir + marker (no git), 3 languages ----
    gpy = VENVPY()
    if not os.path.exists(gpy): print("selftest: graph part skipped (no ~/.hintmap/venv)"); print("selftest ok"); return 0
    gr = os.path.join(tmp, "gproj"); os.makedirs(os.path.join(gr, "ts")); gh = os.path.join(tmp, "ghome")
    genv = dict(env, HINTMAP_HOME=gh, HINTMAP_HUB_MIN="5")
    def w(rel, txt): open(os.path.join(gr, rel), "w").write(txt)
    def gx(*args, cwd=gr, ok=True):
        r = subprocess.run([sys.executable, me, "graph", *args], cwd=cwd, env=genv, capture_output=True, text=True)
        assert (r.returncode == 0) == ok, (args, r.stdout, r.stderr); return r.stdout + r.stderr
    w("pyproject.toml", ""); w("lib.py", "def compute_total(x):\n    return x\n\ndef shared_name():\n    return 1\n")
    w("app.py", "from lib import compute_total\n\ndef run():\n    return compute_total(1)\n")
    w("ts/store.ts", "export function loadStore() { return 1 }\nexport const makeThing = () => 2;\n")
    w("ts/ui.ts", "import { loadStore } from './store';\nexport function view() { loadStore(); return rust_only_helper() + shared_name(); }\n")
    w("m.js", "function rust_only_helper() { return 1 }\nfunction shared_name() { return 2 }\n")
    w("hub.py", "def hub_fn():\n    return 1\n")
    for i in range(6): w(f"h{i}.py", f"from hub import hub_fn\n\ndef f{i}x():\n    return hub_fn()\n")
    r = subprocess.run([sys.executable, me, "graph", "build"], cwd=gr, env=genv, capture_output=True, text=True)
    assert r.returncode != 0 and "graph init" in r.stderr, "build must refuse uninitialized root"
    out = gx("init"); assert "parsed=" in out
    assert open(os.path.join(gr, ".hintmap", ".gitignore")).read() == "*\n" and open(os.path.join(gh, "graphs")).read().split("\n")[0] == os.path.realpath(gr)
    assert os.listdir(gh) == ["graphs"], "graph code must not create mem.db"
    gd = lambda q: sqlite3.connect(os.path.join(gr, ".hintmap", "graph.db")).execute(q).fetchall()
    assert {"compute_total", "loadStore", "makeThing", "rust_only_helper", "hub_fn"} <= {x[0] for x in gd("select name from symbols")}
    E = "select count(*) from edges e join files a on a.id=e.src_file join files b on b.id=e.dst_file left join symbols s on s.id=e.dst_sym where "
    assert gd(E + "e.kind='refs' and e.tier=1 and a.path='app.py' and b.path='lib.py' and s.name='compute_total'")[0][0] == 1
    assert gd(E + "e.kind='imports' and e.tier=1 and a.path='ts/ui.ts' and b.path='ts/store.ts'")[0][0] == 1
    assert gd(E + "e.kind='refs' and e.tier=2 and a.path='ts/ui.ts' and s.name='rust_only_helper'")[0][0] == 1
    assert gd(E + "s.name='shared_name'")[0][0] == 0, "ambiguous name must be dropped"
    assert gd("select hub from nodes n join files f on f.id=n.file_id where f.path='hub.py'") == [(1,)]
    assert "parsed=0" in gx("build")
    w("lib.py", "def compute_total(x):\n    return x + 1\n"); assert "parsed=1 " in gx("build")
    os.remove(os.path.join(gr, "m.js")); gx("build")
    assert not gd("select 1 from files where path='m.js'") and not gd("select 1 from symbols where name='rust_only_helper'")
    assert "[T1" in gx("audit", "--n", "5", "--tier", "T1") and "hub.py" in gx("stats")
    # defect fixes: module-over-init, ts ./x vs ./x/index, binding vs use, nested targets
    for rel in ("pk", "tx/x", "tx/y"): os.makedirs(os.path.join(gr, rel), exist_ok=True)
    w("pk/__init__.py", "def init_only():\n    return 0\n"); w("pk/mod_a.py", "def mod_a_fn():\n    return 1\n"); w("pk/mod_b.py", "def mod_b_fn():\n    return 2\n")
    w("use1.py", "from pk.mod_a import mod_a_fn\n\ndef u1():\n    return mod_a_fn()\n"); w("use2.py", "from pk import mod_b\n\ndef u2():\n    return mod_b\n")
    w("tx/x.ts", "export function xmod() { return 1 }\n"); w("tx/x/index.ts", "export function xidx() { return 1 }\n"); w("tx/y/index.ts", "export function yidx() { return 1 }\n")
    w("tx/c.ts", "import { xmod } from './x';\nimport { yidx } from './y';\nexport function c() { return xmod() + yidx(); }\n")
    w("nest.py", "def outer_fn():\n    def inner_unique_helper():\n        return 1\n    return inner_unique_helper()\n")
    w("local.py", "def other_fn(inner_unique_helper):\n    inner_unique_helper = {'inner_unique_helper': 1}\n    return inner_unique_helper\n\ndef shadow(real_unique_target):\n    real_unique_target = {'real_unique_target': 1}\n    return real_unique_target\n")
    w("real.py", "def real_unique_caller():\n    return real_unique_target()\n"); w("tgt.py", "def real_unique_target():\n    return 1\n")
    gx("build")
    IM = "select count(*) from edges e join files b on b.id=e.dst_file join files a on a.id=e.src_file where e.kind='imports' and a.path='%s' and b.path='%s'"
    assert gd(IM % ("use1.py", "pk/mod_a.py"))[0][0] == 1 and gd(IM % ("use1.py", "pk/__init__.py"))[0][0] == 0, "module must beat package init"
    assert gd(IM % ("use2.py", "pk/mod_b.py"))[0][0] == 1 and gd(IM % ("use2.py", "pk/__init__.py"))[0][0] == 0, "submodule import beats init"
    assert gd(IM % ("tx/c.ts", "tx/x.ts"))[0][0] == 1 and gd(IM % ("tx/c.ts", "tx/x/index.ts"))[0][0] == 0 and gd(IM % ("tx/c.ts", "tx/y/index.ts"))[0][0] == 1
    assert gd(E + "s.name='inner_unique_helper'")[0][0] == 0, "nested def / local binding must not be an edge target or T2 source"
    assert gd(E + "e.tier=2 and a.path='real.py' and s.name='real_unique_target'")[0][0] == 1
    assert gd(E + "a.path='local.py'")[0][0] == 0, "param/local/dict-key named like a unique top-level def must not make an edge"
    # ---- Faz 1: graph_block fixture (named seed neighbour, hub barrier, stale, size) + hook end-to-end ----
    gb = os.path.join(tmp, "gblock"); os.makedirs(gb); gbw = lambda rel, t: open(os.path.join(gb, rel), "w").write(t)
    gbw("pyproject.toml", ""); gbw("core.py", "from hubmod import hub_helper_fn\n\ndef compute_core_total(x):\n    return hub_helper_fn(x)\n")
    gbw("svc.py", "from core import compute_core_total\n\ndef run_service():\n    return compute_core_total(1)\n"); gbw("hubmod.py", "def hub_helper_fn(x):\n    return x\n")
    for i in range(6): gbw(f"leaf{i}.py", f"from hubmod import hub_helper_fn\n\ndef leaf_{i}_fn():\n    return hub_helper_fn({i})\n")
    gbw("big.clj", "".join(f"(defn fn-{i} [x]\n  (inc x))\n\n" for i in range(60)) + "(defn zyxwvu-quartzite [x]\n  x)\n"); gbw("Lib.hs", "module Lib where\n\nhaskellSymbolOne :: Int -> Int\nhaskellSymbolOne x = x + 1\n")  # no-grammar window chunker, grammar-backed .hs
    gbw("merge_request.py", "def zq():\n    return 1\n")  # found by name only, via stemming ("merged requests")
    gbw("deep.json", "[\n" * 5000 + "]\n" * 5000)  # json.loads RecursionError must not fail the build (aeson)
    gbw("minified_bundle.py", "v=" + "+".join(str(i) for i in range(900)) + "\n"); gbw("pbgen.py", "# Code generated by protoc. DO NOT EDIT.\ndef pb_generated_fn():\n    return 1\n")  # minified (one 3 KB line) / generated header
    gbw(".gitattributes", "# c\nattrdef.py linguist-generated=true\n"); gbw("attrdef.py", "def attr_generated_fn():\n    return 1\n")
    gbw("receipt.py", "def render_receipt_total(items):\n    return sum(items)\n")
    os.makedirs(os.path.join(gb, "tests")); gbw("tests/test_receipt.py", "def test_render_receipt_total():\n    assert render_receipt_total([1]) == 1\n")
    gbw("widgets.py", "".join(f"def widget_{i}(x):\n    # widget helper {i}\n    return x\n" for i in range(8)))
    for ext in ("db", "i8"):  # the vector channel needs the exported model; absent -> lexical only
        if os.path.exists(POTION() + "." + ext): shutil.copy(POTION() + "." + ext, os.path.join(gh, "potion." + ext))
    gx("init", gb, cwd=gb); gdb = os.path.join(gb, ".hintmap", "graph.db"); b4db = open(gdb, "rb").read()
    os.environ["HINTMAP_HOME"] = gh
    pr1 = "Fix compute_core_total in core.py so the sum is right"
    L = graph_block(gb, pr1, [[]], {}); ps = [p for p, _ in L]
    assert "svc.py" in ps and any("calls compute_core_total" in t for _, t in L) and not any(p.startswith("leaf") or p == "hubmod.py" for p in ps) and "core.py" not in ps, L
    assert len(" ".join(t for _, t in L)) // 4 <= 700
    gbw("svc.py", "from core import compute_core_total\n\ndef run_service():\n    return compute_core_total(2) + 1\n")  # now stale vs the graph
    L2 = dict(graph_block(gb, pr1, [[]], lg := {})); assert L2["svc.py"] == "svc.py" and lg["stale_dropped"] == 1, (L2, lg)
    assert open(gdb, "rb").read() == b4db, "graph_block wrote to graph.db"
    # retriever: sub-token-only symbol, test penalty, one entry per file, contentless FTS, hook with no semble on PATH
    os.environ["HINTMAP_HOME"] = gh; hs = [h[0] for h in retrieve(gb, ["receipt total"], {}, "lex")[0]]
    assert "receipt.py" in hs[:2] and "tests/test_receipt.py" in hs and hs.index("receipt.py") < hs.index("tests/test_receipt.py"), hs
    assert retrieve(gb, ["pytest the receipt total"], {}, "lex")[0], "test penalty must be off when the query asks for tests"
    hb = retrieve(gb, ["zyxwvu quartzite"], {}, "lex")[0]; assert hb and hb[0][0] == "big.clj" and hb[0][1] > 150, hb  # a word deep in a grammar-less file ranks that file via its window
    assert retrieve(gb, ["haskell symbol one"], {}, "lex")[0][0][0] == "Lib.hs" and gopen(gb, True).execute("SELECT COUNT(*) FROM symbols s JOIN files f ON f.id=s.file_id WHERE f.path='Lib.hs'").fetchone()[0] >= 1, "haskell: indexed with symbols"
    assert retrieve(gb, ["compute_core_total"], {}, "lex")[0][0][0] == "core.py", "exact symbol channel"
    assert retrieve(gb, ["merged requests"], {}, "lex")[0][0][0] == "merge_request.py", "stemmed file-name channel"
    assert pathparts("a/widgetry/mod.rs")[0] == ("widgetry",) and pathparts("a/widgetry/__init__.py")[0] == ("widgetry",) and pathparts("src/api_client.go")[0] == ("api", "client") and pathparts("src/merge_api.go")[0] == ("merge",)
    assert path_score("lib/merge_request.py", ["merged", "request"]) == 1.0 and path_score("lib/merge_request.py", ["merge", "request"], True) == 1.0 and path_score("lib/merge_request.py", ["merged"], True) == 0 and path_score("a/mod.rs", ["a"]) == 0.0
    assert retrieve(gb, ["zzqxjv"], {}, "lex") == [[]] and gdb_files(gb, "deep.json"), "empty candidate list must not crash retrieve; deep json indexed"
    hw = [h[0] for h in retrieve(gb, ["widget helper"], {}, "lex")[0]]; assert hw.count("widgets.py") == 1, hw
    gdbc = sqlite3.connect(gdb); assert gdbc.execute("SELECT COUNT(*) FROM ix WHERE ix MATCH 'receipt'").fetchone()[0] >= 2
    raw = open(gdb, "rb").read().decode("latin1"); assert "def render_receipt_total" not in raw and "return sum(items)" not in raw and "assert render_receipt_total" not in raw, "source text leaked into graph.db"
    if os.path.exists(os.path.join(gh, "potion.db")):
        assert gdbc.execute("SELECT COUNT(v) FROM chunks").fetchone()[0] > 5 and any(retrieve(gb, ["receipt total"], {}, "vec")[0]), "vector channel"
        ex = retrieve(gb, ["receipt total", "widget helper"], {}, "vec"); setg(VEC_EXACT_MAX=0); ap = retrieve(gb, ["receipt total", "widget helper"], {}, "vec"); setg(VEC_EXACT_MAX=40000)
        assert ex == ap, "2-stage vector scan must equal the exact scan while VEC_KEEP >= #chunks"
    tig = {"description": "d", "prompt": pr1, "subagent_type": "coder"}
    hk = lambda: subprocess.run([sys.executable, me, "hook", "pretooluse-agent"], cwd=gb, env=dict(genv, PATH="/usr/bin:/bin", HINTMAP_HOLDOUT_PCT="0"), capture_output=True, text=True,
                                input=json.dumps({"cwd": gb, "tool_name": "Agent", "tool_use_id": "x", "session_id": "s", "tool_input": tig}))
    r = hk(); assert "Code locations (hintmap" in r.stdout and "svc.py" in r.stdout and "  function" not in r.stdout and "  class" not in r.stdout and "semble" not in r.stdout, (r.stdout, r.stderr)
    open(os.path.join(gh, "graph-disabled"), "w").close()
    r = hk(); assert "calls compute_core_total" not in r.stdout, "kill switch ignored"; os.remove(os.path.join(gh, "graph-disabled"))
    # ---- fresh index: PreToolUse sync update, lock-held no-wait, shared walk stamp ----
    gsyms = lambda: {r[0] for r in sqlite3.connect(gdb).execute("SELECT name FROM symbols")}; stamp = os.path.join(gb, ".hintmap", "walk.ok"); lock = os.path.join(gb, ".hintmap", "build.lock")
    hlog = lambda: json.loads(open(os.path.join(gh, "injections.jsonl")).read().splitlines()[-1])
    assert not os.path.exists(lock), "lock leaked"
    if os.path.exists(stamp): os.remove(stamp)
    gbw("newmod.py", "def brand_new_symbol_fn():\n    return 1\n"); gbw("svc.py", "from core import compute_core_total\n\ndef run_service():\n    return compute_core_total(3) + 2\n")
    tig["prompt"] = "Fix brand_new_symbol_fn in newmod.py"; r = hk(); L = hlog()  # (c) 2 changed files -> indexed inside the same hook call, before retrieval
    assert "brand_new_symbol_fn" in gsyms() and L["synced"] is True and L["changed"] == 2 and "newmod.py" in L["emitted"], (r.stdout, L)  # a file the index had never seen is retrievable in the same call
    t0 = time.time(); r = hk(); wall = time.time() - t0; L = hlog(); assert L["walk"] == "skipped", L  # (e) second call inside WALK_TTL: no walk
    assert wall < 0.2 and L["cpu_ms"] < 200 and "t" in L, (wall, L["cpu_ms"], L.get("t"))  # latency budget on the fixture: fresh process, hook <= 200 ms wall and CPU (`hintmap bench-hook ROOT` for real repos)
    assert not any(gdb_files(gb, f) for f in ("minified_bundle.py", "pbgen.py", "attrdef.py")) and gdb_files(gb, "core.py"), "generated/minified files must not be indexed"
    tig["prompt"] = "Fix compute_core_total in core.py"; r = hk(); L = hlog(); tig["prompt"] = pr1  # a file the brief names is not listed again
    blk = r.stdout.split("## Code locations")[-1] if r.stdout else ""
    assert "core.py" in L["emitted"] and L["dropped_named"] >= 1 and "core.py" not in blk, (L, r.stdout)
    os.remove(stamp); gbw("newmod.py", "def locked_sym_fn():\n    return 1\n"); open(lock, "w").close()
    t0 = time.time(); r = hk(); L = hlog(); assert time.time() - t0 < 2 and L["walk"] == "locked" and "locked_sym_fn" not in gsyms() and "newmod.py:" not in r.stdout, (L, r.stdout)  # (d) lock held: no wait, path-only
    os.remove(lock)
    gbw("newmod.py", "def brand_new_symbol_fn():\n    return 1\n"); gbw("svc.py", "from core import compute_core_total\n\ndef run_service():\n    return compute_core_total(1)\n")
    # budgets: no source text in the db, free pages reclaimed, bytes/file
    cols = {r[1] for t in ("files", "symbols", "refs", "imports", "edges", "nodes") for r in sqlite3.connect(gdb).execute(f"PRAGMA table_info({t})")}
    assert not cols & {"content", "text", "body", "source", "snippet"}, "graph.db must not store source text"; assert "content=''" in sqlite3.connect(gdb).execute("SELECT sql FROM sqlite_master WHERE name='ix'").fetchone()[0], "FTS must be contentless"
    nfiles = sqlite3.connect(gdb).execute("SELECT COUNT(*) FROM files").fetchone()[0]; assert os.path.getsize(gdb) / nfiles < 15000, "graph.db budget"
    # ---- impact / edit-check / tests-for / routes / data files / incremental derive on a fixture ----
    gi = os.path.join(tmp, "gimpact"); os.makedirs(os.path.join(gi, "tests")); os.makedirs(os.path.join(gi, "locales")); giw = lambda rel, t: open(os.path.join(gi, rel), "w").write(t)
    giw("pyproject.toml", ""); giw("lib.py", "def core_fn(a):\n    return a\n"); giw("mid.py", "from lib import core_fn\n\ndef mid_fn():\n    return core_fn(1)\n")
    giw("top.py", "from mid import mid_fn\n\ndef top_fn():\n    return mid_fn()\n"); giw("tests/test_top.py", "from top import top_fn\n\ndef test_top():\n    assert top_fn()\n")
    giw("uniq.py", "def unique_target_fn():\n    return 1\n"); giw("other.py", "def other_caller():\n    return unique_target_fn()\n")
    giw("dup1.py", "def dup_fn():\n    return 1\n"); giw("dup2.py", "def dup_fn():\n    return 2\n")
    giw("api.py", "from fastapi import FastAPI\napp = FastAPI()\n\n@app.post(\"/api/v2/batches\")\ndef register_sources():\n    return 1\n\nprint(open(\"/etc/hosts\"))\n")
    giw("ui.ts", "export async function ping() { return fetch(`/api/v2/health/${1}`); }\n")
    giw("locales/en.json", json.dumps({"settings": {"language": {"switch": "Switch language"}}, "home": {"title": "Welcome"}})); giw("package-lock.json", json.dumps({"lockfileVersion": 3, "zzlockkey": {}}))
    giw(".env.example", "API_BASE_URL=http://x\nSECRET_TOKEN=abc\n"); giw("README.md", "# Receipt Parser\n\n## Batch uploads\n\n```\n# not a heading\n```\n")
    gix = lambda *args, ok=True: (lambda r: (r.returncode == 0) == ok and r.stdout + r.stderr or (_ for _ in ()).throw(AssertionError((args, r.stdout, r.stderr))))(subprocess.run([sys.executable, me, *args], cwd=gi, env=genv, capture_output=True, text=True))
    for ext in ("db", "i8"):  # real model if present: the pure-python and the numpy embedder must agree
        if os.path.exists(os.path.expanduser(f"~/.hintmap/potion.{ext}")): shutil.copy(os.path.expanduser(f"~/.hintmap/potion.{ext}"), os.path.join(gh, "potion." + ext))
    gix("graph", "init"); gidb = os.path.join(gi, ".hintmap", "graph.db"); gq = lambda q: sqlite3.connect(gidb).execute(q).fetchall()
    if os.path.exists(os.path.join(gh, "potion.i8")):
        v1 = [r[0] for r in gq("select v from chunks order by id")]; assert any(v1)
        subprocess.run([sys.executable, me, "graph", "build", "--full"], cwd=gi, env=dict(genv, HINTMAP_PY_EMBED_MAX="0"), check=True, capture_output=True); v2 = [r[0] for r in gq("select v from chunks order by id")]
        assert len(v1) == len(v2) and all(a_ == b_ or (a_ and b_ and max(abs(x - y) for x, y in zip(memoryview(a_).cast("b"), memoryview(b_).cast("b"))) <= 2) for a_, b_ in zip(v1, v2)), "pure-python and numpy embeddings differ"
    assert not gq("select 1 from files where path like '%lock%'") and {r[0] for r in gq("select lang from files")} >= {"json", "env", "markdown", "python", "typescript"}, gq("select path from files")
    assert gq("select 1 from symbols where kind='route' and name='/api/v2/batches'") and gq("select 1 from symbols where kind='route' and name='/api/v2/health/'") and not gq("select 1 from symbols where name like '/etc%'"), gq("select name, kind from symbols where kind='route'")
    os.environ["HINTMAP_HOME"] = gh
    assert "api.py" in [x[0] for x in retrieve(gi, ["batches endpoint"], {}, "lex")[0]][:3], "route string must be retrievable"
    assert "locales/en.json" in [x[0] for x in retrieve(gi, ["language switch"], {}, "lex")[0]][:3], "locale json key must be retrievable"
    assert ".env.example" in [x[0] for x in retrieve(gi, ["api base url"], {}, "lex")[0]][:3] and "README.md" in [x[0] for x in retrieve(gi, ["receipt parser batch uploads"], {}, "lex")[0]][:3]
    gl = os.path.join(tmp, "gloc"); os.makedirs(os.path.join(gl, "src", "i18n")); open(os.path.join(gl, "pyproject.toml"), "w").close()  # a big key list: bm25 length-normalises its path hit away, the locale-by-path rule must still surface it
    open(os.path.join(gl, "src", "i18n", "en.json"), "w").write(json.dumps({f"key_{i}_label": f"Label {i}" for i in range(500)}))
    for i in range(6): open(os.path.join(gl, f"loc{i}.py"), "w").write(f"def get_locale_{i}():\n    return 'locale translation strings'\n")
    subprocess.run([sys.executable, me, "graph", "init", gl], cwd=gl, env=genv, check=True, capture_output=True); os.environ["HINTMAP_HOME"] = gh
    assert "src/i18n/en.json" in [x[0] for x in retrieve(gl, ["locale translation strings"], {}, "lex")[0]][:3], "locale data file must surface for a locale query"
    if subprocess.run([gpy, "-c", "import tree_sitter_swift"], capture_output=True).returncode == 0:  # swift: `extension T` must not be a second definition of T; the extension's members are reachable
        gw = os.path.join(tmp, "gswift"); os.makedirs(gw); open(os.path.join(gw, "Package.swift"), "w").close()
        open(os.path.join(gw, "A.swift"), "w").write("final class FocusController {\n    func focusKeyboard() {}\n}\n"); open(os.path.join(gw, "B.swift"), "w").write("extension FocusController {\n    func dockPanelBelow() {}\n}\nfunc useIt(c: FocusController) { c.focusKeyboard() }\n")
        subprocess.run([sys.executable, me, "graph", "init", gw], cwd=gw, env=genv, check=True, capture_output=True); gwq = lambda q: sqlite3.connect(os.path.join(gw, ".hintmap", "graph.db")).execute(q).fetchall()
        assert gwq("select count(*) from symbols where name='FocusController'")[0][0] == 1 and gwq("select parent_id from symbols where name='dockPanelBelow'") == [(None,)] and gwq("select count(*) from edges e join symbols s on s.id=e.dst_sym where s.name='FocusController'")[0][0] >= 1, gwq("select name, kind, parent_id from symbols")
    assert not gq("select 1 from files where path='package-lock.json'") and not gq("select 1 from symbols where name='zzlockkey'")
    ij = json.loads(gix("impact", "core_fn", "--json")); lv = ij["levels"]; assert [x["path"] for x in lv["1"]] == ["mid.py"] and lv["1"][0]["tier"] == 1 and [x["path"] for x in lv["2"]] == ["top.py"] and lv["3"][0]["path"] == "tests/test_top.py" and lv["3"][0]["test"], ij
    assert "3" not in json.loads(gix("impact", "core_fn", "--depth", "2", "--json"))["levels"] and "T2 other.py" in gix("impact", "unique_target_fn") and "tests reached: tests/test_top.py" in gix("impact", "core_fn")
    assert "ambiguous" in gix("impact", "dup_fn") and "dup1.py::dup_fn" in gix("impact", "dup_fn") and "no dependents" in gix("impact", "other_caller") and "no symbol" in gix("impact", "nope_fn")
    assert gix("tests-for", "core_fn").startswith("tests/test_top.py") and gix("tests-for", "top.py").startswith("tests/test_top.py") and "no test files" in gix("tests-for", "unique_target_fn")
    assert "signature unchanged" in gix("edit-check", "mid_fn") and "no callers" not in gix("edit-check", "mid_fn")
    giw("mid.py", "from lib import core_fn\n\ndef mid_fn(extra):\n    return core_fn(extra)\n"); ec = gix("edit-check", "mid_fn")
    assert "signature changed (arity 0 -> 1)" in ec and "top.py:4" in ec and "return mid_fn()" in ec, ec
    assert "signature unchanged" in gix("edit-check", "mid_fn") and "no callers in the index" in gix("edit-check", "other_caller")  # baseline consumed by the first run
    os.remove(os.path.join(gi, "dup2.py")); ec = gix("edit-check", "dup_fn"); assert "dup1.py" in ec or "no symbol" in ec
    # incremental derive == full derive (edges + nodes, by path/name), after an edit, a new file and a removal
    def graph_state():
        c = sqlite3.connect(gidb); q = ("SELECT sf.path, COALESCE(ss.name,''), df.path, COALESCE(ds.name,''), e.kind, e.tier, e.w, e.line, e.n FROM edges e JOIN files sf ON sf.id=e.src_file JOIN files df ON df.id=e.dst_file "
                                        "LEFT JOIN symbols ss ON ss.id=e.src_sym LEFT JOIN symbols ds ON ds.id=e.dst_sym")
        return sorted(c.execute(q).fetchall()), sorted(c.execute("SELECT f.path, n.indeg, n.hub FROM nodes n JOIN files f ON f.id=n.file_id").fetchall()), sorted(c.execute("SELECT path FROM imp i JOIN files f ON f.id=i.file_id").fetchall())
    giw("top.py", "from mid import mid_fn\nfrom lib import core_fn\n\ndef top_fn():\n    return mid_fn(core_fn(2))\n\ndef top_two():\n    return unique_target_fn()\n"); giw("newer.py", "from top import top_two\n\ndef newer_fn():\n    return top_two()\n"); os.remove(os.path.join(gi, "uniq.py"))
    out = gix("graph", "build"); assert "parsed=2" in out and "deleted=2" in out, out
    inc = graph_state(); gix("graph", "build", "--full"); assert inc == graph_state(), "incremental derive differs from full derive"
    giw("lib.py", "def core_fn(a, b=1):\n    return a\n\ndef unique_target_fn():\n    return 3\n"); gix("graph", "build"); inc = graph_state(); gix("graph", "build", "--full"); assert inc == graph_state(), "incremental derive (new def of a name referenced elsewhere) differs"
    gg = os.path.join(tmp, "gitproj"); os.makedirs(gg); open(os.path.join(gg, "package.json"), "w").write("{}"); open(os.path.join(gg, "x.ts"), "w").write("export function xx() {}\n")
    g2 = lambda *c: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *c], cwd=gg, check=True, capture_output=True)
    g2("init", "-q"); g2("add", "."); g2("commit", "-qm", "i")
    gitsnap = lambda: sorted((d, f, open(os.path.join(d, f), "rb").read()) for d, _, fs in os.walk(os.path.join(gg, ".git")) for f in fs if f not in ("index",))
    b4 = gitsnap(); gx("init", cwd=gg); gx("build", cwd=gg)
    assert gitsnap() == b4 and not sh(["git", "status", "--porcelain"], cwd=gg), "graph touched git or showed up in status"
    # ---- role sync: an index built under older TESTP/GENP regexes gets the new roles on its next build (no-op and only=) without re-chunking ----
    rd = sqlite3.connect(gpath(gg)); rd.execute("UPDATE files SET role='test'"); rd.execute("DELETE FROM meta WHERE k='rolep'"); rd.commit(); nch = rd.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]; rd.close()
    gx("build", cwd=gg); rd = sqlite3.connect(gpath(gg)); assert rd.execute("SELECT role FROM files WHERE path='x.ts'").fetchone()[0] == "src" and rd.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == nch, "no-op build did not re-derive roles"
    rd.execute("UPDATE files SET role='test'"); rd.execute("UPDATE meta SET v='stale' WHERE k='rolep'"); rd.commit(); rd.close()
    gx("build", "--files", "x.ts", cwd=gg); rd = sqlite3.connect(gpath(gg)); assert rd.execute("SELECT role FROM files WHERE path='x.ts'").fetchone()[0] == "src", "only= build did not re-derive roles"; rd.close()
    # ---- vector sign codes: write_codes + Hamming prefilter + exact rescoring finds the planted vector ----
    vr = os.path.join(tmp, "vroot"); os.makedirs(os.path.join(vr, ".hintmap")); vd = sqlite3.connect(gpath(vr)); vd.executescript(GSCHEMA); rnd = [os.urandom(256) for _ in range(300)]
    vd.executemany("INSERT INTO chunks(id,file_id,sym_id,lo,hi,v) VALUES(?,?,NULL,0,0,?)", [(i + 1, i // 5 + 1, v) for i, v in enumerate(rnd)])
    vd.executemany("INSERT INTO files(id,path,lang,size,mtime_ns,sha,role,loc) VALUES(?,?,?,?,?,?,?,?)", [(f, f"f{f}.py", "python", 1, 1, "x", "src", 1) for f in range(1, 62)]); vd.execute("INSERT INTO meta VALUES('vec_dim','256')")
    sv = {k: getattr(_M[0], k) for k in ("VEC_EXACT_MAX", "potion_open", "potion_vec")}; setg(VEC_EXACT_MAX=100, potion_open=lambda: (None, None, 256), potion_vec=lambda P, w: list(memoryview(rnd[123]).cast("b")))
    try:
        write_codes(vd, vr); vd.commit(); assert os.path.getsize(codes_path(vr)) == 12 + 300 * 4 + 300 * 32; lg = {}
        assert retrieve(vr, ["alpha"], lg, "vec")[0][0][0] == "f25.py" and lg["chunks_vec"] == 300, lg  # chunk id 124 -> file 124 // 5 + 1 = 25
    finally: setg(**sv)
    # ---- linked worktree: .git FILE -> main root ----
    wm, wt, wh = (os.path.realpath(os.path.join(tmp, n)) for n in ("wmain", "wt1", "whome")); os.makedirs(os.path.join(wm, ".git", "worktrees", "wt1")); os.makedirs(os.path.join(wt, "sub")); os.makedirs(wh)
    open(os.path.join(wt, ".git"), "w").write(f"gitdir: {wm}/.git/worktrees/wt1\n"); os.utime(os.path.join(wt, ".git"), (1e9, 1e9)); open(os.path.join(wm, "pyproject.toml"), "w").write("")
    assert hook_root(os.path.join(wt, "sub")) == wm and prompt_root(f"edit {wt}/sub/x.py") == wm
    # ---- root from the brief: cwd in repo A + brief naming files in repo B -> B; scratch roots never count; linked worktree in scratch -> main ----
    ra, rb = (os.path.realpath(os.path.join(tmp, n)) for n in ("rootA", "rootB")); [os.makedirs(os.path.join(r, "src")) for r in (ra, rb)]
    for r in (ra, rb): open(os.path.join(r, "pyproject.toml"), "w").close()
    open(os.path.join(rb, "src", "b.py"), "w").write("x = 1\n"); open(os.path.join(ra, "src", "a.py"), "w").write("x = 1\n")
    assert prompt_root(f"fix {rb}/src/b.py and {rb}/src and {ra}/src/a.py") == rb and prompt_root(f"fix {ra}/src/a.py and {rb}/src/b.py") == ra, "brief-only: first mentioned root wins"
    sd = tempfile.mkdtemp(dir="/private/tmp"); os.makedirs(os.path.join(sd, ".git")); open(os.path.join(sd, "q.py"), "w").write("x = 1\n")
    swt = os.path.join(sd, "wt"); os.makedirs(swt); os.makedirs(os.path.join(rb, ".git", "worktrees", "w1")); open(os.path.join(swt, ".git"), "w").write(f"gitdir: {rb}/.git/worktrees/w1\n"); os.utime(os.path.join(swt, ".git"), (1e9, 1e9))
    try:
        assert scratch("/opt/homebrew") and scratch("/private/tmp/x") and scratch("/var/folders/a") and scratch(os.path.expanduser("~/Library/x")) and not scratch(ra)
        assert prompt_root(f"edit {sd}/q.py") is None and prompt_root("edit /opt/homebrew/bin and /usr/bin/env") is None, "brief naming only scratch/system paths has no root"
        assert prompt_root(f"edit {swt}/x.py") == rb, "worktree under scratch maps to its normal main checkout"
        rl = lambda: json.loads(open(os.path.join(tmp, "home", "injections.jsonl")).read().splitlines()[-1])
        rp = lambda cwd, brief: hook("pretooluse-agent", json.dumps({"cwd": cwd, "tool_name": "Agent", "tool_use_id": live, "tool_input": {**ti, "prompt": brief}}))
        rp(ra, f"change {rb}/src/b.py"); assert rl()["root"] == rb and rl()["root_from"] == "prompt", rl()
        rp(ra, f"reference {rb}/src/b.py and {rb}/src, edit {ra}/src/a.py"); assert rl()["root"] == ra and rl()["root_from"] == "cwd", rl()  # mixed brief citing a reference repo more often than the cwd repo: cwd
        rp(ra, "no paths here"); assert rl()["root"] == ra and rl()["root_from"] == "cwd", rl()
        rp(ra, f"look at {sd}/q.py"); assert rl()["skipped"] == "no-root" and rl()["root_from"] == "prompt-scratch", rl()  # brief points at a scratch copy: no root, cwd does not stand in
        rp(sd, "no paths"); assert rl()["skipped"] == "no-root" and "root" not in rl(), rl()
    finally: shutil.rmtree(sd, ignore_errors=True)
    # ---- auto-index: two parallel PreToolUse calls start ONE build; over the cap -> too-big, nothing created in the repo; kill switch ----
    ah, ar = os.path.join(tmp, "ahome"), os.path.realpath(os.path.join(tmp, "arepo")); os.makedirs(ar); os.makedirs(ah); open(os.path.join(ar, "pyproject.toml"), "w").write(""); open(os.path.join(ar, "m.py"), "w").write("def m_fn():\n    return 1\n")
    aenv = {k: v for k, v in dict(env, HINTMAP_HOME=ah, HINTMAP_HOLDOUT_PCT="0", PATH="/usr/bin:/bin").items() if k != "HINTMAP_NO_AUTOINDEX"}
    aev = lambda cwd: json.dumps({"cwd": cwd, "tool_name": "Agent", "tool_use_id": "a1", "session_id": "s", "tool_input": {"subagent_type": "coder", "prompt": "fix m_fn"}})
    ahook = lambda cwd, e=aenv: subprocess.Popen([sys.executable, me, "hook", "pretooluse-agent"], cwd=cwd, env=e, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    ps = [ahook(ar) for _ in range(2)]; [p.communicate(aev(ar)) for p in ps]
    alog = lambda: [json.loads(l) for l in open(os.path.join(ah, "injections.jsonl")).read().splitlines()]
    assert sorted(x.get("autoindex") for x in alog()) == ["locked", "started"], alog()
    for _ in range(300):
        if os.path.exists(gpath(ar)) and ar in open(os.path.join(ah, "graphs")).read().split(): break
        time.sleep(0.1)
    else: raise AssertionError("auto-index never registered")
    assert not sh(["git", "status", "--porcelain"], cwd=ar) and open(os.path.join(ar, ".hintmap", ".gitignore")).read() == "*\n"
    br = os.path.realpath(os.path.join(tmp, "bigrepo")); os.makedirs(br); open(os.path.join(br, "pyproject.toml"), "w").write(""); open(os.path.join(br, "z.py"), "w").write("x = 1\n")
    p = ahook(br, dict(aenv, HINTMAP_AUTOINDEX_MAX="1")); p.communicate(aev(br))
    for _ in range(300):
        if any(x.get("skipped") == "too-big" for x in alog()): break
        time.sleep(0.1)
    else: raise AssertionError("no too-big log")
    assert not os.path.exists(os.path.join(br, ".hintmap"))
    kr = os.path.join(tmp, "killrepo"); os.makedirs(kr); open(os.path.join(kr, "pyproject.toml"), "w").write(""); p = ahook(kr, dict(aenv, HINTMAP_NO_AUTOINDEX="1")); p.communicate(aev(kr)); assert "autoindex" not in alog()[-1], alog()[-1]
    # ---- jit: def / refs / show (fresh after edits) and the Bash|Grep|Read hook ----
    jh, jr = os.path.join(tmp, "jhome"), os.path.realpath(os.path.join(tmp, "jrepo")); os.makedirs(jh); os.makedirs(jr)
    jenv = dict(env, HINTMAP_HOME=jh, HINTMAP_HOLDOUT_PCT="0"); jw = lambda rel, t: open(os.path.join(jr, rel), "w").write(t)
    jw("pyproject.toml", ""); jw("lib.py", "def jit_target(alpha, beta):\n    return alpha + beta\n"); jw("use.py", "from lib import jit_target\n\nprint(jit_target(1, 2))\n")
    jw("big.py", "".join(f"def big_fn_{i}(x):\n    return x + {i}\n\n" for i in range(220))); jw("small.py", "def tiny():\n    return 1\n"); jw("notes.md", "words go here\n" * 900); jw("data.json", json.dumps({"k%d" % i: "v" * 20 for i in range(900)}, indent=1))
    jx = lambda *args: (lambda r: r.stdout if r.returncode == 0 else (_ for _ in ()).throw(AssertionError((args, r.stdout, r.stderr))))(subprocess.run([sys.executable, me, *args], cwd=jr, env=jenv, capture_output=True, text=True))
    jx("graph", "init")
    d_ = jx("def", "jit_target"); assert "lib.py:1" in d_ and "def jit_target(alpha, beta)" in d_, d_
    r_ = jx("refs", "jit_target"); assert "use.py:1  from lib import jit_target" in r_ and "    :3  print(jit_target(1, 2))" in r_ and "lib.py:1" not in r_.split("\n", 1)[1], r_
    s_ = jx("show", "jit_target"); assert s_.splitlines()[1:] == ["    1  def jit_target(alpha, beta):", "    2      return alpha + beta"], s_
    assert "lib.py:1-2" in jx("show", "lib.py:2")
    time.sleep(0.05); jw("lib.py", "# edited after indexing\n\ndef jit_target(alpha, beta):\n    gamma = 3\n    return alpha + beta + gamma\n"); jw("use.py", "# new first line\nfrom lib import jit_target\n\nprint(jit_target(1, 2))\n")
    r2 = jx("refs", "jit_target"); assert "use.py:2  from lib import jit_target" in r2 and "    :4  print" in r2 and "use.py:1  " not in r2, r2  # lines come from the live file, no rebuild
    s2 = jx("show", "jit_target"); assert s2.splitlines()[1:] == ["    3  def jit_target(alpha, beta):", "    4      gamma = 3", "    5      return alpha + beta + gamma"], s2  # edited file re-indexed before answering
    assert "lib.py:3" in jx("def", "jit_target")
    def jhook(tn, ti, cwd=None, ev="pretooluse-bash", agent="ag1", **x):
        r = subprocess.run([sys.executable, me, "hook", ev], cwd=jr, env=dict(jenv, **x), input=json.dumps({"tool_name": tn, "tool_input": ti, "cwd": cwd or jr, "agent_id": agent, "session_id": "s1", "tool_use_id": "t1"}), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr; return json.loads(r.stdout)["hookSpecificOutput"] if r.stdout.strip() else None
    o = jhook("Bash", {"command": "rg -n jit_target ."}); assert o and "use.py:2" in o["additionalContext"] and "permissionDecision" not in o, o
    o = jhook("Grep", {"pattern": "jit_target"}); assert o and "def lib.py:3" in o["additionalContext"], o
    time.sleep(0.05); jw("lib.py", "# two\n# more lines above\n" + open(os.path.join(jr, "lib.py")).read())  # edited after the last re-index; the hook must not write graph.db
    o = jhook("Grep", {"pattern": "jit_target"}); assert o and "def lib.py:5" in o["additionalContext"] and "def lib.py:3" not in o["additionalContext"], o
    assert "lib.py:5" in jx("def", "jit_target")
    assert jhook("Bash", {"command": "rg -n jit_target ."}, HINTMAP_HOLDOUT_PCT="100") is None, "holdout arm stays silent"
    assert jhook("Bash", {"command": "rg -n no_such_symbol_zz ."}) is None and jhook("Bash", {"command": "rg -n 'jit_target|tiny' ."}) is None and jhook("Bash", {"command": "ls jit_target"}) is None
    assert jhook("Bash", {"command": "rg -n jit_target ."}, HINTMAP_JIT="0") is None and jhook("Bash", {"command": "rg -n jit_target ."}, HINTMAP_HOOK_DISABLED="1") is None
    o = jhook("Bash", {"command": "cat big.py"}); assert o is None, "default: big whole read allowed silently"
    o = jhook("Read", {"file_path": os.path.join(jr, "big.py")}, agent="ag0", HINTMAP_JIT_DENY="1"); assert o and o["permissionDecision"] == "deny", "HINTMAP_JIT_DENY=1 denies"
    o = jhook("Bash", {"command": "cat big.py"}, HINTMAP_JIT_DENY="1"); assert o and o["permissionDecision"] == "deny" and "hintmap show" in o["permissionDecisionReason"] and "sed -n" in o["permissionDecisionReason"] and "big_fn_" in o["permissionDecisionReason"], o
    assert jhook("Bash", {"command": "cat big.py"}, HINTMAP_JIT_DENY="1") is None, "the same agent repeating the call is let through"
    assert jhook("Read", {"file_path": os.path.join(jr, "big.py")}, HINTMAP_HOME=os.path.join(tmp, "jhome2")) is None  # other home: unregistered, allowed silently
    o = jhook("Read", {"file_path": os.path.join(jr, "big.py")}, HINTMAP_JIT_DENY="1"); assert o is None, "second whole read of the same file by the same agent passes"
    o = jhook("Read", {"file_path": os.path.join(jr, "big.py")}, agent="ag2", HINTMAP_JIT_DENY="1"); assert o and o["permissionDecision"] == "deny" and "offset/limit" in o["permissionDecisionReason"], o
    for tn, ti in (("Bash", {"command": "sed -n 1,50p big.py"}), ("Bash", {"command": "head -n 40 big.py"}), ("Bash", {"command": "cat small.py"}), ("Bash", {"command": "cat notes.md"}), ("Bash", {"command": "cat data.json"}), ("Bash", {"command": "cat big.py | head -20"}),
                   ("Read", {"file_path": os.path.join(jr, "big.py"), "limit": 80}), ("Read", {"file_path": os.path.join(jr, "big.py"), "offset": 10}), ("Read", {"file_path": os.path.join(jr, "small.py")}), ("Read", {"file_path": os.path.join(jr, "notes.md")}), ("Read", {"file_path": os.path.join(jr, "data.json")})):
        assert jhook(tn, ti, HINTMAP_HOLDOUT_PCT="0") is None, (tn, ti)
    jw("big2.py", "".join(f"def b2_{i}(x):\n    return x\n\n" for i in range(220)))
    o = jhook("Read", {"file_path": os.path.join(jr, "big2.py")}); assert o is None, "file created after indexing is not in the index: allowed"
    ur = os.path.realpath(os.path.join(tmp, "unindexed")); os.makedirs(ur); open(os.path.join(ur, "pyproject.toml"), "w").write(""); open(os.path.join(ur, "big.py"), "w").write(open(os.path.join(jr, "big.py")).read())
    assert jhook("Bash", {"command": "cat big.py"}, cwd=ur) is None and jhook("Bash", {"command": "rg -n tiny"}, cwd=ur) is None, "unindexed repo"
    sas = lambda inp, pct="0", **x: subprocess.run([sys.executable, me, "hook", "subagent-start"], cwd=jr, env=dict(jenv, HINTMAP_HOLDOUT_PCT=pct, **x), input=inp, capture_output=True, text=True).stdout
    sev = lambda cwd=jr: json.dumps({"hook_event_name": "SubagentStart", "agent_id": "ag1", "agent_type": "coder", "cwd": cwd, "session_id": "s1"})
    so = json.loads(sas(sev())); sc = so["hookSpecificOutput"]["additionalContext"]; assert so["hookSpecificOutput"]["hookEventName"] == "SubagentStart" and "hintmap q" in sc and "code-navigation" in sc and len(sc.splitlines()) <= 2, so  # SubagentStart pointer
    assert sas(sev(), "100") == "" and sas(sev(), HINTMAP_POINTER="0") == "" and sas(sev(), HINTMAP_HOOK_DISABLED="1") == "" and sas("garbage{") == "" and sas(sev(tmp)) == "", "holdout / off / disabled / error / unindexed: silent"
    sl = [json.loads(l) for l in open(os.path.join(jh, "injections.jsonl")) if '"subagent-start"' in l]; assert {r.get("arm") for r in sl} >= {"inject", "holdout"} and all("prompt" not in r for r in sl), sl
    assert subprocess.run([sys.executable, me, "hook", "pretooluse-bash"], cwd=jr, env=jenv, input="garbage{", capture_output=True, text=True).stdout == ""
    jl = [json.loads(l) for l in open(os.path.join(jh, "injections.jsonl")).read().splitlines()]; assert {x["kind"] for x in jl if x.get("event") == "pretooluse-bash"} >= {"grep-ctx", "read-deny", "read-big"} and all(x["arm"] in ("inject", "holdout") for x in jl if x.get("event") == "pretooluse-bash"), jl
    assert "JIT hook (pretooluse-bash)" in jx("report", "--jit-only")
    from hm.report import jit_ab  # join: by agent_id, or an exact tool_use_id; a short fake id in the log must not match every transcript
    jd = os.path.join(tmp, "jjoin"); os.makedirs(jd); jm = {}
    for nm_, body in (("agent-aaa", ""), ("agent-bbb", ""), ("agent-ccc", '{"id": "toolu_0123456789"}\nta ta ta\n'), ("agent-ddd", "ta ta ta\n")):
        open(os.path.join(jd, nm_ + ".jsonl"), "w").write(body); jm[os.path.join(jd, nm_ + ".meta.json")] = {}
    jrows = [{"event": "pretooluse-bash", "arm": "inject", "kind": "grep-ctx", "ts": "2026-10-09T10:00:00", "ms": 3, "agent_id": "aaa", "root": "/"}, {"event": "pretooluse-bash", "arm": "holdout", "kind": "grep-ctx", "ts": "2026-10-09T10:00:00", "ms": 3, "agent_id": "bbb", "root": "/"},
             {"event": "pretooluse-bash", "arm": "inject", "kind": "grep-ctx", "ts": "2026-10-09T10:00:00", "ms": 3, "tool_use_id": "ta"}, {"event": "pretooluse-bash", "arm": "holdout", "kind": "grep-ctx", "ts": "2026-10-09T10:00:00", "ms": 3, "tool_use_id": "toolu_0123456789"}]
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): jit_ab(jrows, jm, "")
    assert "inject 1  holdout 2" in buf.getvalue(), buf.getvalue()  # aaa -> inject, bbb -> holdout, ccc -> holdout (exact id), ddd unjoined
    # ---- nav: search / map / callees / changed / q ----
    se = jx("search", "jit_target", "alpha"); assert "lib.py:5  def jit_target(alpha, beta):" in se, se  # live line of the edited file, one-line context
    assert "lib.py" in jx("search", "--k", "1", "jit_target").splitlines()[0]
    jw("arms.py", "import hashlib\n\n\ndef pick(tid):\n    x = hashlib.sha1(tid).digest()[0]\n    arm = \"holdout\" if x < 100 else \"inject\"\n    return arm\n\n\ndef report():\n    return 1\n"); jw("armnote.py", "# the holdout list is printed here\ndef printer():\n    return 2\n"); jx("graph", "build")
    ts_ = jx("search", "--terms", "holdout", "arm", "assigned").splitlines()[:3]; assert any("arms.py:6  arm = " in l for l in ts_), ts_  # term search: the line holding two query terms ranks top 3 (NL search is the default engine)
    md = jx("map"); assert "lib.py" in md and "jit_target" in md and "big.py" in md and md.startswith("./: "), md
    assert "more files" in jx("map", "--budget", "1"), "budget cut is reported"
    mf = jx("map", "lib.py"); assert "5-7  def jit_target(alpha, beta)" in mf and "lib.py: 7 lines, 1 symbols" in mf, mf  # outline: line range + signature, no body
    assert "more symbols" in jx("map", "big.py", "--budget", "30") and "return x +" not in mf
    assert "nothing indexed at" in jx("map", "nope/none.py")
    jw("use.py", "from lib import jit_target\n\n\ndef caller():\n    return jit_target(3, 4)\n\n\nprint(jit_target(1, 2))\n")
    jx("graph", "build"); time.sleep(0.05); jw("use.py", "# shifted\n" + open(os.path.join(jr, "use.py")).read())
    ce = jx("callees", "caller"); assert "jit_target  lib.py:5  (called at :6)" in ce, ce  # use.py edited after indexing is re-indexed first; callee line is live
    jw("same.py", "def helper_one():\n    return 1\n\n\ndef helper_two():\n    return helper_one()\n\n\ndef driver_fn():\n    helper_one()\n    helper_two()\n    return jit_target(1, 2)\n"); jx("graph", "build")
    cs = jx("callees", "driver_fn"); assert "helper_one  same.py:1  (called at :10, same file)" in cs and "helper_two  same.py:5" in cs and "jit_target  lib.py:5" in cs, cs  # same-file calls are not edges
    assert "no symbol" in jx("callees", "nosuchsym_zz")
    assert "needs git" in jx("changed")
    if shutil.which("git"):
        gj = lambda *a_: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a_], cwd=jr, check=True, capture_output=True, text=True)
        gj("init", "-q"); gj("add", "-A"); gj("commit", "-qm", "base")
        assert "nothing" in jx("changed")
        time.sleep(0.05); jw("lib.py", open(os.path.join(jr, "lib.py")).read().replace("gamma = 3", "gamma = 4")); jw("fresh.txt", "x\n")
        ch = jx("changed"); assert "lib.py  +1 -1" in ch and "jit_target 5-7  callers: 3 (same.py:12, use.py:6" in ch and "fresh.txt  (new, untracked)" in ch, ch
        gj("add", "-A"); gj("commit", "-qm", "two"); assert "lib.py  +1 -1" in jx("changed", "HEAD~1") and "nothing" in jx("changed")
    assert jx("q", "jit_target").startswith("# q -> sym") and jx("q", "lib.py:5").startswith("# q -> show") and jx("q", "lib.py").startswith("# q -> map") and jx("q", ".").startswith("# q -> map")
    assert jx("q", "where", "is", "alpha", "added").startswith("# q -> search") and jx("q", "zzz_unknown_word").startswith("# q -> search")
    # ---- git history channel: ancestors of HEAD only (no future leakage), co-change, incremental, non-git no-op ----
    if shutil.which("git"):
        from hm.index import hist_build
        from hm.retrieve import hist_files
        hr = os.path.join(tmp, "histrepo"); os.makedirs(hr); g_ = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=hr, check=True, capture_output=True, text=True).stdout.strip()
        g_("init", "-q")
        def gc(files, msg):
            for f in files: open(os.path.join(hr, f), "a").write("x\n")
            g_("add", "-A"); g_("commit", "-qm", msg)
        gc(["a.py", "b.py"], "Fix zebra crossing handler"); old = g_("rev-parse", "HEAD"); gc(["c.py"], "Add quokka parser"); gc(["a.py", "b.py"], "Tweak zebra again")
        def hdb():
            d = sqlite3.connect(":memory:"); d.execute("CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT)"); return d
        os.environ["HM_HIST"] = "1"
        try:
            d = hdb(); g_("checkout", "-q", old); hist_build(d, hr)  # HEAD = base commit: later commits (quokka, zebra again) must not be visible
            assert hist_files(d, "quokka parser") == [] and [f for f, _ in hist_files(d, "zebra")] == ["a.py", "b.py"] or [f for f, _ in hist_files(d, "zebra")] == ["b.py", "a.py"], "future leak"
            assert d.execute("SELECT COUNT(*) FROM hist").fetchone()[0] == 1
            g_("checkout", "-q", "-"); hist_build(d, hr)  # incremental: only the 2 new commits
            assert d.execute("SELECT COUNT(*) FROM hist").fetchone()[0] == 3 and hist_files(d, "quokka")[0][0] == "c.py"
            nd = os.path.join(tmp, "nogit-hist"); os.makedirs(nd); d2 = hdb(); hist_build(d2, nd); assert d2.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='hist'").fetchone()[0] == 0, "non-git root must not create history tables"
        finally: os.environ.pop("HM_HIST", None)
    shutil.rmtree(tmp, ignore_errors=True); print("selftest ok")
