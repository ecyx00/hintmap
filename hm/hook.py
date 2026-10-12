"""hintmap.hook: see README Module ownership."""
import array, datetime, contextlib, functools, hashlib, heapq, itertools, json, math, os, re, signal, sqlite3, struct, sys, time
from hm.common import shutil, HOLDOUT_PCT, HOME, LOWPRI, MARKERS, ONE_THREAD, SELF, _T, abs_paths, argparse, glob, gopen, gpath, graph_root, h, mark, now, query_of, rrf, subprocess, subqueries
from hm.graph import graph_block
from hm.index import graph_changes, run_update
from hm.retrieve import retrieve

def wt_split(p):
    """(worktree root, main root) when p lies inside a linked git worktree: its `.git` FILE says `gitdir: <main>/.git/worktrees/<name>`. No git process. Else None."""
    d = os.path.realpath(p)
    while True:
        g = os.path.join(d, ".git")
        if os.path.isfile(g):
            try: gd = re.match(r"gitdir:\s*(.+)", open(g).read()).group(1).strip()
            except (OSError, AttributeError): return None
            gd = os.path.realpath(os.path.join(d, gd)); main = os.path.dirname(os.path.dirname(os.path.dirname(gd)))
            return (d, main) if os.path.basename(os.path.dirname(gd)) == "worktrees" and os.path.isdir(os.path.join(main, ".git")) else None
        if os.path.exists(g) or d == os.path.dirname(d): return None
        d = os.path.dirname(d)
def hook_root(cwd):
    """nearest ancestor holding .git (dir or worktree file), else nearest project marker (never $HOME, /, ~/Desktop). No git process (~10 ms)."""
    d = os.path.realpath(cwd or os.getcwd()); stop = {"/", os.path.expanduser("~"), os.path.expanduser("~/Desktop")}; up = []
    while True:
        up.append(d)
        if os.path.exists(os.path.join(d, ".git")): return (wt_split(d) or (0, d))[1]  # linked worktree -> its main checkout
        if d == os.path.dirname(d): break
        d = os.path.dirname(d)
    for d in up:
        if d in stop: break
        if any(os.path.exists(os.path.join(d, m)) for m in MARKERS): return d
    return ""
SCRATCH = ("/opt", "/usr", "/Library", "/System", "/tmp", "/private/tmp", "/var", "/private/var", os.path.expanduser("~/Library"))
def scratch(r):
    """system or scratch location (hook_root already maps a linked worktree to its main checkout, so a worktree under /tmp of a normal repo is not scratch)."""
    return any(r == x or r.startswith(x + "/") for x in SCRATCH)
def pick_root(prompt, cwd):
    """(root, root_from). A valid (non-scratch) cwd root wins when the brief names no resolvable path or names one under it (a brief citing a reference repo more often than the repo it works in still means cwd);
    only a brief whose paths all lie outside the cwd root decides by majority. No valid cwd root: the brief's majority, else nothing."""
    c = hook_root(cwd); c = "" if scratch(c) else c
    n = brief_roots(prompt)
    if c and (not n or c in n): return c, "cwd"
    r = top_root(n)
    return ("", "prompt-scratch") if r is None else (r, "prompt") if r else ("", "none")
def brief_roots(prompt):
    n = {}  # insertion order = first mention
    for _, c in abs_paths(prompt):
        r = hook_root(c if os.path.isdir(c) else os.path.dirname(c))
        if r: n[r] = n.get(r, 0) + 1
    return n
def top_root(n):
    r = next(iter(n), "")  # first mentioned
    return None if scratch(r) and r else r
def prompt_root(prompt):
    """root the brief's absolute paths mostly resolve to (ties: first mentioned). Returns "" when the brief names none, and None when the winning root is a scratch/system one (a /tmp copy, /opt/...): no list for the wrong repo."""
    return top_root(brief_roots(prompt))
def named_in(prompt, path):
    """the brief already names this file: its basename as a whole token (covers the relative path too)."""
    return re.search(r"(?<![\w.-])" + re.escape(os.path.basename(path)) + r"(?![\w-])", prompt) is not None
AUTO_MAX, AUTO_RETRY_S = int(os.environ.get("HINTMAP_AUTOINDEX_MAX") or 50000), 6 * 3600
def autoindex(root, log):
    """root without a registered graph: start ONE detached, niced `graph init` (O_EXCL lock in HOME, retried after AUTO_RETRY_S) and return at once; this call gets no code block.
    Needs a project marker; never $HOME, /, /tmp, ~/Library, ~/Desktop. The child refuses roots over AUTO_MAX files (logs `too-big`). Kill switch: HINTMAP_NO_AUTOINDEX=1."""
    if os.environ.get("HINTMAP_NO_AUTOINDEX") == "1" or os.path.exists(os.path.join(HOME(), "graph-disabled")) or graph_ready(root): return
    home = os.path.expanduser("~")
    if scratch(root): return
    if root in {"/", "/tmp", "/private/tmp", home, os.path.join(home, "Library"), os.path.join(home, "Desktop")} or not any(os.path.exists(os.path.join(root, m)) for m in (".git", *MARKERS)): return
    try:
        lock = os.path.join(HOME(), "autoindex", hashlib.sha1(root.encode()).hexdigest()[:12] + ".lock"); os.makedirs(os.path.dirname(lock), exist_ok=True)
        try: os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            if time.time() - os.path.getmtime(lock) < AUTO_RETRY_S: return log.update(autoindex="locked")
            os.utime(lock)
        spawn(["graph", "init", root, "--max-files", str(AUTO_MAX)], root); log["autoindex"] = "started"
    except Exception as e: log["autoindex_err"] = type(e).__name__
def hook_pretooluse(d, log):
    ti = d.get("tool_input") or {}
    log.update(agent_type=ti.get("subagent_type"), session_id=d.get("session_id"), tool_use_id=d.get("tool_use_id"))
    if d.get("tool_name") != "Agent" or log["agent_type"] not in os.environ.get("HINTMAP_AGENT_TYPES", "general-purpose").split(","): return log.update(skipped="not-target")
    prompt = ti.get("prompt", ""); root, log["root_from"] = pick_root(prompt, d.get("cwd"))
    if not root: return log.update(skipped="no-root")
    mark(log, "root"); log["root"] = root; autoindex(root, log); maybe_build(root, log, sync=True); mark(log, "walk")  # before retrieval so it reads the fresh index; also in the holdout arm (index freshness is not the treatment)
    log["prompt_sha1"] = hashlib.sha1(prompt.encode()).hexdigest()[:12]
    log["arm"] = "holdout" if int(hashlib.sha1(str(d.get("tool_use_id")).encode()).hexdigest()[:4], 16) % 100 < HOLDOUT_PCT else "inject"
    parts, hits = [], []
    ix = registered(root) and os.path.exists(gpath(root)); log["indexed"] = bool(ix)  # unregistered root: no code block at all, never indexed from a hook
    if ix:
        try: qs = subqueries(query_of(prompt)); log["subqueries"] = len(qs); parts = retrieve(root, qs, log); hits = [h for h in rrf(parts) if not named_in(prompt, h[0])][:5]
        except Exception as e: log["retr_err"] = type(e).__name__
    mark(log, "retrieve-rest")
    gl = None
    if ix and graph_ready(root):
        try: gl = graph_block(root, prompt, parts, log)
        except Exception as e: log["graph_err"] = type(e).__name__
    mark(log, "graph")
    cand = [re.match(r"\S+?(?::\d+(?:-\d+)?)?(?=\s|$)", t).group(0) for _, t in gl] if gl else [f"{p}:{lo}-{hi}" if lo else p for p, lo, hi, sg in hits]  # compact: path:lo-hi only (bench agent_e2e: same gain as full block, 40% smaller)
    new = [c for c in cand if not named_in(prompt, re.sub(r":\d+(?:-\d+)?$", "", c))]; log["dropped_named"] = len(cand) - len(new)
    def code():
        ls = list(new)
        while len(ls) > 1 and sum(map(len, ls)) > 1800: ls.pop()  # ~450 tokens for code lines
        return "## Code locations (hintmap, suggested starting points; verify before relying)\n" + "\n".join(ls) + "\n" if ls else ""
    blk = code()
    if log["arm"] == "holdout": return
    log["tokens"] = len(blk) // 4
    if not blk: return log.update(skipped="empty")
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": {**ti, "prompt": prompt + "\n\n" + blk.rstrip()}}}
def hook_session_start(d, log):
    """index refresh only (registered roots; no output)"""
    root = hook_root(d.get("cwd"))
    if not root: return
    log["root"] = root
    if registered(root): maybe_build(root, log)  # opt-in only: unregistered roots never get an index from a hook
def hook_subagent_start(d, log):
    """informs only, never blocks: <=2 short lines naming the optional tools, for a subagent whose cwd resolves to an indexed repo. Same holdout hash as the JIT hook (agent_id), arm logged."""
    log.update(agent_id=d.get("agent_id"), agent_type=d.get("agent_type"), session_id=d.get("session_id"))
    if os.environ.get("HINTMAP_POINTER") == "0": return log.update(skipped="pointer-off")
    root = hook_root(d.get("cwd"))
    if not root or scratch(root) or not graph_ready(root): return log.update(skipped="no-index")
    log["root"] = root; log["arm"] = "holdout" if int(hashlib.sha1(str(d.get("agent_id") or d.get("session_id")).encode()).hexdigest()[:4], 16) % 100 < HOLDOUT_PCT else "inject"
    if log["arm"] == "holdout": return
    return {"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext":
            "This repo is indexed by hintmap (optional): `hintmap q <anything>` routes to map/search/def+refs/show; also `hintmap map|search|refs|show`.\nSee the code-navigation skill for when each beats grep."}}
def cmd_hook(a):
    t0, log = time.time(), {}
    def die(*_): raise TimeoutError("hook cap")
    try:
        if os.environ.get("HINTMAP_HOOK_DISABLED") == "1" or os.path.exists(os.path.join(HOME(), "hook-disabled")): return 0
        signal.signal(signal.SIGALRM, die); signal.alarm(5); log["event"] = a.event
        log["boot_cpu_ms"] = int(time.process_time() * 1000); _T[:] = time.perf_counter(), time.process_time(); d = json.loads(sys.stdin.read()); mark(log, "parse")
        out = {"pretooluse-agent": hook_pretooluse, "session-start": hook_session_start, "subagent-start": hook_subagent_start}[a.event](d, log)
        if out: print(json.dumps(out)); sys.stdout.flush()
        mark(log, "emit")
    except BaseException as e: log.setdefault("skipped", type(e).__name__)
    finally:
        signal.alarm(0); tm = os.times(); log["cpu_ms"] = int((tm[0] + tm[1] + tm[2] + tm[3]) * 1000)
        if os.environ.get("HINTMAP_BENCH"): print(json.dumps(log), file=sys.stderr)  # bench-hook: stage timings back to the parent, no log line
        elif (log.get("agent_type") is not None and log.get("skipped") != "not-target") or log.get("build_spawned") or log.get("synced") is not None or a.event != "pretooluse-agent":
            try:
                os.makedirs(HOME(), exist_ok=True)
                with open(os.path.join(HOME(), "injections.jsonl"), "a") as f:
                    f.write(json.dumps({"ts": now(), "ms": int((time.time() - t0) * 1000), **log}) + "\n")
            except Exception: pass
    return 0
def cmd_bench_hook(a):
    """fresh-process PreToolUse hook calls against a registered root: p50/p95 wall + CPU (child rusage: interpreter start, imports, everything) and the median per-stage ms from the hook's own timers."""
    import resource, statistics
    root = graph_root(a.root); db = gopen(root, True)
    names = [r[0] for r in db.execute("SELECT name FROM symbols WHERE length(name) > 5 ORDER BY RANDOM() LIMIT ?", (3 * a.n,))]; db.close()
    env = {**os.environ, "HINTMAP_BENCH": "1", "HINTMAP_HOLDOUT_PCT": "0"}; W, C, T = [], [], {}
    for i in range(a.n):
        q = f"fix the bug in {' and '.join(names[3 * i:3 * i + 2]) or 'the main flow'} and update the tests"
        inp = json.dumps({"tool_name": "Agent", "tool_use_id": f"bench{i}", "cwd": root, "tool_input": {"subagent_type": "coder", "prompt": q}})
        if a.stale: stamp = os.path.join(root, ".hintmap", "walk.ok"); open(stamp, "a").close(); os.utime(stamp, (1e9, 1e9))
        r0, t0 = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
        p = subprocess.run([sys.executable, a.bin, "hook", "pretooluse-agent"], input=inp, capture_output=True, text=True, env=env)
        W.append((time.perf_counter() - t0) * 1000); r1 = resource.getrusage(resource.RUSAGE_CHILDREN); C.append((r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime) * 1000)
        try: L = json.loads(p.stderr.strip().splitlines()[-1])
        except (ValueError, IndexError): continue
        T.setdefault("boot", []).append([0, L.get("boot_cpu_ms", 0)])
        for k, v in L.get("t", {}).items(): T.setdefault(k, []).append(v)
    pct = lambda l, p: sorted(l)[min(len(l) - 1, int(p * len(l)))]
    print(f"{root}: n={a.n} files={nfiles(root)}  wall ms p50={pct(W, .5):.0f} p95={pct(W, .95):.0f}   cpu ms p50={pct(C, .5):.0f} p95={pct(C, .95):.0f}")
    print("stage (median ms, wall/cpu): " + "  ".join(f"{k}={statistics.median(x[0] for x in v):.1f}/{statistics.median(x[1] for x in v):.1f}" for k, v in T.items()))
def registered(root):
    try: return root in open(os.path.join(HOME(), "graphs")).read().splitlines()
    except OSError: return False
def graph_ready(root):
    return not os.path.exists(os.path.join(HOME(), "graph-disabled")) and registered(root) and os.path.exists(gpath(root))
def spawn(args, root):
    lk = [shutil.which("lockf"), "/tmp/heavy-bench.lock"] if shutil.which("lockf") and os.environ.get("HINTMAP_NO_HEAVY_LOCK") != "1" else []  # background builds queue behind other heavy jobs (and run niced via LOWPRI)
    subprocess.Popen([*lk, sys.executable, SELF, *args], cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, preexec_fn=LOWPRI, env={**os.environ, **ONE_THREAD})
SYNC_MAX, WALK_TTL, SMALL_FILES = 20, 30, 2000
def nfiles(root):
    try: return gopen(root, True).execute("SELECT COUNT(*) FROM files").fetchone()[0]
    except sqlite3.Error: return 0
def maybe_build(root, log, sync=False):
    """registered roots only. Lock held -> nothing (readers see changed files path-only). Stamp <WALK_TTL s old -> skip the stat-walk; older stamp (or none on a repo > SMALL_FILES) -> background refresh, no waiting.
    Only a missing stamp on a small repo walks inline. Simultaneous callers (parallel spawns): one claims `walk.run` (O_EXCL)
    and walks, the others sleep (0 CPU, <=2.5 s) until it is gone, then see the stamp. Walk: clean -> stamp; sync and <=SYNC_MAX changed -> re-index them now (stamp on success); else fire-and-forget `graph build`.
    Logs walk/synced/build_spawned. ponytail: a file edited inside WALK_TTL after a clean walk is seen by the next walk, not this one."""
    try:
        if not graph_ready(root): return
        lock, stamp, run = (os.path.join(root, ".hintmap", n) for n in ("build.lock", "walk.ok", "walk.run"))
        age = lambda p: time.time() - os.path.getmtime(p) if os.path.exists(p) else 1e9
        if age(lock) < 120: return log.update(walk="locked")
        if age(stamp) < WALK_TTL: return log.update(walk="skipped")
        if os.path.exists(stamp) or (log.update(nfiles=nfiles(root)) or log["nfiles"] > SMALL_FILES):  # stamp older than TTL, or no stamp on a big repo: the stat-walk (220-480 ms at 16k files) never blocks the hook;
            try: os.close(os.open(run, os.O_CREAT | os.O_EXCL | os.O_WRONLY))  # one refresher at a time (`graph build` stamps a clean walk and drops walk.run); changed files stay path-only (os.stat freshness in graph_block) until it lands
            except FileExistsError:
                if age(run) < 10: return log.update(walk="refreshing")
                os.utime(run)
            log["walk"] = "async"; return spawn(["graph", "build", root], root)
        try: os.close(os.open(run, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); mine = True
        except FileExistsError:
            t = time.time()
            while age(run) < 10 and time.time() - t < 2.5: time.sleep(0.02)
            if age(stamp) < WALK_TTL: return log.update(walk="shared")
            if age(lock) < 120: return log.update(walk="locked")
            mine = age(run) >= 10; mine and os.utime(run)  # a claimer that died leaves a stale file: take over
        try:
            db = gopen(root, True); ch = graph_changes(root, db); db.close(); log.update(walk="done", changed=len(ch))
            if not ch: return open(stamp, "w").close()  # stamp = "clean as of its mtime"; an empty file is the whole result
            if sync and len(ch) <= SYNC_MAX and "" not in ch:
                log["synced"] = run_update(root, ch, 2.5)
                if log["synced"]: open(stamp, "w").close()
                return
            log["build_spawned"] = ch[0][:120]; spawn(["graph", "build", root], root)
        finally:
            if mine and os.path.exists(run): os.unlink(run)
    except Exception as e: log["build_err"] = type(e).__name__
# ---- replay: joins hook injections to what the spawned agent actually edited (design 5.1) ----
def replay_pairs():
    out = []
    for ln in open(os.path.join(HOME(), "injections.jsonl")):
        d = json.loads(ln)
        if d.get("arm") not in ("inject", "holdout") or not d.get("tool_use_id") or not d.get("root"): continue
        mains = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{d['session_id']}.jsonl"))
        if not mains: continue
        brief = aid = None
        for l in open(mains[0]):
            if d["tool_use_id"] not in l: continue
            j = json.loads(l); m = j.get("message") or {}
            for c in m.get("content") if isinstance(m.get("content"), list) else []:
                if c.get("type") == "tool_use" and c.get("id") == d["tool_use_id"]: brief = (c.get("input") or {}).get("prompt")
                if c.get("type") == "tool_result" and c.get("tool_use_id") == d["tool_use_id"] and (mm := re.search(r"agentId: (\w+)", json.dumps(c.get("content")))): aid = mm.group(1)
        edited = set()
        af = os.path.join(os.path.dirname(mains[0]), d["session_id"], "subagents", f"agent-{aid}.jsonl") if aid else ""
        if af and os.path.exists(af):
            for l in open(af):
                if '"Edit"' in l or '"Write"' in l or '"MultiEdit"' in l:
                    for c in (json.loads(l).get("message") or {}).get("content") or []:
                        if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") in ("Edit", "Write", "MultiEdit") and (fp := (c.get("input") or {}).get("file_path", "")).startswith(d["root"] + "/"):
                            edited.add(fp[len(d["root"]) + 1:])
        out.append(dict(id=d["tool_use_id"], arm=d["arm"], root=d["root"], brief=brief, agent=aid, edited=edited))
    return out
def cmd_graph_replay(a):
    pairs = replay_pairs()
    if a.join:
        for p in pairs: print(f"{p['id']} {p['arm']} {p['root']} agent={p['agent']} edited={len(p['edited'])} brief={'yes' if p['brief'] else 'no'}")
    got = [p for p in pairs if p["edited"] and p["brief"]]
    if a.count or a.join:
        print(f"pairs={len(pairs)} with_edits={len(got)} holdout={sum(p['arm'] == 'holdout' for p in got)} inject={sum(p['arm'] == 'inject' for p in got)} roots={sorted({p['root'] for p in got})}")
        if a.count: return
    res = {"E-lex": [], "E": [], "F": []}
    for p in got:
        root = p["root"]; log = {}
        if not (registered(root) and os.path.exists(gpath(root))): continue
        try:
            qs = subqueries(query_of(p["brief"])); fl = lambda parts: [h[0] for h in rrf(parts) if h[0] not in p["brief"]]
            pe = retrieve(root, qs, log, "both"); arms = {"E-lex": fl(retrieve(root, qs, log, "lex")), "E": fl(pe), "F": [x for x, _ in (graph_block(root, p["brief"], pe, log) or [])]}
        except Exception as e: print("skip", p["id"], type(e).__name__); continue
        gold = {f for f in p["edited"] if os.path.exists(os.path.join(root, f))}
        if not gold: continue
        for arm, lst in arms.items(): res[arm].append(tuple(len(gold & set(lst[:k])) / len(gold) for k in (5, 10)))
    for arm, r in res.items(): print(f"arm {arm}: n={len(r)}" + (f" R@5={sum(x[0] for x in r) / len(r):.2f} R@10={sum(x[1] for x in r) / len(r):.2f}" if r else ""))  # old semble arms A/B/D: run the .bak-standalone file
