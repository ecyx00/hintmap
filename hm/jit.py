"""hintmap.jit: see README Module ownership. def / refs / show (answers read from the live file, never stale) and the Bash|Grep|Read PreToolUse hook that offers them at the tool call."""
import hashlib, json, os, re, shlex, signal, sqlite3, sys, time
from hm.common import HOME, HOLDOUT_PCT, graph_root, gopen, gpath, now, ntok
from hm.graph import resolve, cands
from hm.index import run_update

CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".swift", ".go", ".rs", ".zig", ".c", ".h", ".cc", ".cpp", ".hpp", ".m", ".mm", ".java", ".kt", ".scala", ".rb", ".php", ".cs", ".lua", ".sh", ".ex", ".exs", ".hs"}
BIG_LINES = 400  # whole-file read of a code file at/above this many lines is logged as read-big (denied only with HINTMAP_JIT_DENY=1) (about 16 KB / 4k tokens; the live median whole read was ~37 KB and 211 of them were 7.9 MB)
BIG_BYTES = 8000  # below this size the line count is not even taken
IDENT = re.compile(r"^[A-Za-z_$][\w$]{2,}$")
# ---- core: where is X defined / used, answered from the index + the live file ----
def stale(root, path, size, mtime_ns):
    try: st = os.stat(os.path.join(root, path))
    except OSError: return True
    return (st.st_size, st.st_mtime_ns) != (size, mtime_ns)
def freshen(root, db, paths):
    """re-index the indexed files whose (size, mtime) differ from the index (CLI only: writes graph.db). Returns a reopened read-only db."""
    old = {r["path"]: (r["size"], r["mtime_ns"]) for p in set(paths) for r in db.execute("SELECT path, size, mtime_ns FROM files WHERE path=?", (p,))}
    bad = [p for p, (sz, m) in old.items() if stale(root, p, sz, m)]
    if bad:
        db.close(); run_update(root, bad, 20); db = gopen(root, True)
    return db
def disk_lines(root, path):
    try: return open(os.path.join(root, path), errors="replace").read().splitlines()
    except OSError: return []
def sigtext(ls, lo):
    """the definition's first line, continued while parentheses stay open (max 4 lines)."""
    out, depth = [], 0
    for l in ls[lo - 1:lo + 3]:
        out.append(l.strip()); depth += l.count("(") + l.count("[") - l.count(")") - l.count("]")
        if depth <= 0: break
    return re.sub(r"\s*[{:]\s*$", "", " ".join(out))[:200]
def defline_rx(name): return re.compile(r"\b(def|class|function|func|fn|struct|enum|interface|type|protocol|trait|const|let|var|val|actor)\s+" + re.escape(name) + r"(?![\w$])|(?<![\w$])" + re.escape(name) + r"\s*(?:=|:)\s*(?:function|\(|async)")
def name_of(arg): return arg.rpartition("::")[2].rpartition(".")[2]
def find_refs(root, db, name, max_files=60):
    """-> (defs [(path, lo, kind)], hits {path: [(line, text)]}, nfiles_total). Candidate files come from the index (symbol rows + ref rows, definition file first, then by use count);
    the lines always come from the file on disk, so an edited file is never shown with old line numbers. Files created after the index was built are not candidates."""
    defs = [(r["path"], r["lo"], r["kind"], r["role"], r["size"], r["mtime_ns"]) for r in db.execute("SELECT f.path, s.lo, s.kind, f.role, f.size, f.mtime_ns FROM symbols s JOIN files f ON f.id=s.file_id WHERE s.name=? AND s.kind!='route' ORDER BY f.role='test', f.path, s.lo", (name,))]
    cnt = {p: n for p, n in db.execute("SELECT f.path, SUM(r.n) FROM refs r JOIN files f ON f.id=r.file_id WHERE r.name=? GROUP BY f.id", (name,))}
    sd = {}  # definition lines of files edited since indexing come from the live file (hook path: no re-index); no match -> no line number
    for d in list(defs):
        if stale(root, d[0], d[4], d[5]):
            if d[0] not in sd:
                dr = defline_rx(name)
                sd[d[0]] = [i for i, l in enumerate(disk_lines(root, d[0]), 1) if dr.search(l)]
            defs.remove(d)
            defs += [(d[0], i, d[2], d[3], d[4], d[5]) for i in sd[d[0]]] or [(d[0], None, d[2], d[3], d[4], d[5])]
    defs = sorted(dict.fromkeys(defs), key=lambda d: (d[3] == "test", d[0], d[1] or 0))
    fts = {}  # files whose chunks hold the identifier as text (strings, comments, docs, config): the ref rows only cover code identifiers
    parts, wholes = ntok(name); tok = (wholes or parts or [""])[0]
    if tok:
        try:
            for p, n in db.execute("SELECT f.path, COUNT(*) FROM (SELECT rowid FROM ix WHERE ix MATCH ? LIMIT 4000) x JOIN chunks c ON c.id=x.rowid JOIN files f ON f.id=c.file_id GROUP BY f.id", (f'body:"{tok}"',)): fts[p] = n
        except sqlite3.Error: pass
    dp = {d[0] for d in defs}; order = sorted(dp | set(cnt) | set(fts), key=lambda p: (p not in dp, -(cnt.get(p) or 0), -fts.get(p, 0), p))
    rx = re.compile(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])"); hits = {}; dl = {(d[0], d[1]) for d in defs}
    for p in order[:max_files]:
        for i, l in enumerate(disk_lines(root, p), 1):
            if rx.search(l) and (p, i) not in dl: hits.setdefault(p, []).append((i, l.strip()[:110]))
    return defs, hits, len(order)
def fmt_refs(name, defs, hits, nfiles, limit, per_file, under=None):
    ranked = sorted(hits, key=lambda p: (any(d[0] == p for d in defs) is False, p in {d[0] for d in defs if d[3] == "test"}, -len(hits[p])))
    if under: ranked.sort(key=lambda p: not any(p == u or p.startswith(u.rstrip("/") + "/") for u in under))  # the searched path first; the rest stays reachable (shown or in `also in`)
    total = sum(len(hits[p]) for p in ranked); lines = []; shown = 0; sfiles = 0
    for p in ranked:
        if shown >= limit: break
        ls = hits[p][:per_file]; sfiles += 1
        for k, (i, t) in enumerate(ls):
            if shown >= limit: break
            lines.append((f"{p}:{i}" if k == 0 else f"    :{i}") + "  " + t); shown += 1
    head = f"{name}: " + (f"def {defs[0][0]}" + (f":{defs[0][1]}" if defs[0][1] else "") + (f" (+{len(defs) - 1} more defs)" if len(defs) > 1 else "") if defs else "no definition in the index") + f"; {total} uses in {len(ranked)} files"
    cut = total - shown; hidden = [p for p in ranked if p not in {l.split(":")[0] for l in lines if not l.startswith("    :")}]
    tail = f"\n... +{cut} more uses" + (f"; also in: {', '.join(hidden[:12])}" + (f" (+{len(hidden) - 12})" if len(hidden) > 12 else "") if hidden else "") + f" (hintmap refs {name} --all)" if cut > 0 else ""
    return head + "\n" + "\n".join(lines) + tail
# ---- CLI ----
def cmd_def(a):
    root = graph_root(None); db = gopen(root, True); fs, ss, _ = resolve(db, a.symbol)
    if fs: return print(f"{a.symbol} is a file ({fs[0]['path']}); `def` takes a symbol")
    if not ss: return print(f"no symbol {a.symbol!r} in the index")
    db = freshen(root, db, [r["path"] for r in ss]); _, ss, _ = resolve(db, a.symbol); cache = {}
    if not ss: return print(f"no symbol {a.symbol!r} in the index (removed since indexing)")
    for r in ss[:10]:
        ls = cache.setdefault(r["path"], disk_lines(root, r["path"])); print(f"{r['path']}:{r['lo']}  {r['kind']}  {sigtext(ls, r['lo'])}")
    if len(ss) > 10: print(f"... +{len(ss) - 10} more definitions")
def cmd_refs(a):
    root = graph_root(None); db = gopen(root, True); nm = name_of(a.symbol); defs, hits, n = find_refs(root, db, nm, 10 ** 6 if a.all else 60)
    if not defs and not hits: return print(f"no symbol or references for {nm!r} in the index")
    print(fmt_refs(nm, defs, hits, n, 10 ** 6 if a.all else a.limit, 10 ** 6 if a.all else 4))
def span_of(db, root, arg):
    """-> (path, lo, hi, note) | str (message). arg = path:line | symbol | path::symbol; files edited since indexing are re-indexed first."""
    m = re.match(r"^(.+?):(\d+)$", arg)
    if m and not arg.count("::"):
        row = db.execute("SELECT path FROM files WHERE path=? OR path LIKE ?", (m.group(1), "%/" + m.group(1))).fetchone()
        if not row: return f"no indexed file {m.group(1)!r}"
        db = freshen(root, db, [row["path"]]); ln = int(m.group(2))
        s = db.execute("SELECT s.lo, s.hi FROM symbols s JOIN files f ON f.id=s.file_id WHERE f.path=? AND s.kind!='route' AND s.lo<=? AND s.hi>=? ORDER BY s.hi-s.lo LIMIT 1", (row["path"], ln, ln)).fetchone()
        return (row["path"], s["lo"], s["hi"], "") if s else (row["path"], max(1, ln - 15), ln + 15, "no symbol spans that line; window")
    fs, ss, _ = resolve(db, arg)
    if fs: return f"{arg} is a file; use path:line or a symbol"
    if not ss: return f"no symbol {arg!r} in the index"
    db = freshen(root, db, [r["path"] for r in ss]); fs, ss, amb = resolve(db, arg)
    if amb:
        nt = [r for r in ss if r["role"] != "test"]
        if len(nt) == 1: ss = nt
        else: return f"{len(ss)} definitions named {arg}; pick one with show <path:line>:\n{cands(ss)}"
    if not ss: return f"no symbol {arg!r} in the index (removed since indexing)"
    return ss[0]["path"], ss[0]["lo"], ss[0]["hi"], ""
def cmd_show(a):
    root = graph_root(None); db = gopen(root, True); r = span_of(db, root, a.target)
    if isinstance(r, str): return print(r)
    path, lo, hi, note = r; ls = disk_lines(root, path); hi = min(hi, len(ls)); cap = lo + a.max - 1
    print(f"{path}:{lo}-{hi}" + (f"  ({note})" if note else ""))
    for i in range(lo, min(hi, cap) + 1): print(f"{i:5d}  {ls[i - 1]}")
    if hi > cap: print(f"... +{hi - cap} more lines (sed -n {cap + 1},{hi}p {path})")
# ---- hook: PreToolUse on Bash | Grep | Read ----
LONG_ARG = {"-g", "--glob", "-t", "--type", "-T", "--type-not", "-m", "--max-count", "-f", "--file", "-j", "--threads", "--max-depth", "-d", "--directories", "-E", "--encoding", "--iglob", "--sort", "--sortr", "-M", "--max-columns", "--ignore-file", "-A", "-B", "-C", "--context", "--after-context", "--before-context", "--include", "--exclude", "--exclude-dir", "--include-dir", "--type-add"}
def grep_ident(cmd):
    """-> (identifier, [paths], cwd|None) when the command is one rg/grep of a single identifier (optionally after `cd X &&`, optionally piped on), else None."""
    try: lx = shlex.shlex(cmd, posix=True, punctuation_chars=True); lx.whitespace_split = True; toks = list(lx)
    except ValueError: return None
    segs, cur = [], []
    for t in toks:
        if t and set(t) <= set("|&;<>()"):
            segs.append((cur, t)); cur = []
        else: cur.append(t)
    segs.append((cur, ""))
    cwd = None
    if segs and segs[0][0][:1] == ["cd"] and len(segs[0][0]) == 2 and segs[0][1] == "&&" and len(segs) > 1: cwd = segs[0][0][1]; segs = segs[1:]
    a, op = segs[0]
    if op == ">" and a and a[-1].isdigit(): a = a[:-1]
    if not a or os.path.basename(a[0]) not in ("rg", "grep", "egrep", "ripgrep") or op in ("<", "(", ")"): return None
    pats, paths, i = [], [], 1; ex = False
    while i < len(a):
        t = a[i]
        if t == "--": paths += a[i + 1:]; break
        if t.startswith("--"):
            if t in ("--regexp",) and i + 1 < len(a): pats.append(a[i + 1]); ex = True; i += 1
            elif "=" in t and t.startswith("--regexp="): pats.append(t.split("=", 1)[1]); ex = True
            elif t in LONG_ARG: i += 1
        elif t.startswith("-") and len(t) > 1:
            for k, c in enumerate(t[1:], 1):
                if c == "e":
                    ex = True
                    if k == len(t) - 1 and i + 1 < len(a): pats.append(a[i + 1]); i += 1
                    else: pats.append(t[k + 1:])
                    break
                if f"-{c}" in LONG_ARG:
                    if k == len(t) - 1: i += 1
                    break
        else: (paths if (pats or ex) else pats).append(t)
        i += 1
    if len(pats) != 1: return None
    p = re.sub(r"^(\\b|\\<)|(\\b|\\>)$", "", pats[0])
    return (p, paths, cwd) if IDENT.match(p) else None
def read_cmd(cmd):
    """-> (path, cwd|None) for a bare whole-file dump (`cat F`, `head -n N F` with N >= BIG_LINES, `head -N F`), else None. Anything piped, redirected, ranged or multi-file is left alone."""
    if re.search(r"[|&;<>`$()]", cmd.replace("&&", "").strip()): 
        m0 = re.match(r"^\s*cd\s+(\S+)\s*&&\s*(.*)$", cmd)
        if not m0 or re.search(r"[|&;<>`$()]", m0.group(2)): return None
    try: toks = shlex.split(cmd)
    except ValueError: return None
    cwd = None
    if len(toks) > 3 and toks[0] == "cd" and toks[2] == "&&": cwd = toks[1]; toks = toks[3:]
    if len(toks) == 2 and toks[0] == "cat" and not toks[1].startswith("-"): return toks[1], cwd
    if toks and toks[0] == "head":
        n, rest = None, []
        i = 1
        while i < len(toks):
            t = toks[i]
            if t == "-n" and i + 1 < len(toks) and toks[i + 1].lstrip("+").isdigit(): n = int(toks[i + 1]); i += 1
            elif re.fullmatch(r"-n\d+|-\d+", t): n = int(t.lstrip("-n"))
            elif t.startswith("-"): return None
            else: rest.append(t)
            i += 1
        if len(rest) == 1 and n is not None and n >= BIG_LINES: return rest[0], cwd
    return None
def code_lines(path):
    """line count of a code file above BIG_BYTES, else 0 (not code / small / unreadable)."""
    if os.path.splitext(path)[1] not in CODE_EXT: return 0
    try:
        if os.path.getsize(path) < BIG_BYTES: return 0
        with open(path, "rb") as f: return f.read().count(b"\n") + 1
    except OSError: return 0
def top_syms(db, path, n=6):
    return db.execute("SELECT s.name, s.lo, s.hi FROM symbols s JOIN files f ON f.id=s.file_id WHERE f.path=? AND s.kind!='route' AND s.parent_id IS NULL ORDER BY s.hi-s.lo DESC LIMIT ?", (path, n)).fetchall()
def arm_of(d):
    k = str(d.get("agent_id") or d.get("session_id") or d.get("tool_use_id"))  # per agent (else session), so one task sits in one arm
    return "holdout" if int(hashlib.sha1(k.encode()).hexdigest()[:4], 16) % 100 < HOLDOUT_PCT else "inject"
def jit_decide(d, log):
    """-> hook output dict | None. Fills `log` only when the hook would act (also in the holdout arm, where it stays silent)."""
    tn, ti, cwd = d.get("tool_name"), d.get("tool_input") or {}, d.get("cwd") or os.getcwd()
    ident = rpath = None; paths = []
    if tn == "Bash":
        cmd = ti.get("command") or ""
        if not cmd or ("rg" not in cmd and "grep" not in cmd and "cat" not in cmd and "head" not in cmd): return
        g = grep_ident(cmd)
        if g: ident, paths, cd = g
        else:
            r = read_cmd(cmd)
            if not r: return
            rpath, cd = r
        if cd: cwd = cd if os.path.isabs(cd) else os.path.join(cwd, cd)
    elif tn == "Grep":
        p = ti.get("pattern") or ""
        if not IDENT.match(p): return
        ident, paths = p, [ti["path"]] if ti.get("path") else []
    elif tn == "Read":
        if ti.get("offset") or ti.get("limit") or not ti.get("file_path"): return
        rpath = ti["file_path"]
    else: return
    if rpath:
        ap = rpath if os.path.isabs(rpath) else os.path.join(cwd, rpath)
        n = code_lines(ap)
        if n < BIG_LINES: return
    from hm.hook import graph_ready, hook_root
    root = hook_root(os.path.dirname(os.path.realpath(ap)) if rpath else cwd)
    if not root or not graph_ready(root): return
    db = gopen(root, True); log.update(root=root, arm=arm_of(d))
    if ident:
        if not db.execute("SELECT 1 FROM symbols WHERE name=? AND kind!='route' LIMIT 1", (ident,)).fetchone(): return log.update(skipped="unknown-symbol")
        under = []
        for p in paths:
            ap = os.path.realpath(p if os.path.isabs(p) else os.path.join(cwd, p))
            if ap == root or ap.startswith(root + "/"): under.append(os.path.relpath(ap, root))
            else: return
        under = [u for u in under if u != "."]
        defs, hits, nf = find_refs(root, db, ident, 41)
        if nf > 40: return log.update(skipped="too-common")  # a name in dozens of files: a capped list is noise, the agent's own grep is the better tool
        txt = fmt_refs(ident, defs, hits, nf, 14, 3, under)
        log.update(kind="grep-ctx", ident=ident, out=len(txt)); msg = {"hookEventName": "PreToolUse", "additionalContext": "hintmap (index, read from the live files): " + txt}
    else:
        rel = os.path.relpath(os.path.realpath(ap), root)
        if rel.startswith(".."): return
        row = db.execute("SELECT 1 FROM files WHERE path=?", (rel,)).fetchone()
        if not row: return
        if os.environ.get("HINTMAP_JIT_DENY") != "1": return log.update(kind="read-big", path=rel, lines=n, bytes=os.path.getsize(ap))  # default: allowed silently, logged for the adoption report
        sy = top_syms(db, rel); log.update(kind="read-deny", path=rel, lines=n, bytes=os.path.getsize(ap))
        alt = (" Symbols here: " + ", ".join(f"{s['name']} ({s['lo']}-{s['hi']})" for s in sy) + "." if sy else "")
        msg = {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason":
               f"{rel} has {n} lines (~{log['bytes'] // 1024} KB); a whole-file dump is the costliest way to look at it.{alt} Instead: `hintmap show <Symbol>` or `hintmap show {rel}:<line>` (one symbol, numbered), `sed -n A,Bp {rel}`, or Read with offset/limit "
               f"(a ranged Read also satisfies Edit). If you really need all of it, repeat the same call once and it will go through."}
        sk = os.path.join(HOME(), "jit-seen"); key = os.path.join(sk, hashlib.sha1(f"{d.get('agent_id') or d.get('session_id')}|{rel}".encode()).hexdigest()[:20])
        if os.path.exists(key): return log.update(kind="read-allow-repeat")
        if arm_of(d) != "holdout":
            os.makedirs(sk, exist_ok=True); open(key, "w").close()
    if log["arm"] == "holdout": return
    return {"hookSpecificOutput": msg}
def cmd_jit_hook(event=None):
    """fail-open: HINTMAP_JIT=0, a disabled hook, an unindexed repo or any error prints nothing."""
    t0, log = time.time(), {}
    try:
        if os.environ.get("HINTMAP_JIT") == "0" or os.environ.get("HINTMAP_HOOK_DISABLED") == "1" or os.path.exists(os.path.join(HOME(), "hook-disabled")): return 0
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("cap"))); signal.alarm(2)
        d = json.loads(sys.stdin.read()); out = jit_decide(d, log)
        if out: print(json.dumps(out)); sys.stdout.flush()
    except BaseException as e: log.setdefault("skipped", type(e).__name__)
    finally:
        signal.alarm(0)
        if log.get("kind"):
            try:
                tm = os.times()
                with open(os.path.join(HOME(), "injections.jsonl"), "a") as f:
                    f.write(json.dumps({"ts": now(), "ms": int((time.time() - t0) * 1000), "event": "pretooluse-bash", "cpu_ms": int((tm[0] + tm[1] + tm[2] + tm[3]) * 1000),
                                        "tool": d.get("tool_name"), "agent_id": d.get("agent_id"), "session_id": d.get("session_id"), "tool_use_id": d.get("tool_use_id"), **log}) + "\n")
            except Exception: pass
    return 0
