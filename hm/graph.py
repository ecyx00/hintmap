"""hintmap.graph: see README Module ownership."""
import array, datetime, contextlib, functools, hashlib, heapq, itertools, json, math, os, re, signal, sqlite3, struct, sys, time
from hm.common import COMMON, gopen, gpath, graph_root, h, norm, query_of, rrf, subqueries
from hm.index import CONTAINER, run_update, stem
from hm.retrieve import noisy

# ---- impact / edit-check / tests-for: read the graph. Plain interpreter + sqlite; edit-check re-indexes through the venv child. ----
def sym_rows(db, where, args):
    return db.execute(f"SELECT s.id, s.name, s.kind, s.lo, s.hi, s.parent_id, s.ar, s.sg, f.id fid, f.path, f.role FROM symbols s JOIN files f ON f.id=s.file_id WHERE {where}", args).fetchall()
def resolve(db, arg):
    """name | path | path::name -> (files, syms, ambiguous). A bare arg that is (a suffix of) an indexed path is a file target; else a symbol name. Several defs of one name -> ambiguous (callers list candidates)."""
    path, _, nm = arg.partition("::") if "::" in arg else (None, "", arg)
    fl = lambda p: db.execute("SELECT id, path, role FROM files WHERE path=? OR path LIKE ?", (p, "%/" + p)).fetchall()
    if path is None and (fs := fl(arg)): return fs, [], False
    fs = fl(path) if path is not None else []
    if path is not None and not nm: return fs, [], False
    ss = sym_rows(db, "s.name=? AND s.kind!='route'", (nm,)) or sym_rows(db, "s.name=?", (nm,))
    if path is not None: ss = [r for r in ss if r["fid"] in {f["id"] for f in fs}]
    top = [r for r in ss if r["parent_id"] is None or any(CONTAINER.search(p["kind"]) for p in sym_rows(db, "s.id=?", (r["parent_id"],)))]
    ss = top or ss
    return [], ss, len({r["fid"] for r in ss}) > 1 or len(ss) > 1
def cands(ss): return "\n".join(f"  {r['path']}::{r['name']}  ({r['kind']}, line {r['lo']})" for r in ss[:20])
def reach(db, fids, sids, depth):
    """transitive reverse dependencies over graph edges: [(depth, file_id, sym_id|None, tier(worst on the path), kind, line, via)]. Hubs are listed but not expanded (barrier); the target is always expanded."""
    byd, byf = {}, {}
    for r in db.execute("SELECT src_file, src_sym, dst_file, dst_sym, kind, tier, line FROM edges"):
        byf.setdefault(r[2], []).append(r)
        if r[3] is not None: byd.setdefault(r[3], []).append(r)
    hub = {f for f, in db.execute("SELECT file_id FROM nodes WHERE hub=1")}; psym = {}
    def top(sid):  # a caller nested inside a def: climb to the nearest edge-eligible symbol (top-level def or method)
        while sid is not None:
            r = psym.get(sid) or psym.setdefault(sid, db.execute("SELECT s.id, s.parent_id, p.kind pk FROM symbols s LEFT JOIN symbols p ON p.id=s.parent_id WHERE s.id=?", (sid,)).fetchone())
            if r is None or r["parent_id"] is None or CONTAINER.search(r["pk"] or ""): return sid
            sid = r["parent_id"]
    seen = {("s", i) for i in sids} | {("f", i) for i in fids}; front = [(("s", i), 1) for i in sids] + [(("f", i), 1) for i in fids]; out = []; d = 0
    while front and d < depth:
        d += 1; nxt = []
        for (k, i), wt in front:
            for sf, ss, df, ds, kind, tier, line in (byd if k == "s" else byf).get(i, []):
                ss = top(ss) if ss is not None else None; n = ("s", ss) if ss is not None else ("f", sf)
                if n in seen: continue
                seen.add(n); t = max(tier, wt); out.append((d, sf, ss, t, kind, line, (k, i)))
                if sf not in hub: nxt.append((n, t))
        front = nxt
    return out
def gline(root, cache, path, line):
    if path not in cache:
        try: cache[path] = open(os.path.join(root, path), errors="replace").read().splitlines()
        except OSError: cache[path] = []
    ls = cache[path]; return ls[line - 1].strip()[:120] if 0 < line <= len(ls) else ""
def cmd_graph_impact(a):
    t0 = time.time(); root = graph_root(None); db = gopen(root, True); fs, ss, amb = resolve(db, a.target)
    if amb: return print(f"ambiguous: {len(ss)} definitions named {a.target}; pick one as path::name:\n{cands(ss)}")
    if not fs and not ss: return print(f"no symbol or file {a.target!r} in the index")
    pth = dict(db.execute("SELECT id, path FROM files")); role = dict(db.execute("SELECT id, role FROM files")); hub = {f for f, in db.execute("SELECT file_id FROM nodes WHERE hub=1")}
    sname = {r[0]: (r[1], r[2]) for r in db.execute("SELECT id, name, lo FROM symbols")}
    rows = reach(db, [f["id"] for f in fs], [r["id"] for r in ss], a.depth); cache, lv = {}, {}
    for d, sf, sid, t, kind, line, via in rows:
        lv.setdefault(d, []).append({"path": pth[sf], "line": line, "sym": sname[sid][0] if sid else None, "tier": t, "kind": kind, "hub": sf in hub, "test": role[sf] == "test", "via": pth[via[1]] if via[0] == "f" else sname[via[1]][0]})
    for l in lv.values(): l.sort(key=lambda x: (x["tier"], x["path"]))
    tg = [f"{r['path']}::{r['name']}" for r in ss] or [f["path"] for f in fs]
    if a.json:
        for l in lv.values():
            for x in l: x["text"] = gline(root, cache, x["path"], x["line"])
        return print(json.dumps({"target": tg, "depth": a.depth, "levels": lv, "ms": int((time.time() - t0) * 1000)}, ensure_ascii=False))
    print(f"impact of {', '.join(tg)} (depth {a.depth}; T1 = import-resolved, T2 = unique-name; hubs listed, not expanded)")
    if not lv: print("  no dependents in the index (callers outside the indexed scope are invisible)")
    for d in sorted(lv):
        print(f"depth {d}: {len(lv[d])}")
        for x in lv[d][:a.limit]: print(f"  T{x['tier']} {x['path']}:{x['line']}" + (f"  in {x['sym']}()" if x["sym"] else "") + (f"  <- {x['via']}" if d > 1 else "") + f"  {gline(root, cache, x['path'], x['line'])}" + ("  [HUB]" if x["hub"] else "") + ("  [test]" if x["test"] else ""))
        if len(lv[d]) > a.limit: print(f"  ... +{len(lv[d]) - a.limit} more (--limit)")
    tests = sorted({x["path"] for l in lv.values() for x in l if x["test"]})
    if tests: print("tests reached: " + ", ".join(tests))
def cmd_graph_tests(a):
    root = graph_root(None); db = gopen(root, True); fs, ss, amb = resolve(db, a.target)
    if amb: return print(f"ambiguous: {len(ss)} definitions named {a.target}; pick one as path::name:\n{cands(ss)}")
    if not fs and not ss: return print(f"no symbol or file {a.target!r} in the index")
    pth = dict(db.execute("SELECT id, path FROM files")); found = {}  # path -> (rank, why)
    def put(p, rank, why):
        if p not in found or rank < found[p][0]: found[p] = (rank, why)
    for d, sf, sid, t, kind, line, via in reach(db, [f["id"] for f in fs], [r["id"] for r in ss], a.depth):
        if db.execute("SELECT role FROM files WHERE id=?", (sf,)).fetchone()[0] == "test": put(pth[sf], d, f"depth {d} T{t} {kind} line {line}")
    names = {r["name"] for r in ss}; tf = [f["path"] for f in fs] + [r["path"] for r in ss]
    for n in names:  # textual refs the edge builder dropped (ambiguous names): any test file that mentions the name
        for p, ln in db.execute("SELECT f.path, MIN(r.line) FROM refs r JOIN files f ON f.id=r.file_id WHERE r.name=? AND f.role='test' GROUP BY f.id", (n,)): put(p, 5, f"ref {n} line {ln}")
    for p, in db.execute("SELECT path FROM files WHERE role='test'"):  # name convention: test_x.py / x.test.ts / x_test.go
        b = re.sub(r"(^test_|_test$|\.test$|\.spec$|^test$)", "", os.path.splitext(os.path.basename(p))[0].replace(".test", "").replace(".spec", ""))
        if any(b and b == stem(t) for t in tf): put(p, 6, "name convention")
    if a.json: return print(json.dumps([{"path": p, "why": w} for p, (r, w) in sorted(found.items(), key=lambda x: x[1])], ensure_ascii=False))
    if not found: print(f"no test files linked to {a.target} (tests outside the indexed scope are invisible: `hintmap graph stats` shows scope)")
    for p, (r, w) in sorted(found.items(), key=lambda x: (x[1][0], x[0])): print(f"{p}  ({w})")
def cmd_graph_editcheck(a):
    root = graph_root(None); db = gopen(root, True); fs, ss, amb = resolve(db, a.symbol)
    if amb: return print(f"ambiguous: {len(ss)} definitions named {a.symbol}; pick one as path::name:\n{cands(ss)}")
    if fs: return print("edit-check takes a symbol (name or path::name), not a file")
    nm = a.symbol.rpartition("::")[2]
    if ss: path, bef = ss[0]["path"], (ss[0]["ar"], ss[0]["sg"])
    else:  # not in the index: maybe removed by an earlier re-index
        rec = db.execute("SELECT path FROM sigchg WHERE name=?", (nm,)).fetchall()
        if len(rec) != 1: return print(f"no symbol {a.symbol!r} in the index" + ("" if not rec else f" (several recorded: {', '.join(r[0] for r in rec)})"))
        path, bef = rec[0][0], None
    db.close(); st = os.stat(os.path.join(root, path)) if os.path.exists(os.path.join(root, path)) else None
    ok = run_update(root, [path], 20) if st or bef else False
    db = gopen(root, True); aft = [r for r in sym_rows(db, "f.path=? AND s.name=? AND s.kind!='route' ORDER BY s.lo", (path, nm))][:1]
    rec = db.execute("SELECT old_ar, old_sg, new_ar, new_sg FROM sigchg WHERE path=? AND name=?", (path, nm)).fetchone()
    cache = {}; head = f"edit-check {path}::{nm}"
    if not ok: print(f"{head}: WARNING re-index did not run (lock held / no venv / timeout); using the last indexed state")
    old = (rec["old_ar"], rec["old_sg"]) if rec else bef
    if aft and aft[0]["lo"]: new = (aft[0]["ar"], aft[0]["sg"])
    else: new = None
    callers = []
    if aft: callers = [("T%d" % e["tier"], e["sp"], e["line"]) for e in db.execute("SELECT e.tier, e.line, f.path sp FROM edges e JOIN files f ON f.id=e.src_file WHERE e.kind='refs' AND e.dst_sym=? ORDER BY e.tier, f.path", (aft[0]["id"],))]
    else: callers = [("T?", p, ln) for p, ln in db.execute("SELECT f.path, r.line FROM refs r JOIN files f ON f.id=r.file_id WHERE r.name=? AND f.path!=? ORDER BY f.path", (nm, path))]  # symbol gone: textual refs by name
    if new is None: verdict = "REMOVED or renamed: every textual reference by name may break"
    elif old is None or old == new: verdict = "signature unchanged" + (f" (arity {new[0]})" if old else " (no earlier baseline recorded)")
    else: verdict = f"signature changed ({'arity %d -> %d' % (old[0], new[0]) if old[0] != new[0] else 'arity %d, parameter/return text differs' % new[0]})"
    print(f"{head}: {verdict}")
    if aft: print(f"  now: {gline(root, cache, path, aft[0]['lo'])}")
    if verdict.startswith("signature unchanged"): print(f"  {len(callers)} caller(s) in the index; none affected by the signature" if callers else "  no callers in the index")
    elif not callers: print("  no callers in the index (T1/T2); callers outside the indexed scope are invisible")
    else:
        print(f"  {len(callers)} call site(s) that may break:")
        for t, p, ln in callers[:a.limit]: print(f"  {t} {p}:{ln}  {gline(root, cache, p, ln)}")
        if len(callers) > a.limit: print(f"  ... +{len(callers) - a.limit} more")
    if rec and not a.keep:
        w = sqlite3.connect(gpath(root), timeout=30); w.execute("DELETE FROM sigchg WHERE path=? AND name=?", (path, nm)); w.commit(); w.close()  # consumed
# ---- Faz 1: graph in the hook (design 2.3/2.4/6.2). Hook-side code is stdlib+sqlite only, read-only db. ----
MAX_LINES, NDIRECT, NGRAPH, GRAPH_BUDGET_S, ALPHA, ITERS = 8, 5, 2, 0.3, 0.5, 2
def graph_block(root, prompt, parts, log):
    """Seeds = files named in the brief + every retriever hit, per brief part; 2-iteration seeded diffusion over file-level edges (reverse 1.0, forward 0.5, hubs absorb but do not spread);
    part-quota selection; <=8 lines `path:line  role`; os.stat freshness per emitted file. -> [(path, text)] or None when the graph has nothing."""
    t0, c0 = time.monotonic(), time.process_time(); db = gopen(root, True)
    if not db.execute("SELECT 1 FROM files LIMIT 1").fetchone(): return None
    class Rows(dict):  # key -> row, fetched on first use (26k-file repos: loading the whole files table twice cost ~40 ms)
        def __init__(s, sql, f): super().__init__(); s.sql, s.f = sql, f
        def __missing__(s, k):
            r = db.execute(s.sql, (k,)).fetchone()
            if r is None: raise KeyError(k)
            s[k] = v = s.f(r); return v
        def __contains__(s, k):
            try: s[k]; return True
            except KeyError: return False
        def get(s, k, d=None): return s[k] if k in s else d
    files = Rows("SELECT id, mtime_ns, size FROM files WHERE path=?", tuple); lang = Rows("SELECT lang FROM files WHERE path=?", lambda r: r[0]); path_of = Rows("SELECT path FROM files WHERE id=?", lambda r: r[0])
    hubs = {r[0] for r in db.execute("SELECT file_id FROM nodes WHERE hub=1")}
    texts = subqueries(query_of(prompt))
    def glob_(*pats): return [r[0] for r in db.execute("SELECT path FROM files WHERE " + " OR ".join(["path GLOB ?"] * len(pats)), pats)]  # query tokens are [\w./-]: no GLOB metacharacters
    def like(tok): return glob_("*" + tok)
    def by_base(tok): return [p for p in glob_(tok, "*/" + tok) if os.path.basename(p) == tok]
    def by_stem(tok): return [p for p in glob_(tok + ".*", "*/" + tok + ".*") if os.path.splitext(os.path.basename(p))[0] == tok]
    def named(t):
        out = {}
        for tok in set(re.findall(r"[\w./-]*\w\.\w{1,5}\b", t)):
            tok = tok.strip("./"); c = [p for p in like(tok) if p == tok or p.endswith("/" + tok)] if "/" in tok else by_base(tok)
            if len(c) == 1: out[files[c[0]][0]] = 1.0
        for tok in set(re.findall(r"[A-Za-z_]\w{4,}", t)):
            if tok in COMMON or not ("_" in tok or re.search(r"[a-z][A-Z]", tok)): continue
            if len(st := by_stem(tok)) == 1: out.setdefault(files[st[0]][0], 1.0)
            elif 0 < len(fs := {r[0] for r in db.execute("SELECT file_id FROM symbols WHERE name=?", (tok,))}) <= 4:
                for f in fs: out[f] = max(out.get(f, 0), 1.0 / len(fs))
        return out
    seeds, hitinfo, named_ids = [], {}, {files[r[0]][0] for r in db.execute("SELECT path FROM files WHERE instr(?, path) > 0", (prompt,))}
    for t, hl in zip(texts, parts):
        nm = named(t); sd = dict(nm); named_ids |= set(nm)
        for rank, h in enumerate(hl):
            if h[0] in files: f = files[h[0]][0]; sd[f] = max(sd.get(f, 0), 0.5 / (rank + 1)); hitinfo.setdefault(f, h)
        seeds.append(sd)
    log["seeds_named"], log["seeds_hit"] = len(named_ids), len(hitinfo)
    allseed = {f for sd in seeds for f in sd}
    if not allseed: return None
    E = {}
    def load(ids):
        ids = list(ids); q = ",".join("?" * len(ids))
        for r in db.execute(f"SELECT e.src_file, e.dst_file, e.kind, e.tier, e.w, e.line, s.name, s.lo FROM edges e LEFT JOIN symbols s ON s.id=e.dst_sym WHERE e.src_file IN ({q}) OR e.dst_file IN ({q}) LIMIT 4000", ids + ids):
            k, cur = (r[0], r[1]), E.get((r[0], r[1]))
            if cur is None or (r[2] == "refs", r[4]) > (cur[0] == "refs", cur[2]): E[k] = tuple(r[2:])
    load(allseed); hop1 = {x for k in E for x in k} - allseed - hubs
    if hop1 and time.monotonic() - t0 < GRAPH_BUDGET_S: load(hop1)
    adj = {}
    for (a, b), (kind, tier, w, line, dn, dlo) in E.items(): w *= 1 if kind == "refs" else 0.5; adj.setdefault(b, []).append((a, w)); adj.setdefault(a, []).append((b, 0.5 * w))  # imports are a weaker signal than uses
    for u, l in adj.items():
        tot = sum(w for _, w in l); adj[u] = [] if u in hubs else [(v, w / tot) for v, w in l]
    xs = []
    for sd in seeds:
        z = sum(sd.values()) or 1; p0 = {k: v / z for k, v in sd.items()}; x = dict(p0)
        for _ in range(ITERS):
            nx = {k: (1 - ALPHA) * v for k, v in p0.items()}
            for u, m in x.items():
                for v, w in adj.get(u, ()): nx[v] = nx.get(v, 0) + ALPHA * m * w
            x = nx
        xs.append(x)
    qwords = set(norm(prompt)); score, rank_lists, tot = {}, [], {}
    for sd, x in zip(seeds, xs):
        r = sorted((f for f in x if f not in named_ids and f not in hubs and not noisy(path_of[f], lang.get(path_of[f]), qwords)), key=lambda f: -x[f])[:30]; rank_lists.append(r)
        for i, f in enumerate(r): score[f] = score.get(f, 0) + 1 / (60 + i + 1)
        for f, m in x.items(): tot[f] = tot.get(f, 0) + m
    for f in score:  # bridge bonus: reached from >=2 parts without being a seed there
        n = sum(1 for sd, x in zip(seeds, xs) if f in x and f not in sd)
        if n >= 2: score[f] += (n - 1) / 61
    chosen = []
    def take(f, lim):  # no per-directory cap: it dropped gold siblings (gin tree.go behind gin.go/context.go/routes_test.go)
        if f not in chosen and len(chosen) < lim: chosen.append(f)
    ok = lambda p: p in files and not noisy(p, lang.get(p), qwords)  # named files stay eligible: a path:line range beats a bare name
    for hl in parts[:4]:  # direct slots (<=NDIRECT): part quota first (each part's best retriever hit), then the fused file-rank order
        for h in hl:
            if ok(h[0]): take(files[h[0]][0], NDIRECT); break
    for h in rrf(parts):
        if ok(h[0]): take(files[h[0]][0], NDIRECT)
    nd = len(chosen); glim = min(MAX_LINES, nd + NGRAPH)  # then <=NGRAPH graph lines: the best-retrieved direct neighbour (caller or callee) of a named file, then one diffusion line per part / by score
    fo = {h[0]: i for i, h in enumerate(rrf(parts))}; nb = {}
    for (a, b), info in E.items():
        o = b if a in named_ids else a if b in named_ids else None
        if o is not None and o not in named_ids and o not in hubs and not noisy(path_of[o], lang.get(path_of[o]), qwords): nb[o] = max(nb.get(o, 0), info[2])
    for o in sorted(nb, key=lambda o: (fo.get(path_of[o], 99), -nb[o]))[:1]: take(o, glim)
    for r in rank_lists:
        for f in r:
            if f not in chosen: take(f, glim); break
    for f in sorted(score, key=lambda f: -score[f]):
        if not noisy(path_of[f], lang.get(path_of[f]), qwords): take(f, glim)
    for h in rrf(parts):  # graph lines never displace the fused order: slots they leave free go to the next retrieval hits
        if ok(h[0]): take(files[h[0]][0], MAX_LINES)
    lines, tiers, stale, ngraph = [], {}, 0, 0
    for f in chosen:  # direct slots first, graph neighbours after
        p = path_of[f]; best = None
        for (a, b), info in E.items():
            o = b if a == f else a if b == f else None
            if o is not None and (v := tot.get(o, 0) * info[2] * (1 if info[0] == 'refs' else 0.5)) and (best is None or (o in named_ids, v) > (best[0] in named_ids, best[1])): best = (o, v, a == f, info)
        h = hitinfo.get(f)
        if h and (f in chosen[:nd] or best is None or best[0] not in named_ids): txt = f"{p}:{h[1]}-{h[2]}" + (f"  {h[3][:100]}" if h[3] else "") if h[1] else p
        elif best:
            o, _, fwd, (kind, tier, w, line, dn, dlo) = best; g = os.path.basename(path_of[o]); tiers[tier] = tiers.get(tier, 0) + 1; ngraph += 1
            txt = (f"{p}:{line}  ← calls {dn} ({g})" if kind == "refs" else f"{p}:{line}  ← imports {g}") if fwd else (f"{p}:{dlo}  → defines {dn} (used by {g})" if kind == "refs" else f"{p}  → imported by {g}")
        else: txt = p
        try: st = os.stat(os.path.join(root, p)); fresh = st.st_mtime_ns == files[p][1] and st.st_size == files[p][2]
        except OSError: fresh = False
        if not fresh and txt != p: txt, stale = p, stale + 1
        lines.append((p, txt[:200]))
    log.update(graph_ms=int((time.monotonic() - t0) * 1000), graph_cpu_ms=int((time.process_time() - c0) * 1000), graph_n=ngraph, tiers=tiers, stale_dropped=stale, emitted=[p for p, _ in lines])
    return lines
