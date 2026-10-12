"""hintmap.nav: see README Module ownership. search / map / callees / changed / q: orientation and navigation commands over the index. Line numbers always come from the live file (jit.py freshness); every listing has a budget and says how much it cut."""
import os, re, subprocess, sys
from hm.common import graph_root, gopen, query_of, rrf, subqueries
from hm.graph import cands, resolve
from hm.jit import IDENT, defline_rx, disk_lines, find_refs, fmt_refs, freshen, name_of, sigtext, stale, cmd_show
from hm.retrieve import retrieve
from hm.term import term_search

def live_line(root, path, name, lo, size, mt, cache):
    """the definition line of `name` in `path`: the index value when the file is unchanged since indexing, else found in the live file (None when it cannot be found)."""
    if not stale(root, path, size, mt): return lo
    ls = cache.setdefault(path, disk_lines(root, path)); rx = defline_rx(name)
    return next((i for i, l in enumerate(ls, 1) if rx.search(l)), None)
def rel_to(root, arg):
    """argument (absolute, or relative to the cwd) -> repo-relative path when it lies under root, else the argument unchanged."""
    ap = os.path.realpath(arg)
    return ("" if ap == root else os.path.relpath(ap, root)) if ap == root or ap.startswith(root + "/") else arg
# ---- search ----
def nl_rows(root, db, text):
    """the BM25 + vector + symbol engine (retrieve()) -> ([(path, line|None, context)] one row per file, total files)."""
    hits = rrf(retrieve(root, subqueries(query_of(text)))); meta = {r["path"]: (r["size"], r["mtime_ns"]) for h in hits[:40] for r in db.execute("SELECT path, size, mtime_ns FROM files WHERE path=?", (h[0],))}; cache = {}; out = []
    for p, lo, hi, label in hits[:40]:
        sz, mt = meta.get(p, (0, 0)); name = label.split()[-1] if label else ""
        ln = lo if lo and not stale(root, p, sz, mt) else (live_line(root, p, name, lo, sz, mt, cache) if name else None)
        ls = cache.setdefault(p, disk_lines(root, p)); out.append((p, ln, (ls[ln - 1].strip() if ln and ln <= len(ls) else "") or label))
    return out, len(hits)
def tm_rows(root, db, text):
    cache = {}; out = []
    for p, ln, tx, sc, kind in term_search(root, db, text):
        if ln and tx is None: ls = cache.setdefault(p, disk_lines(root, p)); tx = ls[ln - 1].strip() if ln <= len(ls) else ""
        out.append((p, ln, tx or ("(path match)" if kind == "path" else "")))
    return out
def fused_rows(a, b):
    """RRF over files of two row lists; the row shown per file is its first row in a, else in b."""
    sc, first = {}, {}
    for lst in (a, b):
        seen = set()
        for r in lst:
            if r[0] in seen: continue
            seen.add(r[0]); sc[r[0]] = sc.get(r[0], 0) + 1 / (60 + len(seen)); first.setdefault(r[0], r)
    return [first[p] for p in sorted(sc, key=lambda p: -sc[p])]
def search_rows(root, db, text, mode):
    """mode nl | term | both -> [(path, line|None, context)]."""
    if mode == "nl": return nl_rows(root, db, text)[0]
    if mode == "term": return tm_rows(root, db, text)
    return fused_rows(tm_rows(root, db, text), nl_rows(root, db, text)[0])
SEARCH_MODE = os.environ.get("HINTMAP_SEARCH") or "nl"  # nl | term | both
def cmd_search(a):
    root = graph_root(None); db = gopen(root, True); text = " ".join(a.text); rows = search_rows(root, db, text, "term" if getattr(a, "terms", False) else os.environ.get("HINTMAP_SEARCH") or SEARCH_MODE)
    if not rows: return print(f"no hits for {text!r} in the index")
    for p, ln, ctx in rows[:a.k]: print(f"{p}:{ln}  {ctx[:100]}" if ln else f"{p}  {ctx[:100]}")
    if len(rows) > a.k: print(f"... +{len(rows) - a.k} more (hintmap search --k {a.k * 2} ...)")
# ---- map ----
def dir_files(db, d):
    """-> rows (id, path, loc, role, indeg) of the indexed files under directory d ('' = whole repo)."""
    lo, hi = (d + "/", d + "0") if d else ("", "\U0010ffff")
    return db.execute("SELECT f.id, f.path, f.loc, f.role, COALESCE(n.indeg, 0) indeg FROM files f LEFT JOIN nodes n ON n.file_id=f.id WHERE f.path>=? AND f.path<?", (lo, hi)).fetchall()
def top_names(db, fid, n=5):
    rows = db.execute("SELECT name, lo, hi FROM symbols WHERE file_id=? AND kind!='route' AND parent_id IS NULL ORDER BY hi-lo DESC LIMIT ?", (fid, n)).fetchall()
    return [r["name"] for r in sorted(rows, key=lambda r: r["lo"])]
def map_dir(db, d, budget):
    rows = dir_files(db, d)
    if not rows: return f"no indexed files under {d or '.'}"
    pre = d + "/" if d else ""; sub = {}
    for r in rows:
        rest = r["path"][len(pre):]
        if "/" in rest: sub[rest.split("/", 1)[0]] = sub.get(rest.split("/", 1)[0], 0) + 1
    nd = sum(1 for r in rows if "/" not in r["path"][len(pre):])
    out = [f"{d or '.'}/: {len(rows)} files ({nd} directly here), {len(sub)} subdirs" + (": " + ", ".join(f"{k}/ {v}" for k, v in sorted(sub.items(), key=lambda kv: -kv[1])[:14]) + (f", +{len(sub) - 14} more" if len(sub) > 14 else "") if sub else "")]
    size = len(out[0]) + 1; limit = budget * 4; shown = 0
    for r in sorted(rows, key=lambda r: (r["role"] == "test", -r["indeg"], r["path"])):
        names = top_names(db, r["id"]); l = f"{r['path']}  {r['loc'] or 0}L  " + ", ".join(names)
        if size + len(l) + 1 > limit and shown: break
        out.append(l.rstrip()); size += len(l) + 1; shown += 1
    if shown < len(rows): out.append(f"... +{len(rows) - shown} more files (ranked by in-degree, tests last; hintmap map <subdir> or --budget N)")
    return "\n".join(out)
def map_file(root, db, path, budget):
    db = freshen(root, db, [path]); fid = db.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()["id"]; ls = disk_lines(root, path)
    sy = db.execute("SELECT id, name, kind, lo, hi, parent_id FROM symbols WHERE file_id=? AND kind!='route' ORDER BY lo, hi DESC", (fid,)).fetchall(); par = {r["id"]: r["parent_id"] for r in sy}
    out = [f"{path}: {len(ls)} lines, {len(sy)} symbols"]
    if not sy and path.endswith((".md", ".markdown")):  # docs: the headings are the outline
        sy = [(i, l.strip()) for i, l in enumerate(ls, 1) if re.match(r"#{1,4} \S", l)]
        rows = [f"{i}  {t[:100]}" for i, t in sy]
    else:
        rows = []
        for r in sy:
            depth = 0; p = r["parent_id"]
            while p is not None and depth < 4: depth += 1; p = par.get(p)
            rows.append(f"{r['lo']}-{min(r['hi'], len(ls))}  " + "  " * depth + sigtext(ls, r["lo"]))
    size = len(out[0]) + 1; limit = budget * 4; shown = 0
    for l in rows:
        if size + len(l) + 1 > limit and shown: break
        out.append(l); size += len(l) + 1; shown += 1
    if shown < len(rows): out.append(f"... +{len(rows) - shown} more symbols (hintmap map {path} --budget N, or show <symbol>)")
    elif not rows: out.append("(no symbols in the index; hintmap show " + path + ":<line>)")
    return "\n".join(out)
def map_target(db, root, arg):
    """-> ('dir', rel) | ('file', rel) | None. Paths relative to the cwd or the repo root; a unique suffix of an indexed path also counts."""
    rel = rel_to(root, arg).strip("/") if arg not in (None, "", ".") else ""
    if rel == "": return ("dir", "")
    if db.execute("SELECT 1 FROM files WHERE path=?", (rel,)).fetchone(): return ("file", rel)
    if db.execute("SELECT 1 FROM files WHERE path>=? AND path<? LIMIT 1", (rel + "/", rel + "0")).fetchone(): return ("dir", rel)
    r = db.execute("SELECT path FROM files WHERE path LIKE ? LIMIT 2", ("%/" + rel,)).fetchall()
    if len(r) == 1: return ("file", r[0]["path"])
    r = db.execute("SELECT path FROM files WHERE path LIKE ? LIMIT 1", ("%/" + rel + "/%",)).fetchone()
    if r:
        i = r["path"].index("/" + rel + "/"); return ("dir", r["path"][:i + 1 + len(rel)].lstrip("/"))
    return None
def cmd_map(a):
    root = graph_root(None); db = gopen(root, True); t = map_target(db, root, a.target)
    if not t: return print(f"nothing indexed at {a.target!r}" + (" (exists on disk but is not in the index: new, ignored or unsupported; run hintmap graph build)" if os.path.exists(a.target) else ""))
    print(map_dir(db, t[1], a.budget) if t[0] == "dir" else map_file(root, db, t[1], a.budget))
# ---- callees ----
def cmd_callees(a):
    root = graph_root(None); db = gopen(root, True); fs, ss, amb = resolve(db, a.symbol)
    if amb:
        nt = [r for r in ss if r["role"] != "test"]
        if len(nt) == 1: ss, amb = nt, False
    if amb: return print(f"{len(ss)} definitions named {a.symbol}; pick one as path::name:\n{cands(ss)}")
    if not fs and not ss: return print(f"no symbol or file {a.symbol!r} in the index")
    path = fs[0]["path"] if fs else ss[0]["path"]; db = freshen(root, db, [path])
    if ss:  # re-read the definition: the file may have been re-indexed
        ss = [r for r in resolve(db, f"{path}::{ss[0]['name']}")[1] if r["lo"] == ss[0]["lo"]] or resolve(db, f"{path}::{ss[0]['name']}")[1]
        if not ss: return print(f"no symbol {a.symbol!r} in the index (removed since indexing)")
    if ss:
        s = ss[0]; ids = [r[0] for r in db.execute("SELECT id FROM symbols WHERE file_id=? AND lo>=? AND hi<=?", (s["fid"], s["lo"], s["hi"]))]
        where, args, head = f"e.src_sym IN ({','.join('?' * len(ids))})", ids, f"{s['name']} ({path}:{s['lo']}-{s['hi']}) calls"
    else: where, args, head = "e.src_file=?", [fs[0]["id"]], f"{path} uses"
    rows = db.execute(f"SELECT e.line, e.tier, e.kind, d.name dname, d.lo dlo, df.path dpath, df.size dsize, df.mtime_ns dmt, ef.path dfpath, ef.size efsz, ef.mtime_ns efmt FROM edges e JOIN files ef ON ef.id=e.dst_file LEFT JOIN symbols d ON d.id=e.dst_sym LEFT JOIN files df ON df.id=d.file_id WHERE {where} ORDER BY e.tier, e.line", args).fetchall()
    seen, out, cache = set(), [], {}
    if ss:  # edges are cross-file only; same-file calls come from the ref rows of the symbol's span
        for r in db.execute(f"SELECT d.name, d.lo, MIN(r.line) line FROM refs r JOIN symbols d ON d.file_id=r.file_id AND d.name=r.name AND d.kind!='route' WHERE r.sym_id IN ({','.join('?' * len(ids))}) AND d.id NOT IN ({','.join('?' * len(ids))}) AND length(r.name)>=4 GROUP BY d.id ORDER BY MIN(r.line)", ids + ids):
            seen.add((path, r["name"])); out.append(f"{r['name']}  {path}:{r['lo']}  (called at :{r['line']}, same file)")
    for r in rows:
        key = (r["dpath"] or r["dfpath"], r["dname"])
        if key in seen: continue
        seen.add(key)
        if r["dname"]:
            ln = live_line(root, r["dpath"], r["dname"], r["dlo"], r["dsize"], r["dmt"], cache); out.append(f"{r['dname']}  {r['dpath']}" + (f":{ln}" if ln else "") + f"  (called at :{r['line']})")
        else: out.append(f"{r['dfpath']}  ({r['kind']} at :{r['line']})")
    if not out: return print(f"{head} nothing the index resolves (edges are only kept for names that resolve to an indexed definition)")
    print(f"{head} {len(out)} indexed targets")
    print("\n".join(out[:a.limit]) + (f"\n... +{len(out) - a.limit} more (--limit N)" if len(out) > a.limit else ""))
# ---- changed ----
def git(root, *args):
    try: r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired): return None
    return r.stdout if r.returncode == 0 else None
def parse_diff(txt):
    """unified -U0 diff -> {path: {'ranges': [(lo, hi)], 'add': n, 'del': n}} (new-side line ranges; a pure deletion is the line after it)."""
    out, cur = {}, None
    for l in txt.splitlines():
        if l.startswith("+++ "): cur = None if l[4:] == "/dev/null" else out.setdefault(l[6:], {"ranges": [], "add": 0, "del": 0})
        elif l.startswith("@@") and cur is not None:
            m = re.match(r"@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", l)
            if m: n = 1 if m.group(3) is None else int(m.group(3)); c = int(m.group(2)); cur["ranges"].append((c, c + max(n, 1) - 1)); cur["add"] += n; cur["del"] += 1 if m.group(1) is None else int(m.group(1))
    return out
def cmd_changed(a):
    root = graph_root(None); db = gopen(root, True)
    if git(root, "rev-parse", "--is-inside-work-tree") is None: return print("changed: needs git (not a git work tree here or git is missing); nothing to compare against")
    base = f"{a.rev}..HEAD" if a.rev else "HEAD"
    txt = git(root, "diff", "-U0", "--no-color", "--no-ext-diff", "--relative", base)
    if txt is None: txt = git(root, "diff", "-U0", "--no-color", "--no-ext-diff", "--relative") if not a.rev else None  # no commits yet: unstaged only
    if txt is None: return print(f"changed: git could not diff {base}")
    new = [] if a.rev else [p for p in (git(root, "ls-files", "-o", "--exclude-standard") or "").splitlines() if p]; nnew = len(new); new = new[:8]
    ch = parse_diff(txt)
    if not ch and not new: return print(f"changed vs {base}: nothing")
    idx = [p for p in ch if db.execute("SELECT 1 FROM files WHERE path=?", (p,)).fetchone()]
    db = freshen(root, db, idx); tot_a = sum(v["add"] for v in ch.values()); tot_d = sum(v["del"] for v in ch.values()); lines = []; nsym = 0; cache = {}
    for p in sorted(ch):
        v = ch[p]; f = db.execute("SELECT id, role FROM files WHERE path=?", (p,)).fetchone(); head = f"{p}  +{v['add']} -{v['del']}"
        if not f: lines.append(head + "  (not in the index)"); continue
        sy = [r for r in db.execute("SELECT id, name, lo, hi, parent_id FROM symbols WHERE file_id=? AND kind!='route' ORDER BY lo", (f["id"],))]
        hit = [r for r in sy if any(r["lo"] <= hi and r["hi"] >= lo for lo, hi in v["ranges"])]
        hit_ids = {r["id"] for r in hit}; parents = {r["parent_id"] for r in hit}; leaf = [r for r in hit if r["id"] not in parents]  # innermost symbols only
        outside = [rg for rg in v["ranges"] if not any(r["lo"] <= rg[1] and r["hi"] >= rg[0] for r in sy)]
        lines.append(head + (f"  (+ top-level lines {outside[0][0]}-{outside[-1][1]})" if outside else ""))
        for r in leaf:
            nsym += 1
            callers = db.execute("SELECT f2.path, e.line, f2.role FROM edges e JOIN files f2 ON f2.id=e.src_file WHERE e.dst_sym=? AND e.src_file!=? ORDER BY f2.path, e.line", (r["id"], f["id"])).fetchall()
            tests = sorted({c["path"] for c in callers if c["role"] == "test"} | {x[0] for x in db.execute("SELECT DISTINCT f2.path FROM edges e JOIN files f2 ON f2.id=e.src_file WHERE e.dst_file=? AND f2.role='test'", (f["id"],))})
            cl = [c for c in callers if c["role"] != "test"]
            lines.append(f"  {r['name']} {r['lo']}-{r['hi']}  callers: {len(cl)}" + (" (" + ", ".join(f"{c['path']}:{c['line']}" for c in cl[:3]) + (", ..." if len(cl) > 3 else "") + ")" if cl else "") + (f"  tests: {', '.join(tests[:3])}" + (f" +{len(tests) - 3}" if len(tests) > 3 else "") if tests else "  tests: none found"))
    for p in new: lines.append(f"{p}  (new, untracked)")
    if nnew > len(new): lines.append(f"... +{nnew - len(new)} more untracked files")
    print(f"changed vs {base}: {len(ch)} files, {nsym} symbols, +{tot_a} -{tot_d}")
    lim = a.limit
    print("\n".join(lines[:lim]) + (f"\n... +{len(lines) - lim} more lines (--limit N)" if len(lines) > lim else ""))
# ---- q: one entry point, routed by the shape of the argument ----
def route(db, root, arg):
    """-> (kind, detail): 'show' path:line | 'map' path | 'sym' name | 'search' text."""
    arg = arg.strip()
    m = re.match(r"^(.+?):(\d+)$", arg)
    if m and " " not in arg and not arg.count("::"): return ("show", arg)
    if " " not in arg and (("/" in arg or "." in arg or arg in (".", "..")) or os.path.exists(arg)):
        t = map_target(db, root, arg)
        if t: return ("map", arg)
    if " " not in arg:
        nm = name_of(arg)
        if (IDENT.match(nm) or re.match(r"^[A-Za-z_$][\w$]*$", nm)) and db.execute("SELECT 1 FROM symbols WHERE name=? AND kind!='route' LIMIT 1", (nm,)).fetchone(): return ("sym", arg)
        t = map_target(db, root, arg)
        if t: return ("map", arg)
    return ("search", arg)
def cmd_q(a):
    root = graph_root(None); db = gopen(root, True); arg = " ".join(a.text); kind, _ = route(db, root, arg)
    print(f"# q -> {kind}")
    if kind == "show": return cmd_show(type("A", (), dict(target=arg.strip(), max=60)))
    if kind == "map": return cmd_map(type("A", (), dict(target=arg.strip(), budget=1500)))
    if kind == "sym":
        nm = name_of(arg); fs, ss, _ = resolve(db, arg); fresh = freshen(root, db, [r["path"] for r in ss]) if ss else db
        defs, hits, n = find_refs(root, fresh, nm, 60); print(fmt_refs(nm, defs, hits, n, 8, 3))
        if ss:
            cache = {}
            for r in ss[:3]: print(f"{r['path']}:{r['lo']}  {sigtext(cache.setdefault(r['path'], disk_lines(root, r['path'])), r['lo'])}")
        return
    return cmd_search(type("A", (), dict(text=[arg], k=8)))
