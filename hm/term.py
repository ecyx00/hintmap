"""hintmap.term: see README Module ownership. Term search: the words the caller supplies, matched exactly (sub-token, light stem) against symbol names, code lines and paths. No model, no embeddings."""
import math, re
from hm.common import PARTS, STOPW, FOLD
from hm.jit import defline_rx, disk_lines, stale

NLSTOP = set("""where is are does do did how what which who why when the a an of to in on at by as it be or if so we i me my find show look looking locate located defined define definition code file files function functions method class used uses using work works working happen happens get gets set sets into out via than then them they there here about any some each other new old
""".split())
SUF = ("ations", "ation", "ments", "ment", "ings", "ing", "ers", "er", "edly", "ed", "ies", "es", "ly", "s")
MAXT = 10  # terms kept from a long text (highest idf first)
W_SYM, W_LINE, W_PATH = 1.5, 1.0, 0.5

def stem(w):
    for s in SUF:
        if w.endswith(s) and len(w) - len(s) >= 3: return w[:-len(s)] + ("y" if s == "ies" else "")
    return w
def subs(text):
    """identifier / path text -> lowercase sub-token stems (snake_case, camelCase, digits split)."""
    return [stem(p.lower()) for w in re.findall(r"[A-Za-z0-9]+", text.translate(FOLD)) for p in PARTS.findall(w)]
def hit(ts, t):  # does sub-token stem ts match term stem t
    return ts == t or (len(t) >= 5 and ts.startswith(t))
def matched(text, terms):
    ss = subs(text)
    return [t for t in terms if any(hit(s, t) for s in ss)]
def df_of(db, t):
    return db.execute("SELECT COALESCE(SUM(doc),0) FROM ixv WHERE term>=? AND term<?", (t, t + "\U0010ffff")).fetchone()[0]
def get_terms(db, text):
    """-> {stem: idf}: NL stopwords dropped, terms absent from the index dropped, at most MAXT by idf."""
    n = db.execute("SELECT MAX(id) FROM chunks").fetchone()[0] or 1; out = {}
    for s in subs(text):
        if s in out or len(s) < 3 or s in STOPW or s in NLSTOP or s.isdigit(): continue
        d = df_of(db, s)
        if d: out[s] = math.log(1 + n / d)
    return dict(sorted(out.items(), key=lambda kv: -kv[1])[:MAXT])
def term_search(root, db, text, limit=40):
    """-> list of (path, line|None, text, score, kind) best first, at most 2 rows per file."""
    terms = get_terms(db, text)
    if not terms: return []
    tl = list(terms); need = min(2, len(tl)); rows = {}  # (path, line) -> [score, text, kind]
    role = {}
    def add(p, ln, tx, sc, kind):
        if role.get(p) == "test": sc *= 0.7
        k = (p, ln); o = rows.get(k)
        if o: o[0] += sc; o[2] = o[2] if o[2] == "sym" else kind
        else: rows[k] = [sc, tx, kind]
    # symbols: name contains a term (sub-token); live definition line
    seen = set(); cache = {}
    for t in tl:
        for r in db.execute("SELECT s.name, s.lo, f.path, f.role, f.size, f.mtime_ns FROM symbols s JOIN files f ON f.id=s.file_id WHERE s.kind!='route' AND s.name LIKE ? LIMIT 5000", (f"%{t}%",)):
            key = (r["path"], r["name"], r["lo"])
            if key in seen: continue
            seen.add(key); ms = matched(r["name"], tl)
            if not ms: continue
            role[r["path"]] = r["role"]; ln = r["lo"]
            if stale(root, r["path"], r["size"], r["mtime_ns"]):
                ls = cache.setdefault(r["path"], disk_lines(root, r["path"])); rx = defline_rx(r["name"]); ln = next((i for i, l in enumerate(ls, 1) if rx.search(l)), None)
                if ln is None: continue
            add(r["path"], ln, None, W_SYM * sum(terms[m] for m in ms), "sym")
    # lines: FTS picks candidate files, the live files give the lines
    q = " OR ".join(f'"{t}"*' for t in tl); files = {}
    for r in db.execute("SELECT f.path, f.role, f.size, f.mtime_ns FROM ix JOIN chunks c ON c.id=ix.rowid JOIN files f ON f.id=c.file_id WHERE ix MATCH ? ORDER BY bm25(ix) LIMIT 300", (q,)):
        files.setdefault(r["path"], r); role[r["path"]] = r["role"]
        if len(files) >= 30: break
    for p in files:
        ls = cache.setdefault(p, disk_lines(root, p))
        for i, l in enumerate(ls, 1):
            low = l.lower()
            if not any(t in low for t in tl): continue
            ms = matched(l, tl)
            if len(ms) >= need: add(p, i, l.strip(), W_LINE * sum(terms[m] for m in ms), "line")
    # paths
    seenp = set()
    for t in tl:
        for r in db.execute("SELECT path, role FROM files WHERE path LIKE ? LIMIT 2000", (f"%{t}%",)):
            if r["path"] in seenp: continue
            seenp.add(r["path"]); ms = matched(r["path"], tl)
            if ms: role[r["path"]] = r["role"]; add(r["path"], None, None, W_PATH * sum(terms[m] for m in ms), "path")
    out, per = [], {}
    for (p, ln), (sc, tx, kind) in sorted(rows.items(), key=lambda kv: (-kv[1][0], kv[0][0], kv[0][1] or 0)):
        if per.get(p, 0) >= 2: continue
        if kind == "path" and any(k[0] == p and k[1] for k in rows) and per.get(p): continue
        per[p] = per.get(p, 0) + 1; out.append((p, ln, tx, sc, kind))
        if len(out) >= limit: break
    return out
