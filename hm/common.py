"""hintmap.common: see README Module ownership."""
import array, datetime, contextlib, functools, hashlib, heapq, itertools, json, math, os, re, signal, sqlite3, struct, sys, time

class _Lazy:  # import on first attribute use (hook start-up: subprocess+shutil+tempfile+glob+argparse cost ~7 ms CPU and the hook needs none of them)
    def __init__(s, n): s.n = n
    def __getattr__(s, k): m = __import__(s.n); globals()[s.n] = m; return getattr(m, k)
argparse, glob, io, shutil, subprocess, tempfile = map(_Lazy, "argparse glob io shutil subprocess tempfile".split())
SELF = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "hintmap")  # launcher stub; children re-run it (hintmap.py is imported so its bytecode is cached)
def sh(cmd, cwd=None):
    try: r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except OSError: return ""  # tool not installed (git is optional everywhere)
    return r.stdout.strip() if r.returncode == 0 else ""
def now(): return time.strftime("%Y-%m-%dT%H:%M:%S")
_T = [time.perf_counter(), time.process_time()]
def mark(log, name):
    """hook stage timer: attributes wall/CPU ms since the previous mark to `name` (summed over repeats) -> log["t"][name] = [wall, cpu]."""
    w, c = time.perf_counter(), time.process_time(); t = log.setdefault("t", {}); o = t.get(name, (0, 0)); t[name] = [round(o[0] + (w - _T[0]) * 1000, 1), round(o[1] + (c - _T[1]) * 1000, 1)]; _T[:] = w, c
# ---- fingerprints (no git needed: file content hash + graph.db symbol span) ----
def h(b): return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()[:16]
# ---- hooks (fail-open; read-only DB) ----
STOP = set("the and for with that this from into when then also what not are was but all can add fix use new".split())
HOME = lambda: os.environ.get("HINTMAP_HOME") or os.path.expanduser("~/.hintmap")
RG = lambda: shutil.which("rg")
HIST = lambda: os.environ.get("HM_HIST") != "0"  # commit-history channel (default on; HM_HIST=0 turns it off)
CODEISH = r"\w+_\w+|\b[a-z]+[A-Z]\w*|`[^`]+`|\w+\.\w{1,4}\b"
COMMON = set("handle render update create delete value items index start close build parse format apply reset default props state error while switch catch return print super await".split())
NOISE = ["-g", "!node_modules", "-g", "!.venv", "-g", "!dist", "-g", "!*{test,spec}*", "-g", "!**/{tests,__tests__}/**"]
def subqueries(q):
    """Whole brief (<= 4000 chars) + up to 4 bullet/sentence chunks richest in identifiers/file names. Short briefs: one query."""
    if len(q) <= 300: return [q]
    chunks = [c.strip() for c in re.split(r"\n\s*(?:[-*•]|\d+[.)])\s+|(?<=[.!?])\s+|\n\n+", "\n" + q) if len(c.strip()) >= 15]
    best = sorted(chunks, key=lambda c: -len(re.findall(CODEISH, c)))[:4]
    return list(dict.fromkeys([q] + [c[:300] for c in best] + ([q[:300]] if len(best) < 2 else [])))  # + the whole long text: retrieve() extracts mentions / salient terms from it (HM_LONGQ)  # the whole-brief head (usually generic context) only when bullets are scarce
def rrf(lists, k=60):
    """lists of hits (path, start, end, content) -> merged by path, best chunk first seen, Reciprocal Rank Fusion."""
    score, first = {}, {}
    for lst in lists:
        for rank, h in enumerate(lst):
            score[h[0]] = score.get(h[0], 0) + 1 / (k + rank + 1); first.setdefault(h[0], h)
    return [first[p] for p in sorted(score, key=lambda p: -score[p])]
ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}
LOWPRI = lambda: os.nice(10)  # preexec for every child the hook starts
HOLDOUT_PCT = int(os.environ.get("HINTMAP_HOLDOUT_PCT") or 0)  # percent of agents held out of injection for an A/B; 0 = always inject
# ---- own retriever, all inside <root>/.hintmap/graph.db: contentless FTS5 (bm25) + static-embedding channel (int8 vectors); the SAME normalizer at index and query time ----
FOLD = str.maketrans("ıİşŞğĞçÇöÖüÜ", "iIsSgGcCoOuU")
STOPW = {w.translate(FOLD).lower() for w in """the and for with that this from into when then also what not are was but all can has have had its their there here where which who how why you your our
    ve bir bu şu için ile de da mi mı mu mü ne gibi çok daha olarak olan veya ama ise var yok her hem ki ben sen biz siz onlar şey kadar sonra önce
    self cls this none null true false return def const let var async await import new
    add update implement implementation following several returns ensure make sure should also use used using need needs""".split()}
PARTS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
def ntok(text):
    """-> (parts, wholes): camelCase/snake_case/path pieces, and the joined form of every multi-part identifier (so processingStore == processing_store == `processing store`)."""
    ps, ws = [], []
    for w in re.findall(r"[A-Za-z0-9_]+", text.translate(FOLD)):
        p = [x.lower() for x in PARTS.findall(w)]
        if len(p) > 1: ws.append("".join(p))
        ps += p
    keep = lambda l: [t for t in l if len(t) > 1 and t not in STOPW and not t.isdigit()]
    return keep(ps), keep(ws)
def norm(text): return [t for l in ntok(text) for t in l]
def fts_match(toks):
    return " OR ".join(f'"{t}"*' if len(t) >= 5 else f'"{t}"' for t in dict.fromkeys(toks))  # prefix `*` only for >=5-char tokens
POTION = lambda: os.path.join(HOME(), "potion")  # potion.db (WordPiece vocab, meta) + potion.i8 (row per token id: f32 scale + dim int8); built once by `graph model`
VD = int(os.environ.get("HINTMAP_VD") or 256)  # stored/queried vector dims (PCA-ordered prefix of potion 256); accuracy over disk: 56 cost -0.004 dev NDCG
VK = 200  # int8 = round(unit_component * VK), clipped to +-127
_P = {}
def potion_open():
    """(vocab db, fd of the int8 rows, dim) or None; opened once per process and path (a failure is not cached: `graph model` may create it)."""
    p = POTION()
    if p not in _P:
        try:
            db = sqlite3.connect(f"file:{p}.db?mode=ro", uri=True)
            _P[p] = db, os.open(p + ".i8", os.O_RDONLY), int(db.execute("SELECT v FROM meta WHERE k='dim'").fetchone()[0])
        except (sqlite3.Error, OSError, TypeError): return None
    return _P[p]
@functools.lru_cache(maxsize=None)
def wp_ids(P, w):
    """BERT WordPiece, greedy longest match, on one lowercase ASCII word -> token ids ([UNK]=1 when a piece is missing)."""
    db, ids, i = P[0], [], 0
    if len(w) > 100: return (1,)
    while i < len(w):
        e = len(w)
        while e > i and not (r := db.execute("SELECT id FROM vocab WHERE tok=?", (("##" if i else "") + w[i:e],)).fetchone()): e -= 1
        if e == i: return (1,)
        ids.append(r[0]); i = e
    return tuple(ids)
def potion_vec(P, words):
    """mean of the token rows (what model2vec's StaticModel does), unit-normalised, as int8-scaled ints for math.sumprod; None when nothing embeddable. Pure stdlib; repeated tokens are summed once with their count."""
    _, fd, d = P; cnt = {}
    for w in words:
        for i in wp_ids(P, w): cnt[i] = cnt.get(i, 0) + 1
    acc = [0.0] * d
    for i, c in cnt.items():
        row = os.pread(fd, d + 4, i * (d + 4)); s = struct.unpack("<f", row[:4])[0] * c
        acc = [a + s * x for a, x in zip(acc, memoryview(row)[4:].cast("b"))]
    acc = acc[:VD]; nrm = sum(a * a for a in acc) ** .5
    return [max(-127, min(127, round(a / nrm * VK))) for a in acc] if cnt and nrm else None
SUFX = ("ations", "ation", "ings", "ing", "ers", "er", "ed", "es", "s")
@functools.lru_cache(maxsize=None)
def sfx(w):
    for x in SUFX:
        if w.endswith(x) and len(w) - len(x) >= 3: return w[:-len(x)]
    return w
VEC_EXACT_MAX, VEC_PRE, VEC_KEEP, CODE_KEEP = 8000, 64, 1000, 2000  # above VEC_EXACT_MAX chunks the vector scan is 2-stage: Hamming top-VEC_KEEP over centered sign codes (.hintmap/vec.codes), exact int8 rescoring; no codes file -> the 64-dim prefilter scan
T01 = bytes.maketrans(b"\x00\x01", b"01")
def vcode(v, mean): return int(bytes(map(int.__gt__, v, mean)).translate(T01), 2)  # sign of (component - corpus mean): one bit per dim
def codes_path(root): return os.path.join(root, ".hintmap", "vec.codes")
def abs_paths(prompt):
    """(start, existing absolute path) for paths in the prompt (paths may contain spaces: shorten until it exists)."""
    for m in list(re.finditer(r"(?<![\w.])/[\w.@ /-]+", prompt))[:20]:
        c = m.group().rstrip()
        while len(c) > 1 and not os.path.exists(c): c = c[:max(c.rfind(" "), c.rfind("/"))]
        if len(c) > 1: yield m.start(), c
def query_of(prompt):
    """Prompt with absolute paths replaced by their extension-less basename, first 4000 chars."""
    for i, c in reversed(list(abs_paths(prompt))): prompt = prompt[:i] + os.path.splitext(os.path.basename(c))[0] + prompt[i + len(c):]
    return prompt[:4000]
def gpath(root): return os.path.join(root, ".hintmap", "graph.db")
MARKERS = ("pyproject.toml", "package.json", "setup.py", "Cargo.toml", "go.mod", "Package.swift")
def graph_root(path, init=False):
    """No git. init: the given path, else cwd's nearest marker dir. Others: nearest ancestor of cwd/path holding .hintmap/graph.db."""
    d = os.path.realpath(path or os.getcwd()); stop = {"/", os.path.expanduser("~"), os.path.expanduser("~/Desktop")}
    if init and path and d not in stop: return d
    while d not in stop and d != os.path.dirname(d):
        if os.path.exists(gpath(d)) if not init else any(os.path.exists(os.path.join(d, m)) for m in MARKERS): return d
        d = os.path.dirname(d)
    sys.exit("hintmap: no project marker found here; pass a path" if init else "hintmap: no graph here; run `hintmap graph init`")
def gopen(root, ro=False):
    p = gpath(root)
    if not os.path.exists(p): sys.exit(f"hintmap: no graph for {root}; run `hintmap graph init`")
    db = sqlite3.connect(f"file:{p}?mode=ro", uri=True) if ro else sqlite3.connect(p, timeout=30)
    db.row_factory = sqlite3.Row
    return db
SKIPDIR = ("node_modules", ".venv", "venv", "dist", "build", ".hintmap", "__pycache__", "DerivedData", "Pods", "Carthage", "zig-pkg", "zig-out")  # build outputs / dependency checkouts (+ *.xcframework, see scope())
