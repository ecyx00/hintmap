"""hintmap.retrieve: see README Module ownership."""
import array, datetime, contextlib, functools, hashlib, heapq, itertools, json, math, os, re, signal, sqlite3, struct, sys, time
from hm.common import HIST, CODEISH, CODE_KEEP, COMMON, STOPW, VEC_EXACT_MAX, VEC_KEEP, VEC_PRE, VK, codes_path, fts_match, gopen, mark, norm, ntok, potion_open, potion_vec, sfx, vcode

TESTQ = re.compile(r"\b(tests?|pytest|specs?|unittest|jest|vitest)\b", re.I)
ROLE_MULT = {"src": 1.0, "test": 0.3, "gen": 0.3}  # design 1.4(5): test/legacy/vendored/generated penalties (test penalty off when the query asks for tests)
NOISEW = {"landing", "docs", "doc", "examples", "example", "demo", "demos", "marketing", "website", "storybook", "stories", "benchmarks", "benchmark", "scripts", "tools", "experiments"}  # promo/doc-like dir words
def noisy(p, lang, qset):
    """True for landing/docs/examples-like dirs and file-level-only languages (no grammar = no symbols), unless the query names the dir word or the language."""
    dw = {t for d in p.split("/")[:-1] for t in ntok(d)[0]} & NOISEW
    return bool(dw - qset) or bool(lang in DOCLANG and lang not in qset and os.path.splitext(p)[1][1:] not in qset)
NOISE_MULT = 0.4
DATAFILE, DATA_MULT = {"json", "yaml", "toml", "env"}, 0.5
DATAQ = {"config", "configuration", "configure", "configured", "data", "json", "yaml", "yml", "toml", "env", "settings", "setting", "fixture", "fixtures", "schema", "schemas", "manifest", "spec", "specs", "dataset", "locale", "locales", "translation", "translations", "environment", "variable", "variables", "option", "options"}
DOCLANG = {"markdown", "html", "css", "xml"}  # prose/markup files rank below code unless the query names them (config/data files json/yaml/toml/env: no down-weight)
GENERIC_STEM = set("service services store api controller screen screens utils util index client manager helper helpers main app base types type".split())
GENERIC_FILE = {"mod", "index", "__init__", "main", "init", "lib", "default", "package"}  # file stems that name nothing: the parent dir is the effective name (mod.rs, index.js, __init__.py, main.go)
NAME_W, DIR_W, NAME_MIN = 0.02, 0.01, 0.5  # RRF-score units: one rank-1 channel hit is 1/61 = 0.016; NAME_MIN = share of name parts a file needs to join the pool by name alone
def tmatch(a, b, ex=False):
    """does a query token name a path part: equal; unless ex, also equal after suffix stripping (merged ~ merge) or one a prefix of the other (>= 4 chars). ponytail: all-or-nothing, graded 0.8/0.4 scored worse on dev."""
    if a == b: return 1.0
    if ex: return 0.0
    a, b = sfx(a), sfx(b)
    return 1.0 if a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a))) else 0.0
@functools.lru_cache(maxsize=20000)
def pathparts(p):
    """(effective-name parts, parent-dir parts): generic file stems (mod/index/__init__/main...) take the parent dir as their name; generic words (service, utils...) drop out unless nothing else is left."""
    d, f = os.path.split(p); k = os.path.splitext(f)[0]; dn = ntok(os.path.basename(d))[0]
    nm = ntok(k)[0] if k.lower() not in GENERIC_FILE else dn
    nm = [t for t in nm if t not in GENERIC_STEM] or nm
    return tuple(nm), tuple(dn if k.lower() not in GENERIC_FILE else ntok(os.path.basename(os.path.dirname(d)))[0])
def path_score(p, qt, ex=False):
    """0..1: mean over the file's effective-name parts of the best query-token match, plus a small share for the parent dir's parts; halved for generic file names (the definition is usually elsewhere)."""
    nm, dn = pathparts(p)
    if not nm: return 0.0
    a = sum(max(tmatch(t, x, ex) for t in qt) for x in nm) / len(nm)
    if not a: return 0.0
    k = os.path.splitext(os.path.basename(p))[0].lower()
    return (a + (DIR_W / NAME_W) * (sum(max(tmatch(t, x, ex) for t in qt) for x in dn) / len(dn) if dn else 0)) * (0.5 if k in GENERIC_FILE else 1)
LEX_TERMS = 12
BIG_CHUNKS, DF_CUT = 50000, 0.2  # above BIG_CHUNKS chunks, query terms present in > DF_CUT of them are dropped (always keeping the rarest one)
_CODES = {}
def load_codes(root):
    """(chunk ids, code ints, mean) or None. ~15 ms for 234k chunks (int.from_bytes per code); cached per process."""
    if root not in _CODES:
        try:
            raw = memoryview(open(codes_path(root), "rb").read()); n, nb = struct.unpack_from("<II", raw, 4); ids = array.array("I"); ids.frombytes(raw[12:12 + 4 * n]); o = 12 + 4 * n
            _CODES[root] = ids, list(map(int.from_bytes, itertools.chain.from_iterable(struct.iter_unpack(f"{nb}s", raw[o:o + n * nb]))))  # C-level pipeline: 2x faster than a slicing loop
        except (OSError, struct.error, ValueError): _CODES[root] = None
    return _CODES[root]
def nearest(d, k):
    """indices of the k smallest Hamming distances (ties: lowest index first, like heapq.nsmallest, which was 26 ms over 234k): binary-search the cut distance t with translate+count, list the indices with find; all at C speed."""
    b = bytes(map((255).__and__, d))  # ponytail: a distance of exactly 256 wraps to 0 (every bit flipped: never a near neighbour)
    def mark(flag): return b.translate(bytes(int(flag(i)) for i in range(256)))
    lo, hi = 0, 255
    while lo < hi:  # smallest t with #(d <= t) >= k
        mid = (lo + hi) // 2
        if mark(mid.__ge__).count(1) >= k: hi = mid
        else: lo = mid + 1
    def at(m):
        i = m.find(1)
        while i >= 0: yield i; i = m.find(1, i + 1)
    lt = mark(lo.__gt__); return [*at(lt), *itertools.islice(at(mark(lo.__eq__)), k - lt.count(1))]
NN_W, NN_K, NN_TERMS = float(os.environ.get("HM_NN_W", 0.25)), 3, 12  # nn channel weight (0 = off), neighbours per term, term cap
MEN_W = float(os.environ.get("HM_MEN_W", 1.5))  # mention channel weight (long queries, L1)
WT_SYM, WT_NL = {"lex": 1.0, "vec": 0.6, "sym": 1.2, "nn": 0.0, "men": MEN_W}, {"lex": 1.0, "vec": 1.0, "sym": 1.2, "nn": NN_W, "men": MEN_W}  # channel weights in the per-sub-query file RRF (NL lex 0.3 -> 1.0 after the detail=full bm25 fix, see notes)
FIN_ROLE = float(os.environ.get("HM_FIN_ROLE", 0.5))
HOW_W = float(os.environ.get("HM_HOW_W", 0.004))
I18NQ = {"locale", "locales", "i18n", "l10n", "translation", "translations", "translate", "translatable", "translated", "localization", "localisation", "localize", "localized", "localised", "language", "languages"}
I18NP = {"locales", "locale", "i18n", "l10n", "translations", "translation", "lang", "languages", "intl", "messages"}  # dir / file-stem words of locale data files
I18N_N, WT_I18N = 2, 2.0  # at most 2 locale files, fused like a third channel (rank 0 = 2x a top-1 hit of one channel)
GRAV_W, COH_W = float(os.environ.get("HM_GRAV", 0.01)), float(os.environ.get("HM_COH", 0.5))  # long queries only (islong): size gravity (log10 loc) and file coherence (RRF mass of the file's other chunks); 0 = off. Gated: ungated they gain on SWE and lose 0.017 NDCG on semble-bench
BM_W = (2.0, 3.0, 1.0)  # bm25 column weights: path, name (symbol + parent), body
def symbol_cands(db, q):
    """identifier-like query words -> chunks of symbols with that exact name (case-exact first). [(chunk id, score)]  # ponytail: sub-token stage (bm25 on the name column) scored worse on dev than exact-only; the lex channel already covers sub-tokens"""
    ws = [w for w in dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", q)) if w.lower() not in STOPW and w not in COMMON]
    ws = [w for w in ws if len(ws) == 1 or re.search(r"_|[a-z][A-Z]|^[A-Z][a-z]+[A-Z]|^[A-Z]{2,}[a-z]", w)][:6]
    if not ws: return []
    var = list(dict.fromkeys(v for w in ws for v in (w, w.lower(), w[0].upper() + w[1:], w.upper())))
    out = {}
    for cid, nm, par in db.execute(f"SELECT c.id, s.name, s.parent_id FROM symbols s JOIN chunks c ON c.sym_id=s.id WHERE s.name IN ({','.join('?' * len(var))}) ORDER BY c.id LIMIT 300", var):
        out[cid] = (1.0 if nm in ws else 0.9) * (1.0 if par is None else 0.8)
    return sorted(out.items(), key=lambda x: -x[1])
WT_HIST, HIST_K, HIST_TERMS, HIST_DECAY = float(os.environ.get("HM_WT_HIST", 0.5)), 30, 16, 1000
def hist_files(db, q):
    """channel H: bm25 of the query over commit messages (top HIST_K) -> [(path, score)] best first; a file scores sum(bm25 of the commits that touched it) / (1 + recency rank / HIST_DECAY). [] when the history tables are absent."""
    toks = list(dict.fromkeys(norm(q)))[:60]
    try:
        if len(toks) > HIST_TERMS:
            df = dict(db.execute(f"SELECT term, doc FROM histv WHERE term IN ({','.join('?' * len(toks))})", toks)); toks = sorted(toks, key=lambda t: df.get(t, 3))[:HIST_TERMS]
        if not toks: return []
        mx = db.execute("SELECT MAX(rowid) FROM hist").fetchone()[0] or 0; sc = {}
        for rid, b in db.execute("SELECT rowid, bm25(hist) FROM hist WHERE hist MATCH ? ORDER BY 2 LIMIT ?", (fts_match(toks), HIST_K)).fetchall():
            for (f,) in db.execute("SELECT path FROM hfiles WHERE rid=?", (rid,)): sc[f] = sc.get(f, 0) - b / (1 + (mx - rid) / HIST_DECAY)
        return sorted(sc.items(), key=lambda x: -x[1])
    except sqlite3.Error: return []
def retrieve(root, qs, log=None, ch=None):
    """Per sub-query ranked file hits [(path, lo, hi, 'kind name')]: FTS5 bm25 and cosine over int8 chunk vectors (ch: lex|vec|both), each reranked
    (role penalty, definition boost, per-file coherence + saturation), RRF-fused per sub-query. Read-only, stdlib only."""
    ch = ch or os.environ.get("HINTMAP_CHANNELS") or "both"; db = gopen(root, True); log = log if log is not None else {}
    if db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name IN ('ix', 'fx')").fetchone()[0] < 2: return [[] for _ in qs]  # no index yet / pre-schema-10 graph (rebuild pending)
    P = potion_open() if ch != "lex" and db.execute("SELECT v FROM meta WHERE k='vec_dim'").fetchone() not in (None, ("0",)) else None
    mark(log, "open"); lists = []; nmax = db.execute("SELECT MAX(id) FROM chunks").fetchone()[0] or 0
    sal = [salient(db, q, nmax) if islong(q) and "L2" in lq() else None for q in qs]  # long query, L2: top-K terms by idf replace the raw term lists (bm25, soft-lexical, name channel, vector)
    vq = [P and potion_vec(P, s_ if s_ is not None else ntok(q)[0][:80]) for q, s_ in zip(qs, sal)]
    big = nmax if nmax > BIG_CHUNKS else 0; codes = mean = vrows = None
    if P and any(vq) and nmax > VEC_EXACT_MAX and (codes := load_codes(root)):
        mean = json.loads(db.execute("SELECT v FROM meta WHERE k='vec_mean'").fetchone()[0])
    vrows = [] if codes else db.execute("SELECT id, file_id, v FROM chunks WHERE v IS NOT NULL").fetchall() if P and any(vq) else []
    log["chunks_vec"] = len(codes[0]) if codes else len(vrows); flat = None; mark(log, "vec-load")
    for q, v, s_ in zip(qs, vq, sal):
        toks = list(dict.fromkeys(norm(q)))[:60]; cands = {}; df = {}
        if s_ is not None: toks = s_
        elif len(toks) > LEX_TERMS or big:  # keep the most discriminative terms: lowest document frequency (unseen terms rank as rare; they may still prefix-match)
            df = dict(db.execute(f"SELECT term, doc FROM ixv WHERE term IN ({','.join('?' * len(toks))})", toks)); toks = sorted(toks, key=lambda t: df.get(t, 3))[:LEX_TERMS]
            if big: toks = [t for t in toks if df.get(t, 0) <= big * DF_CUT] or toks[:1]  # stop-word-like terms (in 22% of the chunks) cost 10-20 ms of bm25 each and carry little
        if ch != "vec" and toks:
            cands["lex"] = [(r[0], -r[1]) for r in db.execute("SELECT rowid, bm25(ix, ?, ?, ?) FROM ix WHERE ix MATCH ? ORDER BY 2 LIMIT 100", (*BM_W, fts_match(toks)))]
        if NN_W and ch != "vec" and P and toks and len(q.split()) > 1 and not (re.search(CODEISH, q) and len(q.split()) <= 5):  # soft lexical match: static-embedding neighbours of the NL query terms as a second, lighter bm25 channel (FTS5 OR has no per-term weight)
            try: nb = list(dict.fromkeys(w for t in toks for r in P[0].execute("SELECT nb FROM nn WHERE tok=?", (t,)) for w in r[0].split()[:NN_K] if w not in toks))[:NN_TERMS]
            except sqlite3.Error: nb = []
            if nb: cands["nn"] = [(r[0], -r[1]) for r in db.execute("SELECT rowid, bm25(ix, ?, ?, ?) FROM ix WHERE ix MATCH ? ORDER BY 2 LIMIT 100", (*BM_W, fts_match(nb)))]
        mark(log, "lex")
        if v:
            if codes:  # Hamming over the centered sign codes (C-speed map: xor + bit_count), best VEC_KEEP rescored exactly from the live int8 vectors
                qc = vcode(v, mean); rid = sorted({codes[0][k] for k in nearest(list(map(int.bit_count, map(qc.__xor__, codes[1]))), CODE_KEEP)} | {i for i, _ in cands.get("lex", ())})  # + the FTS candidates: their cosine is exact too
                sc = sorted(((math.sumprod(v, memoryview(b).cast("b")) / VK / VK, i) for i, b in db.execute(f"SELECT id, v FROM chunks WHERE id IN ({','.join('?' * len(rid))})", rid)), reverse=True)[:100]
            elif len(vrows) > VEC_EXACT_MAX:  # big index: 2-stage scan, first VEC_PRE dims of every chunk (potion rows are PCA-ordered) -> full dims for the best VEC_KEEP
                if flat is None: flat = (memoryview(b"".join(r[2] for r in vrows)).cast("b"), [r[0] for r in vrows])
                d = len(vrows[0][2]); mv, rid = flat; pre = v[:VEC_PRE]
                top = heapq.nlargest(VEC_KEEP, ((math.sumprod(pre, mv[o:o + VEC_PRE]), k) for k, o in zip(range(len(rid)), range(0, len(mv), d))))
                sc = sorted(((math.sumprod(v, mv[k * d:(k + 1) * d]) / VK / VK, rid[k]) for _, k in top), reverse=True)[:100]
            else: sc = sorted(((math.sumprod(v, memoryview(b).cast("b")) / VK / VK, i) for i, _, b in vrows), reverse=True)[:100]
            cands["vec"] = [(i, s) for s, i in sc if s > 0.2]
        mark(log, "vec")
        if ch != "vec" and (sc_ := symbol_cands(db, q)): cands["sym"] = sc_
        if islong(q) and "L1" in lq() and (mc := mention_cands(db, q)): cands["men"] = mc
        ids = {i for l in cands.values() for i, _ in l}
        meta = {r[0]: r[1:] for r in db.execute(f"SELECT c.id, f.path, f.role, c.lo, c.hi, s.name, s.kind, f.lang FROM chunks c JOIN files f ON f.id=c.file_id LEFT JOIN symbols s ON s.id=c.sym_id WHERE c.id IN ({','.join('?' * len(ids))})", list(ids))} if ids else {}
        qset, wantt, fused, cs = set(norm(q)), bool(TESTQ.search(q)) and not islong(q), {}, {}  # an issue-length text that says "test" is rarely asking for test files
        wt = WT_SYM if re.search(CODEISH, q) and len(q.split()) <= 5 else WT_NL  # symbol-like short query trusts lexical; natural language trusts vectors (measured: lexical alone matches generic words)
        for cn, l in cands.items():
            if not l: continue  # a channel with no candidates (every query token unseen / all below the cosine floor)
            adj = {}
            for i, s in l:
                p, rl, lo, hi, nm, kd, lg = meta[i]
                m = ROLE_MULT.get(rl, 1.0) if not (rl == "test" and wantt) else 1.0
                if noisy(p, lg, qset): m *= NOISE_MULT
                if lg in DATAFILE and not qset & DATAQ: m *= DATA_MULT  # command specs / fixtures / manifests rank below code unless the query is about data or config
                if os.path.basename(p) == "__init__.py": m *= 0.5
                if nm and (nt := ntok(nm)) and (nt[1] or nt[0] or [''])[-1] in qset: m *= 1.5  # definition boost: symbol table name == query token
                adj[i] = s * m
            byf = {}
            for i in sorted(adj, key=lambda i: -adj[i]): byf.setdefault(meta[i][0], []).append(i)
            tot = {f: sum(adj[i] for i in c) for f, c in byf.items()}; mx = max(tot.values()) or 1
            fin = {i: adj[i] * (1 + 0.2 * tot[f] / mx if k == 0 else 0.5 ** k) for f, c in byf.items() for k, i in enumerate(c[:3])}  # coherence on the best chunk, saturation on the rest
            seen = set()  # RRF over FILE ranks (a file's best chunk): chunk-level fusion let the channel with many chunks per file drown the other
            for j, i in enumerate(sorted(fin, key=lambda i: -fin[i])):
                cs[i] = cs.get(i, 0) + wt[cn] / (60 + j + 1)  # chunk-level score (RRF over chunk ranks, summed over channels): picks which chunk of a file to show; the file's rank is unchanged
                if meta[i][0] not in seen: seen.add(meta[i][0]); r = len(seen) - 1; fused[meta[i][0]] = fused.get(meta[i][0], (0, i)); fused[meta[i][0]] = (fused[meta[i][0]][0] + wt[cn] / (60 + r + 1), fused[meta[i][0]][1])
        if (qset | set(re.findall(r"[a-z0-9]+", q.lower()))) & I18NQ:  # (raw words too: ntok drops i18n / l10n)
            # locale/translation words in the query -> locale data files by path (bm25 length-normalises a path hit away on a 400-word key list); reference language first, then the biggest
            rows = db.execute("SELECT c.id, f.path, f.role, c.lo, c.hi, f.lang, f.loc FROM files f JOIN chunks c ON c.file_id=f.id WHERE f.lang IN ('json','yaml','toml') AND c.sym_id IS NULL").fetchall()
            pick = sorted((r for r in rows if I18NP & {t for d in r[1].split("/")[:-1] + [os.path.splitext(os.path.basename(r[1]))[0]] for t in ntok(d)[0] + [d.lower()]}), key=lambda r: (ntok(os.path.splitext(os.path.basename(r[1]))[0])[0] != ["en"], -r[6], r[1]))[:I18N_N]
            for rk, r in enumerate(pick): meta[r[0]] = (r[1], r[2], r[3], r[4], None, None, r[5]); cur = fused.get(r[1], (0, r[0])); fused[r[1]] = (cur[0] + WT_I18N / (60 + rk + 1), cur[1])
        qt = [t for t in dict.fromkeys(ntok(q)[0]) if len(t) >= 3 and (not big or t in toks or t not in df)] if s_ is None else [t for t in toks if len(t) >= 3]  # name channel: stem/prefix match of query tokens against each candidate's effective file name (+ parent dir); files matching by name alone join the pool
        if qt:
            for r in db.execute("SELECT c.id, f.path, f.role, c.lo, c.hi, f.lang FROM fx JOIN chunks c ON c.id=fx.rowid JOIN files f ON f.id=c.file_id WHERE fx MATCH ? ORDER BY bm25(fx) LIMIT 150", (" OR ".join(f'"{sfx(t)}"*' for t in qt),)):
                if r[1] not in fused and path_score(r[1], qt) >= NAME_MIN and not noisy(r[1], None, qset): meta[r[0]] = (r[1], r[2], r[3], r[4], None, None, r[5]); fused[r[1]] = (0.0, r[0])
        def adopt(paths):  # files with no candidate chunk yet: meta + the file's first chunk (existing files only)
            ps = [f for f in paths if f not in fused]
            for r in db.execute(f"SELECT MIN(c.id), f.path, f.role, c.lo, c.hi, f.lang FROM files f JOIN chunks c ON c.file_id=f.id WHERE f.path IN ({','.join('?' * len(ps))}) GROUP BY f.id", ps) if ps else ():
                meta[r[0]] = (r[1], r[2], r[3], r[4], None, None, r[5])
                yield r[1], r[0]
        if HIST() and (hl := hist_files(db, q)):
            known = {r[0] for r in db.execute(f"SELECT path FROM files WHERE path IN ({','.join('?' * len(hl))})", [f for f, _ in hl])}; role = dict(db.execute(f"SELECT path, role FROM files WHERE path IN ({','.join('?' * len(known))})", list(known))) if known else {}
            for f, i in adopt(known): fused[f] = (0.0, i)
            hl = sorted(((f, s * (ROLE_MULT.get(role[f], 1.0) if not wantt else 1.0)) for f, s in hl if f in role), key=lambda x: -x[1])
            for r, (f, _) in enumerate(hl): fused[f] = (fused[f][0] + WT_HIST / (60 + r + 1), fused[f][1])
        for f in fused:
            one = len(q.split()) == 1; ps = path_score(f, qt, one)  # a single-word query is a symbol lookup: only a file named exactly that gets the boost
            if ps and not (one and ps < 0.99) and not noisy(f, None, qset): fused[f] = (fused[f][0] + NAME_W * ps * (ROLE_MULT.get(meta[fused[f][1]][1], 1.0) if not wantt else 1.0), fused[f][1])
        if HOW_W and re.match(r"how\b", q.strip(), re.I) and fused:  # hub prior (big files are where "how does X work" lives), only for how-queries: ungated it gained a third as much
            for f, loc in db.execute(f"SELECT path, loc FROM files WHERE path IN ({','.join('?' * len(fused))})", list(fused)): fused[f] = (fused[f][0] + HOW_W * math.log10(1 + min(loc or 0, 3000)), fused[f][1])
        if (GRAV_W or COH_W) and fused and islong(q):  # long (issue / brief) queries: big files and files with several matching chunks are where the fix lives; short phrase queries measured the opposite
            if GRAV_W:
                for f, loc in db.execute(f"SELECT path, loc FROM files WHERE path IN ({','.join('?' * len(fused))})", list(fused)): fused[f] = (fused[f][0] + GRAV_W * math.log10(1 + min(loc or 0, 3000)), fused[f][1])
            if COH_W:
                tot, tp = {}, {}
                for i, v in cs.items(): f = meta[i][0]; tot[f] = tot.get(f, 0) + v; tp[f] = max(tp.get(f, 0), v)
                for f in fused: fused[f] = (fused[f][0] + COH_W * (tot.get(f, 0) - tp.get(f, 0)), fused[f][1])
        best = {}
        for i, v in cs.items():
            if meta[i][0] in fused and (meta[i][0] not in best or (v, -i) > (cs[best[meta[i][0]]], -best[meta[i][0]])): best[meta[i][0]] = i
        for f, i in best.items(): fused[f] = (fused[f][0], i)  # show the file's best-scoring chunk, not the first one a channel listed (chunk-level NDCG +0.001 dev; preferring inner symbols over containers / a 2nd chunk per file both scored lower)
        if FIN_ROLE != 1:  # role penalty again on the fused score: the per-channel penalty only reorders a channel's own list, and RRF then rewards a test file that tops several lists
            for f, (sc_, i) in fused.items():
                if meta[i][1] in ("test", "gen") and not (wantt and meta[i][1] == "test"): fused[f] = (sc_ * FIN_ROLE, i)
        lists.append([(f, *(lambda m: (m[2], m[3], f"{re.sub(r'^variable_declarator$', 'const', re.sub(r'_?(definition|declaration|item|specifier|def)$', '', m[5]))} {m[4]}".strip() if m[4] else ""))(meta[fused[f][1]])) for f in sorted(fused, key=lambda f: -fused[f][0])[:12]])
    return lists

# ---- long queries (HM_LONGQ = comma list, default "L1,L2": L1 mentions, L2 salient terms; "off" disables; L3 sections scored worse and was removed) ----
LONGQ_CHARS, SAL_K, MEN_MAX = 250, 24, 20
lq = lambda: os.environ.get("HM_LONGQ", "L1,L2").replace(",", " ").split()
islong = lambda q: len(q) > LONGQ_CHARS
def salient(db, q, n):
    """top SAL_K query terms by idf * (1 + log tf) in this repo's index; terms the index never saw are dropped (typos / prose), unlike the raw cap that ranks them rarest."""
    tf = {}
    for t in norm(q): tf[t] = tf.get(t, 0) + 1
    if not tf: return []
    df = dict(db.execute(f"SELECT term, doc FROM ixv WHERE term IN ({','.join('?' * len(tf))})", list(tf)))
    sc = {t: math.log((n + 1) / (df[t] + 0.5)) * (1 + math.log(tf[t])) for t in tf if df.get(t)}
    return sorted(sc, key=lambda t: -sc[t])[:SAL_K] or list(tf)[:LEX_TERMS]
TBF = re.compile(r'File "([^"]+)", line \d+, in (\w+)')
PATHM = re.compile(r"[\w@.-]+(?:/[\w@.-]+)*\.(?:py|pyi|js|jsx|ts|tsx|mjs|java|go|rs|rb|php|c|h|cc|cpp|hpp|cs|swift|kt|scala)\b")
DOTM = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
IDENT = re.compile(r"\b(?:[A-Za-z]+_\w+|[a-z]+[A-Z]\w*|[A-Z][a-z0-9]+[A-Z]\w*|\w{3,}(?=\())")
def mentions(q):
    """code-shaped mentions in priority order: [(kind, text, weight)] kind p=file path, s=symbol, d=dotted module path; traceback frames first, then paths, backticked spans, dotted names, identifiers."""
    out = {}
    def add(k, t, w):
        if t and (k, t) not in out and (k == "p" or (t.lower() not in STOPW and t not in COMMON and len(t) > 2)): out[(k, t)] = w
    for m in TBF.finditer(q): add("p", m.group(1), 1.0); add("s", m.group(2), 1.0)
    for m in PATHM.finditer(q): add("p", m.group(), 1.0)
    for sp in re.findall(r"`([^`\n]{2,80})`", q):
        for m in DOTM.finditer(sp): add("d", m.group(), 0.9)
        for m in re.finditer(r"[A-Za-z_]\w{2,}", sp): add("s", m.group(), 0.9)
    for m in DOTM.finditer(q):
        if not PATHM.fullmatch(m.group()): add("d", m.group(), 0.8)
    for m in IDENT.finditer(q): add("s", m.group(), 0.7)
    return [(k, t, w) for (k, t), w in out.items()][:MEN_MAX]
def mention_cands(db, q):
    """mentions resolved against the file table (suffix match of the path) and the symbol table (exact name; a name defined in many places scores lower, > 8 definitions: skipped). A traceback function is looked up inside the frame's file first. [(chunk id, score)]"""
    out = {}
    def put(cid, s): out[cid] = max(out.get(cid, 0), s)
    def file_chunk(fid): return (db.execute("SELECT MIN(id) FROM chunks WHERE file_id=?", (fid,)).fetchone() or (None,))[0]
    def by_path(t):
        ps = t.strip("./").split("/")
        for i in range(len(ps)):
            suf = "/".join(ps[i:]); rows = db.execute("SELECT id FROM files WHERE path=? OR substr(path, -?) = ?", (suf, len(suf) + 1, "/" + suf)).fetchall()
            if rows: return [r[0] for r in rows[:4]] if len(rows) <= 3 else []
        return []
    def by_sym(name, w, files=None):
        rows = db.execute("SELECT c.id, s.file_id FROM symbols s JOIN chunks c ON c.sym_id=s.id WHERE s.name=?", (name,)).fetchall()
        if files is not None: rows = [r for r in rows if r[1] in files]
        if 0 < len(rows) <= 8:
            for cid, _ in rows: put(cid, w / len(rows) ** 0.5)
    ms, frames = mentions(q), {}
    for m in TBF.finditer(q): frames.setdefault(m.group(1), set()).add(m.group(2))
    for k, t, w in ms:
        if k == "p":
            fs = by_path(t)
            for f in fs:
                if (c := file_chunk(f)) is not None: put(c, w / len(fs))
            for fn in frames.get(t, ()): by_sym(fn, 1.0, set(fs)) if fs else None
        elif k == "d":
            ps = t.split(".")
            if len(ps) > 1 and (r := db.execute("SELECT id FROM files WHERE instr(path, ?) > 0 LIMIT 4", ("/".join(ps) + ".",)).fetchall()) and len(r) <= 3:
                for (f,) in r:
                    if (c := file_chunk(f)) is not None: put(c, w / len(r))
            for x in ps[-2:]: by_sym(x, w)
        else:
            by_sym(t, w)
    return sorted(out.items(), key=lambda x: -x[1])[:100]
