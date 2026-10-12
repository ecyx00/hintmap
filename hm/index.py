"""hintmap.index: see README Module ownership."""
import array, collections, datetime, contextlib, functools, hashlib, heapq, itertools, json, math, os, re, signal, sqlite3, struct, sys, time
from hm.common import HIST, COMMON, HOME, ONE_THREAD, POTION, RG, SELF, SKIPDIR, VD, VEC_EXACT_MAX, VK, codes_path, glob, shutil, gopen, gpath, graph_root, h, ntok, potion_open, potion_vec, sfx, subprocess, vcode, wp_ids

def write_codes(db, root):
    """chunks.c (sign code per vector, ceil(dim/8) bytes, centered on meta vec_mean = mean of a ~4k sample, fixed until the next full rebuild) + the flat file the hook scans: b"HMC1", n, n x uint32 chunk ids, n x code. Only written above VEC_EXACT_MAX chunks. A stale file only costs recall (rescoring reads the live vectors)."""
    if (db.execute("SELECT MAX(id) FROM chunks").fetchone()[0] or 0) <= VEC_EXACT_MAX:
        with contextlib.suppress(OSError): os.unlink(codes_path(root))
        return
    if "c" not in {r[1] for r in db.execute("PRAGMA table_info(chunks)")}: db.execute("ALTER TABLE chunks ADD COLUMN c BLOB")
    m = db.execute("SELECT v FROM meta WHERE k='vec_mean'").fetchone()
    if m is None:
        smp = [memoryview(b).cast("b") for (b,) in db.execute("SELECT v FROM chunks WHERE v IS NOT NULL AND id % 37 = 0 LIMIT 4000")]
        mean = [round(sum(col) / len(smp)) for col in zip(*smp)]; db.execute("INSERT OR REPLACE INTO meta VALUES('vec_mean',?)", (json.dumps(mean),)); db.execute("UPDATE chunks SET c=NULL")
    else: mean = json.loads(m[0])
    nb = (len(mean) + 7) // 8
    while rows := db.execute("SELECT id, v FROM chunks WHERE c IS NULL AND v IS NOT NULL LIMIT 20000").fetchall():  # batches: all 234k x 256 B vectors at once were ~150 MB
        db.executemany("UPDATE chunks SET c=? WHERE id=?", [(vcode(memoryview(b).cast("b"), mean).to_bytes(nb, "big"), i) for i, b in rows])
    n = db.execute("SELECT COUNT(c) FROM chunks").fetchone()[0]
    with open(codes_path(root) + ".tmp", "wb") as f:
        f.write(b"HMC1" + struct.pack("<II", n, nb)); array.array("I", (r[0] for r in db.execute("SELECT id FROM chunks WHERE c IS NOT NULL ORDER BY id"))).tofile(f)
        for (c,) in db.execute("SELECT c FROM chunks WHERE c IS NOT NULL ORDER BY id"): f.write(c)
    os.replace(codes_path(root) + ".tmp", codes_path(root))
# ---- code graph: <root>/.hintmap/graph.db (derived, deletable); tree-sitter/numpy only inside `graph build|model`, run under ~/.hintmap/venv ----
GSCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY, path TEXT UNIQUE, lang TEXT, size INT, mtime_ns INT, sha TEXT, role TEXT, loc INT);
CREATE TABLE IF NOT EXISTS symbols(id INTEGER PRIMARY KEY, file_id INT, name TEXT, kind TEXT, lo INT, hi INT, parent_id INT, ar INT, sg INT);  -- ar/sg: parameter count + hash of the signature text (edit-check; no source stored)
CREATE TABLE IF NOT EXISTS refs(file_id INT, sym_id INT, name TEXT, line INT, n INT, u INT);
CREATE TABLE IF NOT EXISTS imports(file_id INT, spec TEXT, line INT);
CREATE INDEX IF NOT EXISTS sym_name ON symbols(name); CREATE INDEX IF NOT EXISTS sym_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS imports_file ON imports(file_id);
-- derived, rebuilt in one transaction at the end of every changed build
CREATE TABLE IF NOT EXISTS imp(file_id INT, dst INT, line INT, PRIMARY KEY(file_id, dst));
CREATE TABLE IF NOT EXISTS edges(id INTEGER PRIMARY KEY, src_file INT, src_sym INT, dst_file INT, dst_sym INT, kind TEXT, tier INT, w REAL, line INT, n INT);
CREATE TABLE IF NOT EXISTS nodes(file_id INTEGER PRIMARY KEY, indeg INT, hub INT);
-- retriever: one chunk per symbol span + one file-level row (path tokens, top-level words); FTS5 is contentless (no source text), v = int8 static-embedding vector or NULL
-- signature baseline for `edit-check`: first-seen signature of a symbol whose def changed/vanished in a re-index (the hook re-indexes right after a subagent), consumed by edit-check
CREATE TABLE IF NOT EXISTS sigchg(path TEXT, name TEXT, old_ar INT, old_sg INT, new_ar INT, new_sg INT, ts REAL, PRIMARY KEY(path, name));
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, file_id INT, sym_id INT, lo INT, hi INT, v BLOB);
CREATE INDEX IF NOT EXISTS chunk_file ON chunks(file_id); CREATE INDEX IF NOT EXISTS chunk_sym ON chunks(sym_id);  -- chunk_sym: symbol_cands joins symbols->chunks (a scan of 234k chunks = 30 ms without it)
CREATE VIRTUAL TABLE IF NOT EXISTS ix USING fts5(path, name, body, content='', contentless_delete=1, detail=full, tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE IF NOT EXISTS ixv USING fts5vocab(ix, 'row');
CREATE VIRTUAL TABLE IF NOT EXISTS fx USING fts5(path, content='', contentless_delete=1, detail=full, tokenize='unicode61 remove_diacritics 2');  -- file-level rows only (same rowid as their ix row): the name channel's prefix match over 26k docs instead of all 234k chunks (29 ms -> 1)
"""
DEFT = re.compile(r"(function|method|class|interface|struct|enum|trait|protocol|type_alias|module|object|constructor)_?(definition|declaration|item|specifier)$"
                  r"|^(function|method)_signature$|^(function|method|class)_def|^variable_declarator$|^type_spec$|^(method|singleton_method|class|module)$|^(function|data_type|newtype|type_synomym|type_synonym|instance)$")
IDT = {"identifier", "property_identifier", "type_identifier", "field_identifier", "simple_identifier", "constant", "name", "shorthand_property_identifier"}
IMPT = re.compile(r"import|^use_declaration$|^preproc_include$|^using_directive$")
NOCODE = {"markdown", "json", "yaml", "toml", "html", "css", "scss", "xml", "asciidoc", "text", "ini", "csv"}
TESTP = re.compile(r"(^|/)(tests?|__tests__|specs?|testing|testdata|fixtures?|__fixtures__|__mocks__|e2e)/|(^|/)test_([^/]*\.(py|cpp|cc|c|dart|lua)|helpers?[^/]*)$|[_.-](tests?|spec)\.\w+$|(^|/)[^/]*(Tests?|Spec)\.(java|kt|scala|swift|cs|php)$|(^|/)(conftest|gtest|gmock)[^/]*$")  # test files and fixture dirs across languages (cf. semble ranking/penalties.py)
GENP = re.compile(r"\.d\.ts$|\.min\.js$|(^|/)(generated|vendor|third_party|legacy|compat|examples?|deprecated)/|\.pb\.go$|_pb2\.py$")
IMPKW = re.compile(r"\b(from|import|use|as|crate|self|super|static|pub|using|namespace|include|require|export|type|default)\b")
CONTAINER = re.compile(r"class|interface|struct|enum|trait|protocol|object|module|impl")  # parents whose children (methods) are cross-file reachable
USEANC = re.compile(r"^(type|type_annotation|generic_type|type_arguments|decorator)$|heritage|extends|implements|superclass|base_clause")
MAX_BYTES, MAX_OUT, HUB_MIN = 1_000_000, 12, int(os.environ.get("HINTMAP_HUB_MIN") or 25)
def cmd_graph_init(a):
    root = graph_root(a.path, True)
    if getattr(a, "max_files", None) and (n := len(scope(root, ["."]))) > a.max_files:  # auto-index guard: nothing is created in the repo
        with open(os.path.join(HOME(), "injections.jsonl"), "a") as f: f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "autoindex", "root": root, "skipped": "too-big", "nfiles": n}) + "\n")
        return print(f"too-big: {n} files > {a.max_files}")
    os.makedirs(os.path.join(root, ".hintmap"), exist_ok=True)
    with open(os.path.join(root, ".hintmap", ".gitignore"), "w") as f: f.write("*\n")  # self-ignoring dir; no repo config touched
    db = sqlite3.connect(gpath(root)); db.execute("PRAGMA journal_mode=WAL"); db.executescript(GSCHEMA)
    if a.paths: db.execute("INSERT OR REPLACE INTO meta VALUES('paths',?)", (a.paths,)); db.commit()
    db.close()
    def add_line(f, ln):
        have = open(f).read().splitlines() if os.path.exists(f) else []
        if ln not in have:
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, "a") as fh: fh.write(("\n" if have and not open(f).read().endswith("\n") else "") + ln + "\n")
    add_line(os.path.join(HOME(), "graphs"), root)  # the hook trusts a graph.db only if its root is listed here
    print(f"initialized {gpath(root)}"); graph_build(root, True)
EXT = {}
for _l, _e in {"python": "py pyi", "javascript": "js jsx mjs cjs", "typescript": "ts mts cts", "tsx": "tsx", "rust": "rs", "go": "go", "java": "java", "c": "c h", "cpp": "cc cpp cxx hpp hh hxx c++ h++ ipp inl tpp cu cuh",
               "c_sharp": "cs", "ruby": "rb rake gemspec", "php": "php", "swift": "swift", "kotlin": "kt kts", "scala": "scala sc", "svelte": "svelte", "vue": "vue", "sql": "sql", "bash": "sh bash zsh ksh fish",
               "lua": "lua", "dart": "dart", "elixir": "ex exs", "groovy": "groovy gradle", "r": "r", "perl": "pl pm", "haskell": "hs", "zig": "zig", "clojure": "clj cljs cljc", "erlang": "erl hrl",
               "ocaml": "ml mli", "fsharp": "fs fsx", "nim": "nim", "julia": "jl", "objc": "m mm", "elm": "elm", "terraform": "tf", "nix": "nix", "proto": "proto", "graphql": "graphql gql", "vim": "vim",
               "solidity": "sol", "powershell": "ps1 psm1", "make": "mk mak", "cmake": "cmake", "asm": "asm s", "d": "d", "v": "v", "fortran": "f f90 f95", "cobol": "cob cbl", "lisp": "lisp el scm rkt",
               "tcl": "tcl", "pascal": "pas", "crystal": "cr", "odin": "odin", "wgsl": "wgsl glsl hlsl", "html": "html htm", "css": "css scss sass less", "xml": "xml xsd xsl"}.items():
    EXT.update({"." + x: _l for x in _e.split()})
NAMED = {"makefile": "make", "gnumakefile": "make", "dockerfile": "docker", "cmakelists.txt": "cmake", "gemfile": "ruby", "rakefile": "ruby", "justfile": "make", "jenkinsfile": "groovy", "vagrantfile": "ruby"}
GRAM = {"python": ("tree_sitter_python", "language"), "javascript": ("tree_sitter_javascript", "language"), "typescript": ("tree_sitter_typescript", "language_typescript"),
        "tsx": ("tree_sitter_typescript", "language_tsx"), "rust": ("tree_sitter_rust", "language"), "go": ("tree_sitter_go", "language"), "java": ("tree_sitter_java", "language"),
        "c": ("tree_sitter_c", "language"), "cpp": ("tree_sitter_cpp", "language"), "ruby": ("tree_sitter_ruby", "language"), "c_sharp": ("tree_sitter_c_sharp", "language"), "swift": ("tree_sitter_swift", "language"),
        "haskell": ("tree_sitter_haskell", "language"), "zig": ("tree_sitter_zig", "language"), "php": ("tree_sitter_php", "language_php"), "kotlin": ("tree_sitter_kotlin", "language"), "scala": ("tree_sitter_scala", "language"),
        "lua": ("tree_sitter_lua", "language"), "bash": ("tree_sitter_bash", "language"), "elixir": ("tree_sitter_elixir", "language")}
# ponytail: grammars are per-language wheels in ~/.hintmap/venv (`uv pip install tree-sitter-<lang>`); a language without one is indexed file-level only (path + identifier words, no symbols/edges)

def get_parser(lang):
    import importlib, tree_sitter
    try: m, f = GRAM[lang]; return tree_sitter.Parser(tree_sitter.Language(getattr(importlib.import_module(m), f)()))
    except Exception: return None
def paths_of(root, db):
    """index scope: graph.db meta 'paths' (default ['.'])"""
    v = db.execute("SELECT v FROM meta WHERE k='paths'").fetchone()
    return re.split(r"[,\s]+", v[0].strip()) or ["."] if v else ["."]
def scope(root, sp):
    if (rg := RG()):
        r = subprocess.run([rg, "--files", "--null", "--no-require-git", "-j1", "--hidden", "-g", "!.*/", *[x for d in SKIPDIR for x in ("-g", f"!{d}")], "-g", "!*.xcframework", *sp], cwd=root, capture_output=True, text=True)
        out = r.stdout.split("\0")
    else:  # ponytail: no .gitignore semantics without rg
        out = [os.path.join(d, f) for q in sp for d, ds, fs in os.walk(os.path.join(root, q)) for ds[:] in [[x for x in ds if x not in SKIPDIR and not x.startswith(".")]] for f in fs]
        out = [os.path.relpath(p, root) for p in out]
    return sorted({os.path.normpath(p) for p in out if p and not set(os.path.normpath(p).split(os.sep)) & set(SKIPDIR) and not any(x.endswith(".xcframework") for x in p.split(os.sep))})
CALLDEF = {"def", "defp", "defmodule", "defmacro", "defmacrop", "defprotocol", "defimpl", "defdelegate", "defguard", "defstruct", "defexception"}  # grammars where a definition is a macro call node with a `target` field (elixir)
def isdef(n, t): return tcls(t, DEFT) or (t == "call" and n.child_by_field_name("target") is not None and n.child_by_field_name("target").text.decode("utf8", "replace") in CALLDEF)
def defname(n):
    if n.type == "call":
        a = next((c for c in n.children if c.type == "arguments"), None); a = a.named_child(0) if a is not None else None
        while a is not None and a.type in ("call", "binary_operator") and a.named_child_count: a = a.child_by_field_name("target") or a.child_by_field_name("left") or a.named_child(0)
        w = re.findall(r"\w+", a.text.decode("utf8", "replace")) if a is not None else []
        return w[-1] if w else None
    x = n.child_by_field_name("name"); d = n
    while x is None and d is not None:  # C/C++: name sits in the declarator chain
        d = d.child_by_field_name("declarator")
        x = d if d is not None and d.type in IDT else d.child_by_field_name("name") if d is not None else None
    x = x or next((c for c in n.children if c.type in IDT), None)  # kotlin/dart: first identifier child
    if n.type == "variable_declarator":  # JS arrow/function consts only
        v = n.child_by_field_name("value")
        if v is None or not re.search(r"arrow|function|lambda|class", v.type): return None
    return x.text.decode("utf8", "replace") if x is not None and re.fullmatch(r"\w+", x.text.decode("utf8", "replace")) else None
@functools.lru_cache(maxsize=None)
def tcls(t, rx):  # node-type regex results are memoised: gparse visits ~600k nodes per 60 KLOC and re.search per node was 25% of build CPU
    return bool(rx.search(t))
SKIPT = re.compile(r"^(string|comment|line_comment|block_comment|string_literal|template_string|string_content|raw_string_literal|interpreted_string_literal)$")
NOTUSE = re.compile(r"attribute|member|field_expression|selector|keyword|pair|property"); CALLT = re.compile(r"call|invocation|^new_expression$")

def isuse(n):
    """Reference position that looks like use (call target, type, decorator, heritage, new, JSX tag), not binding. ponytail: member access (`mod.fn()`) never counts; add when receiver-is-imported-module resolution exists."""
    p = n.parent
    if p is None or tcls(p.type, NOTUSE): return 0
    if tcls(p.type, DEFT) and p.child_by_field_name("name") == n: return 0
    if tcls(p.type, CALLT) and (p.child_by_field_name("function") == n or p.child_by_field_name("constructor") == n
            or (p.child_by_field_name("name") == n and p.child_by_field_name("object") is None)): return 1
    if p.type.startswith("jsx_") and p.child_by_field_name("name") == n: return 1
    if p.type == "argument_list" and p.parent is not None and p.parent.type == "class_definition": return 1  # python bases
    if n.type == "type_identifier": return 1
    a = p
    for _ in range(3):
        if a is None: break
        if tcls(a.type, USEANC): return 1
        a = a.parent
    return 0
BODY_CAP = 300
ROUTE = re.compile(r"/[\w\-./:{}<>*~@%+=?&]*"); SYSP = re.compile(r"/(etc|usr|tmp|var|bin|dev|opt|home|Users|proc|sys|lib)(/|$)")

def route_of(n):
    """string node that is the FIRST argument of a call/decorator and starts with `/` -> (path, callee word) else None. Any grammar: argument-list node types all contain 'argument'. Template strings are cut at `${`."""
    p = n.parent
    if p is None or "argument" not in p.type or n.text[:12].lstrip(b"fFrRbBu")[1:2] != b"/": return None
    tx = n.text[:200].decode("utf8", "replace").lstrip("fFrRbBu"); s = tx[1:].split(tx[0])[0].split("${")[0]
    if len(s) < 3 or not ROUTE.fullmatch(s) or not re.search("[A-Za-z]", s) or SYSP.match(s) or p.named_child(0) != n: return None
    c = p.parent; f = c.child_by_field_name("function") or c.child_by_field_name("name") if c is not None else None
    d = c.parent.parent if c is not None and c.parent is not None and c.parent.type == "decorator" else None  # decorated handler (FastAPI/Flask/NestJS-style): its name joins the route words
    hd = defname(d.child_by_field_name("definition")) if d is not None and d.child_by_field_name("definition") is not None else ""
    return s, (re.findall(r"\w+", f.text.decode("utf8", "replace"))[-1].lower() if f is not None else ""), hd or ""
def sigof(n, src):
    """(parameter count, hash of the whitespace-normalised text from the node start up to its body) for edit-check; 0/0 for nodes without a body."""
    m = (n.child_by_field_name("value") if n.type == "variable_declarator" else None) or n
    b = m.child_by_field_name("body"); ps = m.child_by_field_name("parameters") or m.child_by_field_name("parameter")
    end = min(b.start_byte if b is not None else n.end_byte, n.start_byte + 400)
    return (ps.named_child_count if ps is not None else 0), int(h(re.sub(rb"\s+", b" ", src[n.start_byte:end]))[:7], 16)
def add_chunk(db, fid, sym, lo, hi, path, pp, name, words, pend, cap=BODY_CAP):
    """one retriever row: `path` col (file-level rows only), `name` col (symbol + parent), `body` col (identifier/comment/string words of the span, own span only)."""
    nt, nw = ntok(name); bt, bw = ntok(" ".join(words)); body = (bt + bw)[:cap]
    cid = db.execute("INSERT INTO chunks(file_id,sym_id,lo,hi) VALUES(?,?,?,?)", (fid, sym, lo, hi)).lastrowid
    db.execute("INSERT INTO ix(rowid,path,name,body) VALUES(?,?,?,?)", (cid, path, " ".join(nt + nw), " ".join(body)))
    if path: db.execute("INSERT INTO fx(rowid,path) VALUES(?,?)", (cid, path))
    pend.append((cid, (pp + nt + bt)[:150]))  # embedding words: path pieces + names + body pieces (no joined forms)
def gparse(db, fid, parser, src, path, pend):
    cur, refs, par, up = parser.parse(src).walk(), {}, None, []  # TreeCursor walk: ~2x cheaper than materialising n.children lists
    body, names = {None: []}, {}
    while True:
        n = cur.node; t = n.type; sp = par; down = True
        if tcls(t, SKIPT):
            down = False  # strings/comments hold no references (f-string interpolations are lost; ponytail) but their words feed the retriever
            if "comment" in t or "string" in t:
                body[par].append(n.text[:160].decode("utf8", "replace"))
                if "string" in t and (rt := route_of(n)):  # route strings (first call/decorator argument starting with "/") become `route` symbols: findable by their path tokens
                    sp = db.execute("INSERT INTO symbols(file_id,name,kind,lo,hi,parent_id,ar,sg) VALUES(?,?,?,?,?,?,0,0)", (fid, rt[0], "route", n.start_point[0] + 1, n.start_point[0] + 1, par)).lastrowid
                    body[sp] = [rt[1], rt[0], rt[2], "route endpoint"]; names[sp] = (rt[0] + " " + rt[2], par, n.start_point[0] + 1, n.start_point[0] + 1); sp = par
        elif tcls(t, IMPT) or (t == "export_statement" and n.child_by_field_name("source")) or (t == "call_expression" and n.text.startswith(b"require(")):
            spec = " ".join(re.findall(r"""['"][^'"\n]+['"]|[\w.:]+""", n.text[:600].decode("utf8", "replace")))[:300]
            db.execute("INSERT INTO imports VALUES(?,?,?)", (fid, spec, n.start_point[0] + 1)); body[par].append(spec); down = False
        elif isdef(n, t) and not (t == "class_declaration" and (dk := n.child_by_field_name("declaration_kind")) is not None and dk.type == "extension") and (nm := defname(n)):  # swift `extension T` is not a definition of T (it would make every extended type ambiguous); its members attach to the enclosing scope
            sp = db.execute("INSERT INTO symbols(file_id,name,kind,lo,hi,parent_id,ar,sg) VALUES(?,?,?,?,?,?,?,?)", (fid, nm, t, n.start_point[0] + 1, n.end_point[0] + 1, par, *sigof(n, src))).lastrowid
            body[sp] = []; names[sp] = (nm, par, n.start_point[0] + 1, n.end_point[0] + 1)
        elif t in IDT:
            w = n.text.decode("utf8", "replace"); body[par].append(w)
            if len(w) >= 4 and w not in COMMON and not (par is not None and n.parent is not None and tcls(n.parent.type, DEFT) and n.parent.child_by_field_name("name") == n):  # a def's own name is not a reference
                r = refs.setdefault((par, w), [0, n.start_point[0] + 1, 0]); r[0] += 1; r[2] |= isuse(n)
        if down and cur.goto_first_child(): up.append(par); par = sp; continue
        while not cur.goto_next_sibling():
            if not cur.goto_parent():
                db.executemany("INSERT INTO refs VALUES(?,?,?,?,?,?)", [(fid, s_, nm, v[1], v[0], v[2]) for (s_, nm), v in refs.items()])
                pt, pw = ntok(path); pp = pt
                add_chunk(db, fid, None, 0, 0, " ".join(pt + pw), pp, "", body[None], pend)
                for sid, (nm, pr, lo, hi) in names.items():  # a container's row also carries the first 30 words of each direct child (class chunks otherwise miss their methods' vocabulary)
                    if pr in names: body[pr] = body[pr] + body[sid][:30]
                for sid, (nm, pr, lo, hi) in names.items(): add_chunk(db, fid, sid, lo, hi, "", pp, nm + " " + (names[pr][0] if pr in names else ""), body[sid], pend)
                return
            par = up.pop()
WIN, WIN_MIN, WIN_OVER, WIN_MAX = 50, 30, 8, 400  # fallback chunker: ~50-line windows cut at a blank line (else a dedent) in the last 20 lines, 8 lines overlap, at most 400 windows per file
IDW = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
def file_chunk(db, fid, path, src, pend):
    """no grammar: a file-level row (path tokens + identifier words of the head) plus, for files over one window, overlapping line windows with their identifier words (own text only)."""
    text = src.decode("utf8", "replace"); pt, pw = ntok(path)
    add_chunk(db, fid, None, 0, 0, " ".join(pt + pw), pt, "", IDW.findall(text[:30000]), pend)
    lines = text.split("\n"); n = len(lines)
    if n <= WIN: return
    lo = 0
    for _ in range(WIN_MAX):
        end = min(lo + WIN, n)
        if end < n:
            for i in range(end, lo + WIN_MIN, -1):  # prefer a blank line, else a line that is not indented
                if not lines[i - 1].strip(): end = i; break
            else:
                for i in range(end, lo + WIN_MIN, -1):
                    if lines[i].strip() and not lines[i][0].isspace(): end = i; break
        add_chunk(db, fid, None, lo + 1, end, "", pt, "", list(dict.fromkeys(IDW.findall("\n".join(lines[lo:end])))), pend)
        if end >= n: break
        lo = max(end - WIN_OVER, lo + 1)
DATAEXT = {".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".md": "markdown", ".mdx": "markdown"}
ENVF = re.compile(r"^\.?env(\.[\w-]+)*\.(example|sample|template)$")
LOCKF = re.compile(r"(^|/)(package-lock\.json|npm-shrinkwrap\.json|pnpm-lock\.yaml|uv\.lock|poetry\.lock|Cargo\.lock|Pipfile\.lock|composer\.lock|Gemfile\.lock|go\.sum|flake\.lock|pubspec\.lock|yarn\.lock|bun\.lockb?)$|\.lock$|\.min\.\w+$|\.(map|snap)$")
DATA_CAP = 400
KEYRX = {"yaml": re.compile(r"^\s*(?:-\s+)?([A-Za-z_][\w.\-/]*)\s*:", re.M), "toml": re.compile(r"^\s*(?:\[+\s*([\w.\-]+)|([A-Za-z_][\w.\-]*)\s*=)", re.M), "env": re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", re.M)}
DATAL = {"json", "yaml", "toml", "markdown", "env"}
def stem(p):  # what another file's import text would name: file stem, or the dir for index/__init__ files
    k = os.path.splitext(os.path.basename(p))[0]; return os.path.basename(os.path.dirname(p)) or k if k in ("index", "__init__", "mod") else k
GEN_V, GENHEAD = "1", re.compile(rb"Code generated|DO NOT EDIT|@generated|auto-?generated|automatically generated", re.I)
MINIFIED_AVG = 300  # code files only (data/prose keep their long lines): chars/line over a whole file >= 2 KB: hand-written code measured <= ~160 (dense selftest 122, prose docs up to ~200); minified bundles are in the thousands
def gen_attrs(root):
    """[(glob, generated?)] from root/.gitattributes: linguist-generated / linguist-vendored (set, =true, or negated by `-`/=false). Last match wins, like git."""
    out = []
    try: lines = open(os.path.join(root, ".gitattributes"), errors="replace").read().splitlines()
    except OSError: return out
    for ln in lines:
        t = ln.split("#")[0].split()
        for a in t[1:]:
            m = re.fullmatch(r"(-?)linguist-(?:generated|vendored)(?:=(\w+))?", a)
            if m: out.append((t[0].lstrip("/"), not m.group(1) and m.group(2) not in ("false", "0")))
    return out
def gen_file(p, src, attrs):
    """generated/vendored/minified by repo metadata and content, never by a path list."""
    import fnmatch
    b, hit = os.path.basename(p), None
    for g, v in attrs:
        if fnmatch.fnmatchcase(p, g) or fnmatch.fnmatchcase(p, g.rstrip("/") + "/*") or ("/" not in g and fnmatch.fnmatchcase(b, g)): hit = v
    if hit is not None: return hit
    if GENHEAD.search(b"\n".join(src[:2000].split(b"\n")[:5])): return True
    return len(src) >= 2000 and lang_of(p) not in DATAL and len(src) / (src.count(b"\n") + 1) > MINIFIED_AVG
def lang_of(p):
    """code language by extension, or a data/doc kind (json/yaml/toml/markdown/env) — except lockfiles, minified/generated files and dotfiles."""
    b = os.path.basename(p)
    if ENVF.match(b): return "env"
    if lg := EXT.get(os.path.splitext(p)[1].lower()) or NAMED.get(b.lower()): return lg
    lg = DATAEXT.get(os.path.splitext(p)[1].lower())
    return lg if lg and not LOCKF.search(p) and not GENP.search(p) and not b.startswith(".") else None
def data_words(lang, txt):
    """config/data/doc file -> words: json/yaml/toml/env keys (+ short json string values), markdown headings. No long values, no code."""
    ws = []
    if lang == "json":
        try:
            def walk(x, d=0):
                if isinstance(x, dict):
                    for k, v in x.items(): ws.append(str(k)[:80]); d < 8 and walk(v, d + 1)
                elif isinstance(x, list):
                    for v in x[:50]: d < 8 and walk(v, d + 1)
                elif isinstance(x, str) and len(x) <= 40 and re.search("[A-Za-z]", x): vals.append(x)
            vals = []; walk(json.loads(txt)); ws = list(dict.fromkeys(ws)) + vals
        except (ValueError, RecursionError): ws = re.findall(r'"([^"\n]{1,60})"\s*:', txt)  # jsonc / broken / absurdly deep json (RecursionError from json.loads itself)
    elif lang == "markdown":
        fence = False
        for l in txt.splitlines():
            if l.lstrip().startswith("```"): fence = not fence
            elif not fence and (m := re.match(r"#{1,6}\s+(.+?)\s*#*\s*$", l)): ws.append(m.group(1)[:80])
    else: ws = [next(g for g in m.groups() if g) for m in KEYRX[lang].finditer(txt)]
    return ws
def data_chunk(db, fid, path, lang, src, pend):
    pt, pw = ntok(path); add_chunk(db, fid, None, 0, 0, " ".join(pt + pw), pt, "", data_words(lang, src.decode("utf8", "replace")), pend, DATA_CAP)
def symsigs(db, fid):
    d = {}
    for n, a, g in db.execute("SELECT name, ar, sg FROM symbols WHERE file_id=? AND kind!='route' ORDER BY lo", (fid,)): d.setdefault(n, (a, g))
    return d
def note_sigs(db, path, old, new):
    """record changed/removed signatures (old,new = {name: (ar, sg)}); a symbol back at its recorded baseline drops its row."""
    for n, (oa, og) in old.items():
        na, ng = new.get(n, (-1, 0)); r = db.execute("SELECT old_ar, old_sg FROM sigchg WHERE path=? AND name=?", (path, n)).fetchone()
        if r: oa, og = r
        if (na, ng) == (oa, og): db.execute("DELETE FROM sigchg WHERE path=? AND name=?", (path, n))
        else: db.execute("INSERT OR REPLACE INTO sigchg VALUES(?,?,?,?,?,?,?)", (path, n, oa, og, na, ng, time.time()))
def drop_file(db, fid):
    ids = db.execute("SELECT id FROM chunks WHERE file_id=?", (fid,)).fetchall(); db.executemany("DELETE FROM ix WHERE rowid=?", ids); db.executemany("DELETE FROM fx WHERE rowid=?", ids)
    for t in ("chunks", "symbols", "refs", "imports"): db.execute(f"DELETE FROM {t} WHERE file_id=?", (fid,))
_NPY = {}
def embed(db, pend, dim):
    """venv-only (numpy): chunk vector = unit mean of its WordPiece token rows from potion.i8, int8 round(x*VK)."""
    P = potion_open()
    if len(pend) <= PY_EMBED_MAX:  # numpy import alone costs ~65 ms CPU: small incremental updates use the hook's pure-python path (same vectors up to float rounding)
        db.executemany("UPDATE chunks SET v=? WHERE id=?", [(array.array("b", v).tobytes(), cid) for cid, words in pend if (v := potion_vec(P, words))]); return
    import numpy as np
    if "np" not in _NPY: raw = np.fromfile(POTION() + ".i8", dtype=np.uint8).reshape(-1, P[2] + 4); _NPY["np"] = raw[:, :4].copy().view("<f4")[:, 0], raw[:, 4:].view(np.int8)
    sc, E = _NPY["np"]
    out = []
    for cid, words in pend:
        ids = [i for w in words for i in wp_ids(P, w)]
        if not ids: continue
        x = (E[ids].astype(np.float32) * sc[ids, None]).mean(0)[:VD]; nr = np.linalg.norm(x)
        if nr: out.append((np.clip(np.rint(x / nr * VK), -127, 127).astype(np.int8).tobytes(), cid))
    db.executemany("UPDATE chunks SET v=? WHERE id=?", out)
EMBED_BATCH = 20000
PY_EMBED_MAX = int(os.environ.get("HINTMAP_PY_EMBED_MAX") or 200)
def potion_dim():
    p = potion_open()
    return min(p[2], VD) if p else 0
def imp_targets(path, spec, byp, full, sfx):
    """spec text -> unique repo file ids; per import, longest candidate wins (module before package init). ponytail: barrel re-exports, `import *`, tsconfig/package aliases, dynamic imports are not followed."""
    d = os.path.dirname(path)
    def cand(ch, drop=True):  # candidates longest-first; `drop` adds the parent (trailing symbol name)
        dots = len(ch) - len(ch.lstrip(".")); parts = [x for x in re.split(r"\.|::", ch) if x]
        base = os.path.join(d, *[".."] * (dots - 1)) if dots else ""
        return [(os.path.normpath(os.path.join(base, *parts[:k])), bool(dots)) for k in ([len(parts), len(parts) - 1] if drop else [len(parts)]) if k]
    def hit(cs):  # first candidate resolving to exactly one file; module files beat package/index files (stored under key+"\0")
        for c, rel in cs:
            c2 = re.sub(r"\.[mc]?[jt]sx?$", "", c)
            for k in (c2, c2 + "\0"):
                ids = {byp[c]} if c in byp and k == c2 else (full if rel else sfx).get(k, set())
                if len(ids) == 1: return ids
        return set()
    if m := re.search(r"""['"]([^'"]+)['"]""", spec):
        s = m.group(1); rel = s.startswith("."); groups = [[(os.path.normpath(os.path.join(d, s)) if rel else s.lstrip("@~/"), rel)]]
    elif m := re.match(r"\s*from\s+(\.*[\w.]*)\s+import\s+(.*)", spec, re.S):  # python: `from pkg import name` -> pkg/name.py if a submodule, else pkg(.py|/__init__.py)
        b = m.group(1); groups = [g for x in re.findall(r"\w+", m.group(2)) if x != "as" and hit(g := cand(b + ("" if b.endswith(".") else ".") + x, False))]
        if not groups and b.strip("."): groups = [cand(b)]
    else: groups = [cand(ch) for ch in re.findall(r"\.*\w[\w.:]*", IMPKW.sub(" ", spec))]
    return set().union(*[hit(g) for g in groups]) if groups else set()
def derive(db, inc=None):
    """Rebuild imp/edges/nodes. inc=(changed_ids, deleted_ids, names, stems): only recompute the source files an edit can affect (changed files; files referencing a name the changed files define or defined;
    files whose import text mentions an added/removed file's stem; importers of removed files) — equal to a full derive. ponytail: tie order of the out-degree cap may differ from a full derive."""
    byp, full, sfx = {}, {}, {}
    for fid, p in db.execute("SELECT id, path FROM files"):
        byp[p] = fid; k = os.path.splitext(p)[0]; keys = [k] + ([os.path.dirname(k) + "\0"] if os.path.basename(k) in ("index", "__init__", "mod") and os.path.dirname(k) else [])
        for key in keys:
            full.setdefault(key, set()).add(fid); ps = key.split("/")
            for i in range(len(ps)): sfx.setdefault("/".join(ps[i:]), set()).add(fid)
    db.execute("CREATE TEMP TABLE IF NOT EXISTS t_s(id INTEGER PRIMARY KEY)"); db.execute("CREATE TEMP TABLE IF NOT EXISTS t_c(id INTEGER PRIMARY KEY)"); db.execute("CREATE TEMP TABLE IF NOT EXISTS t_n(name TEXT PRIMARY KEY)")
    for t in ("t_s", "t_c", "t_n"): db.execute(f"DELETE FROM {t}")
    if inc is None or len(inc[3]) > 50:
        for t in ("imp", "edges", "nodes"): db.execute(f"DELETE FROM {t}")
        db.execute("INSERT INTO t_s SELECT id FROM files"); only = ""
    else:
        chg, dele, names, stems = inc; only = " AND r.file_id IN (SELECT id FROM t_s)"
        db.executemany("INSERT OR IGNORE INTO t_s VALUES(?)", [(i,) for i in chg]); db.executemany("INSERT OR IGNORE INTO t_c VALUES(?)", [(i,) for i in {*chg, *dele}])
        db.executemany("INSERT OR IGNORE INTO t_n VALUES(?)", [(n,) for n in names]); db.execute("INSERT OR IGNORE INTO t_n SELECT DISTINCT name FROM symbols WHERE file_id IN (SELECT id FROM t_c)")
        db.execute("INSERT OR IGNORE INTO t_s SELECT DISTINCT file_id FROM refs WHERE name IN (SELECT name FROM t_n)")
        for st in filter(None, stems): db.execute("INSERT OR IGNORE INTO t_s SELECT DISTINCT file_id FROM imports WHERE spec LIKE ?", (f"%{st}%",))
        db.execute("INSERT OR IGNORE INTO t_s SELECT file_id FROM imp WHERE dst IN (SELECT id FROM t_c)")
        db.execute("DELETE FROM t_s WHERE id NOT IN (SELECT id FROM files)")
        db.execute("DELETE FROM imp WHERE file_id IN (SELECT id FROM t_s) OR file_id IN (SELECT id FROM t_c)")
        db.execute("DELETE FROM edges WHERE src_file IN (SELECT id FROM t_s) OR src_file IN (SELECT id FROM t_c)")
    for fid, spec, ln in db.execute("SELECT file_id, spec, line FROM imports WHERE file_id IN (SELECT id FROM t_s)"):
        p = db.execute("SELECT path FROM files WHERE id=?", (fid,)).fetchone()[0]
        for dst in imp_targets(p, spec, byp, full, sfx) - {fid}: db.execute("INSERT OR IGNORE INTO imp VALUES(?,?,?)", (fid, dst, ln))
    db.execute("INSERT INTO edges(src_file,dst_file,kind,tier,w,line,n) SELECT file_id, dst, 'imports', 1, 1.0, line, 1 FROM imp WHERE file_id IN (SELECT id FROM t_s)")
    # T1: referenced name defined in a file the source imports. T2: name defined exactly once in the repo. T3 (>=2 defs, not imported) is dropped.
    db.create_function("cl", 1, lambda k: int(bool(CONTAINER.search(k or ""))))  # edge targets: top-level defs and methods only, never nested/local defs
    db.create_function("ok", 2, lambda n, strict: int(n not in COMMON and (len(n) >= 5 or (not strict and len(n) >= 4) or bool(strict and ("_" in n or re.search(r"[a-z][A-Z]", n))))))
    J = "CROSS JOIN" if only else "JOIN"  # incremental: start from the few source files' refs (the planner otherwise builds an automatic index over all refs, ~20 ms)
    db.execute(f"""INSERT INTO edges(src_file,src_sym,dst_file,dst_sym,kind,tier,w,line,n)
        WITH es AS (SELECT s.* FROM symbols s LEFT JOIN symbols p ON p.id=s.parent_id WHERE s.parent_id IS NULL OR cl(p.kind)),
        nm AS (SELECT name, COUNT(*) c FROM es GROUP BY name)
        SELECT r.file_id, r.sym_id, s.file_id, s.id, 'refs', CASE WHEN i.dst IS NULL THEN 2 ELSE 1 END, CASE WHEN i.dst IS NULL THEN 0.4 ELSE 1.0 END, MIN(r.line), SUM(r.n)
        FROM refs r {J} es s ON s.name=r.name AND s.file_id!=r.file_id {J} nm ON nm.name=r.name
        LEFT JOIN imp i ON i.file_id=r.file_id AND i.dst=s.file_id
        WHERE ((i.dst IS NOT NULL AND ok(r.name,0)) OR (nm.c=1 AND r.u=1 AND ok(r.name,1))){only}
        GROUP BY r.file_id, r.sym_id, s.id""")
    db.execute(f"""DELETE FROM edges WHERE id IN (SELECT id FROM (SELECT id, row_number() OVER (PARTITION BY src_file ORDER BY tier, n DESC, line) rn
        FROM edges WHERE kind='refs' AND src_file IN (SELECT id FROM t_s)) WHERE rn>{MAX_OUT})""")  # out-degree cap per source file
    db.execute("DELETE FROM nodes")
    deg = db.execute("SELECT dst_file, COUNT(DISTINCT src_file) FROM edges GROUP BY dst_file").fetchall()
    vals = sorted(c for _, c in deg); thr = max(HUB_MIN, vals[int(len(vals) * .99)] if len(vals) >= 100 else 0)
    db.executemany("INSERT INTO nodes VALUES(?,?,?)", [(f, c, int(c > thr)) for f, c in deg])  # hub = barrier flag; Faz 1 will refuse to traverse through it
VENVPY = lambda: os.environ.get("HINTMAP_GRAPH_PY") or os.path.expanduser("~/.hintmap/venv/bin/python")
def in_venv(need):
    """import `need`, else re-exec this command under the venv interpreter (once)."""
    try: [__import__(m) for m in need]
    except ImportError:
        py = VENVPY()
        if os.environ.get("HINTMAP_REEXEC") or not os.path.exists(py): sys.exit(f"hintmap: {need} not importable; create the venv: uv venv ~/.hintmap/venv && uv pip install -p ~/.hintmap/venv/bin/python tree-sitter tree-sitter-python tree-sitter-typescript tree-sitter-javascript numpy (tried {py})")
        os.environ["HINTMAP_REEXEC"] = "1"; os.execv(py, [py, SELF, *sys.argv[1:]])
SCHEMA_V = "10"  # bump = every index re-chunks on its next build (grammar/extension/chunker changes)
HIST_MAX, HIST_FILES = 5000, 30
def hist_build(db, root):
    """optional git-history index (on unless HM_HIST=0): FTS5 hist(msg) over the normalized subject+body of the last HIST_MAX non-merge commits reachable from HEAD (rowid = commit order, oldest first) + hfiles(rid, path).
    Incremental: only commits since the stored HEAD; a stored HEAD that is no longer an ancestor (rebase, older checkout) rebuilds. Stores words and paths, never source. ponytail: incremental runs never trim past HIST_MAX."""
    if not HIST() or not os.path.exists(os.path.join(root, ".git")) or not shutil.which("git"): return
    def git(*a):
        r = subprocess.run(["git", "-c", "core.quotepath=false", *a], cwd=root, capture_output=True, text=True, errors="replace"); return r.stdout if r.returncode == 0 else None
    head = (git("rev-parse", "HEAD") or "").strip()
    if not head: return
    db.executescript("CREATE VIRTUAL TABLE IF NOT EXISTS hist USING fts5(msg, tokenize='unicode61 remove_diacritics 2'); CREATE VIRTUAL TABLE IF NOT EXISTS histv USING fts5vocab(hist, 'row');"
                     "CREATE TABLE IF NOT EXISTS hfiles(rid INT, path TEXT); CREATE INDEX IF NOT EXISTS hfiles_rid ON hfiles(rid); CREATE INDEX IF NOT EXISTS hfiles_path ON hfiles(path);")
    old = db.execute("SELECT v FROM meta WHERE k='hist_head'").fetchone(); old = old and old[0]
    if old == head: return
    rng = f"{old}..HEAD" if old and git("merge-base", "--is-ancestor", old, "HEAD") is not None else "HEAD"
    if rng == "HEAD": db.executescript("DELETE FROM hist; DELETE FROM hfiles;")
    out = git("log", rng, "--no-merges", f"-n{HIST_MAX}", "--name-only", "--format=%x01%B%x02") or ""
    nxt = db.execute("SELECT COALESCE(MAX(rowid), 0) FROM hist").fetchone()[0] + 1
    for rec in reversed([r for r in out.split("\x01") if r.strip()]):  # git lists newest first; rowid grows with time
        msg, _, fl = rec.partition("\x02"); fs = [f for f in fl.split("\n") if f]
        if not fs or len(fs) > HIST_FILES: continue
        db.execute("INSERT INTO hist(rowid, msg) VALUES(?,?)", (nxt, " ".join(ntok(msg[:2000])[0]))); db.executemany("INSERT INTO hfiles VALUES(?,?)", [(nxt, f) for f in fs]); nxt += 1
    db.execute("INSERT OR REPLACE INTO meta VALUES('hist_head',?)", (head,))
def sync_roles(db):
    """files.role is written at index time; when TESTP/GENP change, re-derive it in place (no re-chunk, no SCHEMA_V bump)."""
    h = hashlib.sha1((TESTP.pattern + "\0" + GENP.pattern).encode()).hexdigest()[:12]; r = db.execute("SELECT v FROM meta WHERE k='rolep'").fetchone()
    if r and r[0] == h: return
    db.create_function("role", 1, lambda p: "test" if TESTP.search(p) else "gen" if GENP.search(p) else "src")
    db.execute("UPDATE files SET role=role(path)"); db.execute("INSERT OR REPLACE INTO meta VALUES('rolep',?)", (h,)); db.commit()
def graph_build(root, full=False, only=None):
    """only: exact relpaths to re-index (hook fast path; exit 3 = lock held, 2 = schema/model stale, caller falls back to a full build). Takes the same O_EXCL lock."""
    if not os.path.exists(gpath(root)): sys.exit(f"hintmap: no graph for {root}; run `hintmap graph init`")
    if not full and not only:  # no-op fast path: stat-walk in the plain interpreter, no venv re-exec, no tree-sitter import
        db = gopen(root)
        try:
            sync_roles(db); mv = db.execute("SELECT v FROM meta WHERE k='schema'").fetchone(); vd = db.execute("SELECT v FROM meta WHERE k='vec_dim'").fetchone()
            if mv and mv[0] == SCHEMA_V and vd and vd[0] == str(potion_dim()) and not graph_changes(root, db) and (vd[0] == "0" or os.path.exists(codes_path(root)) or (db.execute("SELECT MAX(id) FROM chunks").fetchone()[0] or 0) <= VEC_EXACT_MAX):
                db.execute("INSERT OR REPLACE INTO meta VALUES('built_at',?)", (str(time.time()),)); db.commit(); open(os.path.join(root, ".hintmap", "walk.ok"), "w").close()
                return print(f"files={db.execute('SELECT COUNT(*) FROM files').fetchone()[0]} parsed=0 unchanged=all deleted=0 (no-op)")
        finally: db.close()
    in_venv(["tree_sitter", "tree_sitter_python"]); from pathlib import Path  # numpy is imported lazily by embed() (only big batches)
    lock = os.path.join(root, ".hintmap", "build.lock")
    try: os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        if time.time() - os.path.getmtime(lock) < 120: return print("graph build already running") or (3 if only else 0)
        os.utime(lock)  # ponytail: stale-lock takeover is racy; fine for a manual/one-shot builder
    t0 = time.time(); db = gopen(root); pend, vdim = [], potion_dim()
    if only: signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # hook timeout: unwind through `finally` (rollback + unlock) instead of dying with the lock held
    try:
        for ix in ("refs_file", "refs_name"): db.execute(f"DROP INDEX IF EXISTS {ix}")  # unused by derive/hook (EXPLAIN QUERY PLAN), ~1.2 MB on a mid-size repo
        sv, vd = (db.execute(f"SELECT v FROM meta WHERE k='{k}'").fetchone() for k in ("schema", "vec_dim"))
        if not only:
            for t, col in (("refs", "u"), ("symbols", "ar")):  # old column layouts: drop (DDL autocommits); the schema check below then rebuilds everything
                if col not in [r[1] for r in db.execute(f"PRAGMA table_info({t})")]: db.execute(f"DROP TABLE {t}")
        if only and (not sv or sv[0] != SCHEMA_V or not vd or vd[0] != str(vdim)): return 2
        if not sv or sv[0] != SCHEMA_V or not vd or vd[0] != str(vdim): db.executescript("DROP TABLE IF EXISTS ixv; DROP TABLE IF EXISTS ix; DROP TABLE IF EXISTS fx;" + GSCHEMA); full = True  # schema migration / potion appeared or changed: re-chunk everything
        sync_roles(db)
        gv = db.execute("SELECT v FROM meta WHERE k='gen_v'").fetchone(); recheck = not gv or gv[0] != GEN_V; attrs = gen_attrs(root); db.execute("CREATE TABLE IF NOT EXISTS genskip(path TEXT PRIMARY KEY, size INT, mtime_ns INT)"); gs = {r[0]: (r[1], r[2]) for r in db.execute("SELECT * FROM genskip")}; gseen = {}  # first build with this rule: re-read unchanged files once so already-indexed generated ones drop out
        sp = paths_of(root, db); db.execute("INSERT OR REPLACE INTO meta VALUES('paths',?)", (" ".join(sp),))
        if full:
            db.execute("INSERT INTO ix(ix) VALUES('delete-all')"); db.execute("INSERT INTO fx(fx) VALUES('delete-all')")
            for t in ("edges", "nodes", "imp", "imports", "refs", "symbols", "chunks", "files"): db.execute(f"DELETE FROM {t}")
        old = {r["path"]: r for r in db.execute("SELECT * FROM files")}
        seen, parsers, c = set(), {}, dict(parsed=0, unchanged=0, deleted=0, big=0, nogrammar=0); chg, dele, names0, stems = set(), set(), set(), set()  # derive(inc) inputs
        if only:  # same filter as scope() minus .gitignore (ponytail: a gitignored new file the subagent wrote is indexed until the next full walk calls it "removed")
            only = {q for q in map(os.path.normpath, only) if not q.startswith("..") and not os.path.isabs(q) and not any(x in SKIPDIR or x.startswith(".") for x in (q.split(os.sep)[:-1] + ([] if ENVF.match(os.path.basename(q)) else q.split(os.sep)[-1:])))
                    and any(r == "." or q == r.strip("/") or q.startswith(r.strip("/") + "/") for r in sp)}
        for p in (sorted(only) if only is not None else scope(root, sp)):
            lang = lang_of(p)
            if not lang: continue
            try: s = os.stat(os.path.join(root, p))
            except OSError: continue
            if s.st_size > MAX_BYTES: c["big"] += 1; continue
            o = old.get(p)
            if not recheck and gs.get(p) == (s.st_size, s.st_mtime_ns): gseen[p] = gs[p]; continue  # known generated file, unchanged
            if o and not recheck and o["size"] == s.st_size and o["mtime_ns"] == s.st_mtime_ns: seen.add(p); c["unchanged"] += 1; continue  # before any grammar load: a no-op build loads none
            if lang not in parsers: parsers[lang] = None if lang in DATAL else get_parser(lang)
            with open(os.path.join(root, p), "rb") as f: src = f.read()
            if b"\0" in src[:1024]: continue  # binary masquerading as text/data
            if gen_file(p, src, attrs): gseen[p] = (s.st_size, s.st_mtime_ns); continue  # not seen: a previously indexed copy is deleted below
            seen.add(p)
            sha = h(src)
            if o and o["sha"] == sha:
                db.execute("UPDATE files SET mtime_ns=? WHERE id=?", (s.st_mtime_ns, o["id"])); c["unchanged"] += 1; continue
            row = (lang, s.st_size, s.st_mtime_ns, sha, "test" if TESTP.search(p) else "gen" if GENP.search(p) else "src", src.count(b"\n") + 1)
            if o:
                fid = o["id"]; names0.update(r[0] for r in db.execute("SELECT name FROM symbols WHERE file_id=?", (fid,))); osig = symsigs(db, fid); drop_file(db, fid)
                db.execute("UPDATE files SET lang=?,size=?,mtime_ns=?,sha=?,role=?,loc=? WHERE id=?", (*row, fid))
            else: fid = db.execute("INSERT INTO files(path,lang,size,mtime_ns,sha,role,loc) VALUES(?,?,?,?,?,?,?)", (p, *row)).lastrowid; stems.add(stem(p))
            chg.add(fid)
            try:
                if lang in DATAL: data_chunk(db, fid, p, lang, src, pend)
                elif parsers[lang]: gparse(db, fid, parsers[lang], src, p, pend)
                else: c["nogrammar"] += 1; file_chunk(db, fid, p, src, pend)
            except Exception:  # one bad file must never fail the build: drop its partial rows, index it file-level
                drop_file(db, fid); c["failed"] = c.get("failed", 0) + 1; file_chunk(db, fid, p, src, pend)
            if o: note_sigs(db, p, osig, symsigs(db, fid))
            c["parsed"] += 1
            if vdim and len(pend) >= EMBED_BATCH: embed(db, pend, vdim); pend.clear()  # bounded RAM: the words of 234k chunks held at once were ~400 MB
        for p in (set(old) & only if only is not None else set(old)) - seen:
            names0.update(r[0] for r in db.execute("SELECT name FROM symbols WHERE file_id=?", (old[p]["id"],))); stems.add(stem(p)); dele.add(old[p]["id"]); note_sigs(db, p, symsigs(db, old[p]["id"]), {})
            drop_file(db, old[p]["id"]); db.execute("DELETE FROM files WHERE id=?", (old[p]["id"],)); c["deleted"] += 1
        if only is None: db.execute("DELETE FROM genskip")
        db.executemany("INSERT OR REPLACE INTO genskip VALUES(?,?,?)", [(k, *v) for k, v in gseen.items()])
        if only is not None: db.executemany("DELETE FROM genskip WHERE path=?", [(q,) for q in only if q not in gseen])
        db.execute("DELETE FROM sigchg WHERE ts<?", (time.time() - 14 * 86400,))
        if vdim and pend: embed(db, pend, vdim)
        if c["parsed"] or c["deleted"] or full: derive(db, None if full else (chg, dele, names0, stems))
        if not only: hist_build(db, root)
        db.executemany("INSERT OR REPLACE INTO meta VALUES(?,?)", [("built_at", str(time.time())), ("schema", SCHEMA_V), ("vec_dim", str(vdim))] + ([("gen_v", GEN_V)] if only is None else []))
        if vdim and (c["parsed"] or c["deleted"] or full or not os.path.exists(codes_path(root))): write_codes(db, root)
        db.commit()
        pc, fl = (db.execute(f"PRAGMA {x}").fetchone()[0] for x in ("page_count", "freelist_count"))
        if fl * 5 > pc and only is None: db.execute("VACUUM")  # reclaim after EVERY build once >20% of pages are free
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally: db.close(); os.unlink(lock)
    if only is None: open(os.path.join(root, ".hintmap", "walk.ok"), "w").close()
    if only is not None: return print(f"only={len(only)} parsed={c['parsed']} deleted={c['deleted']} {int((time.time() - t0) * 1000)}ms")
    db = gopen(root, True); q = lambda s: db.execute(s).fetchall()
    ed = " ".join(f"{k}/T{t}={n}" for k, t, n in q("SELECT kind, tier, COUNT(*) FROM edges GROUP BY kind, tier"))
    print(f"files={len(seen)} parsed={c['parsed']} unchanged={c['unchanged']} deleted={c['deleted']} skipped(>1MB)={c['big']} file-level-only={c['nogrammar']} "
          f"symbols={q('SELECT COUNT(*) FROM symbols')[0][0]} chunks={q('SELECT COUNT(*) FROM chunks')[0][0]} vec_dim={vdim} refs={q('SELECT COUNT(*) FROM refs')[0][0]} edges: {ed or 'none'} "
          f"ambiguous-names={q('SELECT COUNT(*) FROM (SELECT 1 FROM symbols GROUP BY name HAVING COUNT(*)>1)')[0][0]} {int((time.time() - t0) * 1000)}ms")
def build_cmd(a):
    root = graph_root(a.path)
    try: return graph_build(root, a.full, a.files)
    finally:
        if not a.files:
            with contextlib.suppress(OSError): os.unlink(os.path.join(root, ".hintmap", "walk.run"))  # the hook's refresher claim (see maybe_build)
def cmd_graph_model(a):
    """one-time export of the static embedding model from the local HF cache (no download): WordPiece vocab -> potion.db, per-row-scaled int8 rows (optional PCA to --dim) -> potion.i8."""
    in_venv(["numpy"]); import numpy as np
    d = a.src or (glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--minishlab--potion-code-16M-v2/snapshots/*")) or sys.exit("hintmap: potion-code-16M-v2 not in the HF cache (no download is attempted)"))[0]  # --src: any model2vec dir (model.safetensors + tokenizer.json)
    with open(os.path.join(d, "model.safetensors"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hd = json.loads(f.read(n))["embeddings"]; E = np.frombuffer(f.read(), dtype={"F16": np.float16, "F32": np.float32}[hd["dtype"]])[: np.prod(hd["shape"])].reshape(hd["shape"]).astype(np.float32)
    if a.dim < E.shape[1]:  # uncentred PCA: dot products (hence cosines of mean vectors) are preserved up to the dropped variance
        w, V = np.linalg.eigh(E.T @ E); E = E @ V[:, ::-1][:, : a.dim]
    sc = np.abs(E).max(1) / 127; sc[sc == 0] = 1; Q = np.rint(E / sc[:, None]).astype(np.int8)
    os.makedirs(HOME(), exist_ok=True); rows = np.empty((len(E), a.dim + 4), np.uint8); rows[:, :4] = sc.astype("<f4").view(np.uint8).reshape(-1, 4); rows[:, 4:] = Q.view(np.uint8)
    rows.tofile(POTION() + ".i8"); vocab = json.load(open(os.path.join(d, "tokenizer.json")))["model"]["vocab"]
    if os.path.exists(POTION() + ".db"): os.unlink(POTION() + ".db")
    db = sqlite3.connect(POTION() + ".db"); db.executescript("CREATE TABLE vocab(tok TEXT PRIMARY KEY, id INT) WITHOUT ROWID; CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);")
    db.executemany("INSERT INTO vocab VALUES(?,?)", vocab.items()); db.execute("INSERT INTO meta VALUES('dim',?)", (str(a.dim),)); db.commit(); db.close()
    print(f"potion: {len(E)} tokens x {a.dim} int8 -> {os.path.getsize(POTION() + '.i8') / 1e6:.1f} MB + vocab {os.path.getsize(POTION() + '.db') / 1e6:.1f} MB; run `graph build --full` on each root"); build_nn()
def build_nn(k=4, cut=0.7):
    """term neighbours for soft lexical matching: per whole-word vocab token its top-k tokens with cosine >= cut -> potion.db table nn(tok, nb 'word word ...', best first). Venv-only (numpy); reads the exported rows back."""
    in_venv(["numpy"]); import numpy as np
    db = sqlite3.connect(POTION() + ".db"); d = int(db.execute("SELECT v FROM meta WHERE k='dim'").fetchone()[0])
    R = np.fromfile(POTION() + ".i8", np.uint8).reshape(-1, d + 4); E = R[:, 4:].view(np.int8).astype(np.float32) * R[:, :4].copy().view("<f4")
    voc = dict(db.execute("SELECT id, tok FROM vocab")); ids = np.array([i for i in sorted(voc) if voc[i].isalpha() and voc[i].isascii() and voc[i].islower() and len(voc[i]) >= 3])
    X = E[ids]; X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9); out = []
    for a in range(0, len(ids), 2048):
        S = X[a:a + 2048] @ X.T; S[np.arange(len(S)), np.arange(a, a + len(S))] = 0; top = np.argsort(-S, 1)[:, :k]
        out += [(voc[ids[a + r]], " ".join(voc[ids[j]] for j in t if S[r, j] >= cut)) for r, t in enumerate(top)]
    db.execute("DROP TABLE IF EXISTS nn"); db.execute("CREATE TABLE nn(tok TEXT PRIMARY KEY, nb TEXT) WITHOUT ROWID"); db.executemany("INSERT INTO nn VALUES(?,?)", [o for o in out if o[1]]); db.commit(); db.execute("VACUUM"); db.close()
    print(f"nn: {len(out)} tokens, potion.db {os.path.getsize(POTION() + '.db') / 1e6:.1f} MB")
def cmd_graph_stats(a):
    root = graph_root(a.path); db = gopen(root, True); q = lambda s: db.execute(s).fetchall()
    bt = q("SELECT v FROM meta WHERE k='built_at'")
    print(f"root: {root}\ndb: {gpath(root)} ({os.path.getsize(gpath(root)) / 1e6:.1f} MB), last build {int(time.time() - float(bt[0][0])) if bt else '?'}s ago")
    nf = max(q("SELECT COUNT(*) FROM files")[0][0], 1); sz = os.path.getsize(gpath(root))
    try: fts = q("SELECT SUM(pgsize) FROM dbstat WHERE name LIKE 'ix%'")[0][0] or 0
    except sqlite3.Error: fts = 0
    vb = q("SELECT COALESCE(SUM(LENGTH(v)),0), COUNT(v), COUNT(*) FROM chunks")[0]
    print(f"disk: graph.db {sz / nf / 1000:.1f} KB/file ({nf} files, budget 15): FTS {fts / nf / 1000:.1f} KB/file, vectors {vb[0] / nf / 1000:.1f} KB/file ({vb[1]}/{vb[2]} chunks), scope {paths_of(root, db)}")
    if sz / nf > 15000: print("  BUDGET BREACH: graph.db > 15 KB per indexed file")
    fl = q("PRAGMA freelist_count")[0][0]; pc = q("PRAGMA page_count")[0][0]
    if fl * 5 > pc: print(f"  BUDGET BREACH: {fl}/{pc} pages free (build should have vacuumed)")
    print("lang  (files / loc / symbols / defs per KLOC):")
    for lang, n, loc, sy in q("SELECT lang, COUNT(*), SUM(loc), (SELECT COUNT(*) FROM symbols s JOIN files g ON g.id=s.file_id WHERE g.lang=f.lang) FROM files f GROUP BY lang ORDER BY 3 DESC"):
        d = sy * 1000 / max(loc, 1); print(f"  {lang}: {n} / {loc} / {sy} / {d:.1f}" + ("  DEGRADED (file-level only)" if loc >= 300 and d < 5 and lang not in DATAL and lang in GRAM else ""))
    print("role: " + ", ".join(f"{r}={n}" for r, n in q("SELECT role, COUNT(*) FROM files GROUP BY role")))
    print("kind: " + ", ".join(f"{k}={n}" for k, n in q("SELECT kind, COUNT(*) FROM symbols GROUP BY kind ORDER BY 2 DESC LIMIT 8")))
    print("edges: " + ", ".join(f"{k}/T{t}={n}" for k, t, n in q("SELECT kind, tier, COUNT(*) FROM edges GROUP BY kind, tier")))
    print("top hubs (in-degree = distinct source files):")
    for p, d, hub in q("SELECT f.path, n.indeg, n.hub FROM nodes n JOIN files f ON f.id=n.file_id ORDER BY n.indeg DESC LIMIT 8"): print(f"  {d:4d}  {p}{'  [HUB]' if hub else ''}")
def cmd_graph_audit(a):
    root = graph_root(a.path); db = gopen(root, True); cache = {}
    rows = db.execute("SELECT e.kind, e.tier, e.line, sf.path sp, df.path dp, ds.name dn FROM edges e JOIN files sf ON sf.id=e.src_file JOIN files df ON df.id=e.dst_file "
                      "LEFT JOIN symbols ds ON ds.id=e.dst_sym" + (" WHERE e.tier=?" if a.tier else "") + " ORDER BY random() LIMIT ?",
                      ([int(a.tier[1])] if a.tier else []) + [a.n]).fetchall()
    for r in rows:
        if r["sp"] not in cache: cache[r["sp"]] = open(os.path.join(root, r["sp"]), errors="replace").read().splitlines()
        ls = cache[r["sp"]]; txt = ls[r["line"] - 1].strip()[:110] if r["line"] <= len(ls) else "?"
        print(f"[T{r['tier']} {r['kind']}] {r['sp']}:{r['line']}  {txt}  → {r['dp']}" + (f"::{r['dn']}" if r["dn"] else ""))
def graph_changes(root, db):
    """cheap stat-walk over indexed extensions: [] if nothing changed, else the changed/new/removed relpaths. ponytail: a first file of a never-seen extension is not noticed (use `graph build --full`)."""
    old = {p: (sz, m) for p, sz, m in db.execute("SELECT path, size, mtime_ns FROM files")}
    try: gs = {p: (sz, m) for p, sz, m in db.execute("SELECT path, size, mtime_ns FROM genskip")}  # generated files the builder skipped: unchanged = clean
    except sqlite3.Error: gs = {}
    gv = db.execute("SELECT v FROM meta WHERE k='gen_v'").fetchone(); stale = [""] if old and (not gv or gv[0] != GEN_V) else []  # "" = no path, forces a full build once (generated-file re-check)
    exts = {os.path.splitext(p)[1] for p in old}; seen, out = set(), []
    for p in scope(root, paths_of(root, db)):
        if os.path.splitext(p)[1] not in exts or not lang_of(p): continue
        try: st = os.stat(os.path.join(root, p))
        except OSError: continue
        if st.st_size > MAX_BYTES: continue
        seen.add(p)
        if old.get(p) != (st.st_size, st.st_mtime_ns) and gs.get(p) != (st.st_size, st.st_mtime_ns): out.append(p)
    return stale + out + sorted(set(old) - seen)
def run_update(root, files, timeout):
    """sync incremental re-index of exactly `files` (tree-sitter lives in the venv, so a child interpreter; ~110 ms wall per call, see doctor numbers in the notes). True = index updated; False = skipped (lock held, stale schema, timeout, no venv)."""
    py = VENVPY()
    if not os.path.exists(py): return False
    p = subprocess.Popen([py, SELF, "graph", "build", root, "--files", *files], cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, **ONE_THREAD})
    try: return p.wait(timeout=timeout) == 0
    except subprocess.TimeoutExpired: return False
    finally:
        if p.poll() is None:  # timeout or hook alarm: SIGTERM lets the build roll back and drop its lock
            p.terminate()
            try: p.wait(0.5)
            except subprocess.TimeoutExpired: p.kill()
