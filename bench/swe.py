#!/usr/bin/env python3
"""Issue -> gold-patch-file localisation benchmark (SWE-bench Lite + Multilingual). See bench/results/*-swe-loc.md.

python3 bench/swe.py [--hintmap <repo checkout>] [--n 100] [--workers 3] [--only lite|multi] [--keep]
Scratch (clones, HINTMAP_HOME) lives in /tmp/hm-swe-scratch; only bench/results/ is kept.
Arms: hook = hintmap retrieve (full problem_statement / first 300 chars); bm25 = plain FTS5 bm25 over whole-file text.
Gold = pre-image paths of the `patch` diff (`--- a/<path>`); files the patch creates cannot be retrieved and are dropped. CHANGELOG/HISTORY/NEWS files are excluded (nobody can name them from an issue).
"""
import cache, argparse, concurrent.futures as cf, datetime, importlib.machinery, importlib.util, json, os, random, re, shutil, sqlite3, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

SCRATCH = Path(os.environ.get("HM_SWE_TMP", "/tmp/hm-swe-scratch"))
SEED, KS, BUILD_TIMEOUT = 20261005, (1, 5, 10), 300
BM25 = bool(os.environ.get("HM_SWE_BM25"))  # whole-file bm25 baseline arm: needs a checkout (--bm25)
def set_arms():  # [(name, HM_LONGQ value)]: every arm runs on the same built index; the default single arm keeps the names hook_full / hook_300
    global ARMS, HARMS
    ARMS = [a.split("=", 1) for a in os.environ.get("HM_SWE_ARMS", "=").split(";")]; HARMS = [f"hook_{k}{'_' + n if n else ''}" for n, _ in ARMS for k in ("full", "300")]
set_arms()
DATASETS = {"lite": "princeton-nlp/SWE-bench_Lite", "multi": "SWE-bench/SWE-bench_Multilingual"}  # multilingual id: org moved to SWE-bench/
LANG = {l: ls for l, ls in {  # Multilingual has no language column; mapped by repo
    "java": "apache/druid apache/lucene javaparser/javaparser google/gson projectlombok/lombok reactivex/rxjava",
    "js/ts": "axios/axios babel/babel facebook/docusaurus immutable-js/immutable-js preactjs/preact mrdoob/three.js vuejs/core",
    "go": "caddyserver/caddy gin-gonic/gin gohugoio/hugo hashicorp/terraform prometheus/prometheus",
    "php": "briannesbitt/carbon laravel/framework php-cs-fixer/php-cs-fixer phpoffice/phpspreadsheet",
    "ruby": "faker-ruby/faker fastlane/fastlane fluent/fluentd jekyll/jekyll jordansissel/fpm rubocop/rubocop",
    "rust": "astral-sh/ruff burntsushi/ripgrep sharkdp/bat tokio-rs/axum tokio-rs/tokio uutils/coreutils nushell/nushell",
    "c": "jqlang/jq redis/redis valkey-io/valkey micropython/micropython",
    "c++": "fmtlib/fmt nlohmann/json"}.items()}
REPO_LANG = {r: l for l, rs in LANG.items() for r in rs.split()}

def fetch(ds):
    f = cache.CACHE / "datasets" / (ds.replace("/", "__") + ".json")  # the HF rows API is slow and flaky: fetched once
    if f.exists(): return json.loads(f.read_text())
    rows = []
    for o in range(0, 300, 100):
        u = f"https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split=test&offset={o}&length=100"
        rows += [r["row"] for r in json.load(urllib.request.urlopen(u, timeout=60))["rows"]]
    f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(rows)); return rows

LOG_FILE = re.compile(r"(?i)(^|/)(changelog|history|news)(\.\w+)?$")  # release-note files: no localizer can name them from an issue
def gold(patch): return [p for p in dict.fromkeys(re.findall(r"^--- a/(.+?)\s*$", patch, re.M)) if not LOG_FILE.search(p)]
def gold_lines(patch):
    """{path: changed lines (+/-) in the patch}, same file filter as gold(); 'primary' gold = the max"""
    out, cur = {}, None
    for l in patch.splitlines():
        if l.startswith("diff --git"): cur = None
        elif l.startswith("--- a/"): cur = l[6:].strip(); out.setdefault(cur, 0)
        elif cur and l[:1] in "+-" and not l.startswith(("+++", "---")): out[cur] += 1
    return {p: n for p, n in out.items() if not LOG_FILE.search(p)}

def stratified(tasks, key, n, rng):
    g = {}
    for t in tasks: g.setdefault(key(t), []).append(t)
    q = {k: max(1, int(len(v) * n / len(tasks))) for k, v in g.items()}
    for k in sorted(g, key=lambda k: -(len(g[k]) * n / len(tasks) % 1)):  # largest remainder up to n
        if sum(q.values()) >= n: break
        if q[k] < len(g[k]): q[k] += 1
    return [t for k in sorted(g) for t in rng.sample(g[k], min(q[k], len(g[k])))]

def sh(*a, cwd=None, **kw): return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True, **kw).stdout.strip()

# ---- baseline: sqlite FTS5 bm25 over whole files, one tokeniser for docs and queries ----
def toks(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 1]

def bm25_rank(root, queries, k=10):
    db = sqlite3.connect(":memory:"); db.execute("CREATE VIRTUAL TABLE d USING fts5(body, path UNINDEXED)")
    for p in sh("git", "ls-files", "-z", cwd=root).split("\0"):
        f = root / p
        try:
            if not f.is_file() or f.stat().st_size > 1_000_000: continue
            txt = f.read_text("utf-8")
        except (UnicodeDecodeError, OSError): continue
        db.execute("INSERT INTO d VALUES (?, ?)", (" ".join(toks(txt)), p))
    out = []
    for q in queries:
        t = list(dict.fromkeys(toks(q)))
        if not t: out.append([]); continue
        out.append([r[0] for r in db.execute("SELECT path FROM d WHERE d MATCH ? ORDER BY bm25(d) LIMIT ?", (" OR ".join(f'"{w}"' for w in t), k))])
    return out

# ---- per-repo worker ----
def load_hm(path):
    l = importlib.machinery.SourceFileLoader("hintmap_mod", str(path / "hintmap.py")); m = importlib.util.module_from_spec(importlib.util.spec_from_loader("hintmap_mod", l)); l.exec_module(m)
    return m

def rank(files, g):
    return next((i + 1 for i, f in enumerate(files) if f in g), None)

def run_repo(a):
    repo, tasks, hmroot, keep = a
    slug = repo.replace("/", "__"); d0 = cache.repo_dir(slug); d = d0; home = SCRATCH / "home" / slug; url = f"https://github.com/{repo}.git"
    home.mkdir(parents=True, exist_ok=True)
    for f in ("potion.db", "potion.i8"):
        if not (home / f).exists(): (home / f).symlink_to(Path.home() / ".hintmap" / f)
    env = dict(os.environ, HINTMAP_HOME=str(home), HINTMAP_GRAPH_PY=str(Path.home() / ".hintmap/venv/bin/python"))
    os.environ.update(HINTMAP_HOME=env["HINTMAP_HOME"], HINTMAP_GRAPH_PY=env["HINTMAP_GRAPH_PY"])
    hm = load_hm(hmroot); res, built, warm = [], None, False
    for t in sorted(tasks, key=lambda t: t["base_commit"]):
        r = dict(instance_id=t["instance_id"], repo=repo, group=t["group"], base_commit=t["base_commit"], gold=t["gold"])
        try:
            if built != t["base_commit"]:
                d, r["build_s"], r["cached"] = cache.index(d0, url, t["base_commit"], hmroot, env, want_checkout=BM25); built = t["base_commit"]; warm = False
            q_full, q_short = t["problem_statement"], t["problem_statement"][:300]
            if not warm: hm.retrieve(str(d), [q_short]); warm = True  # warm potion cache; not scored
            for n, fl in ARMS:
                for k_ in ("HM_LONGQ", "HM_RK", "HM_HEAD"): os.environ.pop(k_, None)  # unnamed default arm = the product defaults; an arm value is HM_LONGQ, or "ENV=val|ENV=val" (rank3 experiments)
                if fl and "=" not in fl: os.environ["HM_LONGQ"] = fl
                for kv in filter(None, fl.split("|")) if "=" in fl else ():
                    k_, _, v_ = kv.partition("="); os.environ[k_] = v_
                head = bool(os.environ.get("HM_HEAD"))
                for k, q in (("full", q_full), ("300", q_short)):
                    w = time.perf_counter(); hits = hm.retrieve(str(d), [q])[0] if not (head and len(q) > 500) else sys.modules['hm.common'].rrf(hm.retrieve(str(d), [q, q[:500]])); ms = (time.perf_counter() - w) * 1000
                    r[f"hook_{k}{'_' + n if n else ''}"] = dict(top=list(dict.fromkeys(h[0] for h in hits))[:10], ms=ms)
            if BM25: b = bm25_rank(d, [q_full, q_short]); r["bm25_full"], r["bm25_300"] = dict(top=b[0]), dict(top=b[1])
        except subprocess.TimeoutExpired as e:
            r["skip"] = f"timeout {e.timeout}s ({'index build' if 'graph' in str(e.cmd) else 'git fetch'})"; built = None
        except Exception as e:
            r["skip"] = f"{type(e).__name__}: {str(e)[-200:]}"; built = None
        res.append(r); print(repo, t["instance_id"], r.get("skip", "ok"), flush=True)
    shutil.rmtree(home, ignore_errors=True)
    return res

# ---- scoring ----
def score(r, arm):
    g, top = set(r["gold"]), r[arm]["top"]; rk = rank(top, g)
    s = {f"{m}@{k}": 0.0 for m in ("acc", "hit") for k in KS}
    for k in KS:
        s[f"acc@{k}"] = float(g <= set(top[:k])); s[f"hit@{k}"] = float(bool(g & set(top[:k])))
    s["mrr"] = 1 / rk if rk else 0.0
    return s

def summarize(rows):
    out = {}
    for ds in ("lite", "multi"):
        rs = [r for r in rows if r["ds"] == ds and "skip" not in r]
        groups = {"ALL": rs}
        for r in rs: groups.setdefault(r["group"], []).append(r)
        out[ds] = {}
        for gk, g in groups.items():
            o = {"n": len(g)}
            for arm in (*HARMS, *(("bm25_full", "bm25_300") if BM25 else ())):
                sc = [score(r, arm) for r in g]
                o[arm] = {m: sum(s[m] for s in sc) / len(sc) for m in sc[0]} if sc else {}
            ms = sorted(r[a]["ms"] for r in g for a in HARMS)
            if ms: o["latency_ms"] = dict(p50=statistics.median(ms), p95=ms[int(.95 * (len(ms) - 1))])
            out[ds][gk] = o
    return out

PAPERS = """
## Published file-level numbers (reference only, different task subsets and protocols; not comparable one-to-one)

| System | Dataset | File-level result | Uses LLM | Source |
|---|---|---|---|---|
| Agentless (localization stage) | SWE-bench Lite | "contains ground-truth file" 81.67% combined prompting+embedding (78.67% prompting only, 67.67-70.33% embedding only), 300 tasks, GPT-4o (Table 2) | yes (GPT-4o + embedding retrieval) | Xia et al. 2024, arXiv:2407.01489 |
| LocAgent | Loc-Bench / SWE-bench Lite subset | file Acc@1/3/5 on SWE-bench Lite: LocAgent Claude-3.5 77.74/91.97/94.16; Qwen2.5-32B(ft) 75.91/90.51/92.70; Agentless GPT-4o 67.15/74.45/74.45; SWE-agent Claude-3.5 77.37/87.23/90.15 (Table 4) | yes (agent, fine-tuned Qwen / Claude) | Chen et al. 2025, arXiv:2503.09089 |
| CoSIL | SWE-bench Lite | file Top-1/3/5 on Lite, Qwen2.5-Coder-32B: CoSIL 61.3/78.0/83.7; Agentless-FL 58.3/73.0/77.3; LocAgent 64.7/72.7/74.0 (Table II) | yes (call-graph agent) | Jiang et al. 2025, arXiv:2503.22424 |
| BM25 / embeddings | SWE-bench Lite subset | file Acc@1/3/5 38.69/51.82/61.68 (LocAgent Table 4; percentages imply a 137-task Lite subset, our inference); E5-base-v2 49.64/74.45/80.29; CodeRankEmbed 52.55/77.74/84.67 | no | Chen et al. 2025 (LocAgent) |
"""

def md(res):
    L = [f"# SWE issue -> gold-file localisation ({res['date']}, hintmap `{res['hintmap_sha']}`)", "",
         f"Seed {SEED}; {res['n_scored']} scored / {res['n_sampled']} sampled; gold = files edited by `patch` (pre-image paths; files the patch creates are dropped). "
         "`hook_*` = hintmap `retrieve`, top 10 files; `bm25_*` = plain FTS5 bm25 over whole-file text. `full` = whole problem_statement, `300` = first 300 chars.", ""]
    for ds, name in (("lite", "SWE-bench Lite (Python, by repo)"), ("multi", "SWE-bench Multilingual (by language)")):
        L += [f"## {name}", "", "| group | n | arm | Acc@1 | Acc@5 | Acc@10 | Hit@1 | Hit@5 | Hit@10 | MRR | hook ms p50/p95 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for gk, o in sorted(res["summary"][ds].items(), key=lambda kv: (kv[0] != "ALL", kv[0])):
            if not o["n"]: continue
            for arm in (*HARMS, *(("bm25_full", "bm25_300") if BM25 else ())):
                s = o[arm]; lat = f"{o['latency_ms']['p50']:.0f}/{o['latency_ms']['p95']:.0f}" if arm.startswith("hook") and "latency_ms" in o else ""
                L.append(f"| {gk} | {o['n']} | {arm} | " + " | ".join(f"{s[m]:.2f}" for m in ("acc@1", "acc@5", "acc@10", "hit@1", "hit@5", "hit@10", "mrr")) + f" | {lat} |")
        L.append("")
    sk = [r for r in res["rows"] if "skip" in r]
    L += [f"## Skipped ({len(sk)})", ""] + [f"- {r['instance_id']}: {r['skip']}" for r in sk] + ["", f"Wall time {res['wall_s']:.0f}s, workers {res['workers']}. Latency was measured with several repos running concurrently, so it is an upper bound.", PAPERS]
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hintmap", default=str(Path(__file__).resolve().parent.parent)); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED, help="sampling seed (fresh-sample check)"); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--only", choices=list(DATASETS)); ap.add_argument("--keep", action="store_true"); ap.add_argument("--bm25", action="store_true"); ap.add_argument("--pilot", type=int, default=0, help="only the first N of cache.PILOT (known losses first)"); ap.add_argument("--new-repos", type=int, default=-1, help="small-disk sample: -1 = no filter; N = cached repos plus at most N uncached ones per dataset"); ap.add_argument("--out", default=""); ap.add_argument("--arms", default="", help="name=HM_LONGQ;name=... e.g. 'base=;L1=L1;L4=L1,L2'")
    a = ap.parse_args(); cache.heavy(); cache.clean_partial(); a.workers = cache.workers(a.workers)
    if a.bm25: os.environ["HM_SWE_BM25"] = "1"; globals()["BM25"] = True  # worker processes re-import this module: env carries it
    if a.arms: os.environ["HM_SWE_ARMS"] = a.arms; set_arms()
    hmroot = Path(a.hintmap).resolve(); t0 = time.time(); rng = random.Random(a.seed); picked = []
    for ds, did in DATASETS.items():
        if a.only and a.only != ds: continue
        rows = fetch(did)
        for r in rows: r["gold"] = gold(r["patch"]); r["group"] = r["repo"] if ds == "lite" else REPO_LANG[r["repo"].lower()]
        rows = [r for r in rows if r["gold"]]
        if a.new_repos >= 0:  # small-disk sample: tasks whose index is already cached (no fetch, no build), plus all tasks of at most N repos that have none
            hit = lambda r: any(cache.CACHE.glob(f"idx/*/{r['repo'].replace('/', '__')}@{r['base_commit'][:12]}"))
            have = {r["repo"] for r in rows if hit(r)}; new = sorted({r["repo"] for r in rows} - have); random.Random(a.seed).shuffle(new); new = set(new[:a.new_repos])
            rows = [r for r in rows if hit(r) or r["repo"] in new]
        for r in stratified(rows, lambda r: r["group"], a.n, rng): r["ds"] = ds; picked.append(r)
    if a.pilot: picked = [r for r in picked if r["instance_id"] in cache.PILOT[:a.pilot]]
    by = {}
    for r in picked: by.setdefault(r["repo"], []).append(r)
    SCRATCH.mkdir(parents=True, exist_ok=True); rows = []
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for part in ex.map(run_repo, [(k, v, hmroot, a.keep) for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))]): rows += part
    ds_of = {t["instance_id"]: t["ds"] for t in picked}
    for r in rows: r["ds"] = ds_of[r["instance_id"]]
    sha = sh("git", "rev-parse", "--short", "HEAD", cwd=hmroot); date = datetime.date.today().isoformat()
    res = dict(date=date, hintmap_sha=sha, seed=a.seed, n_sampled=len(rows), n_scored=sum("skip" not in r for r in rows), workers=a.workers, wall_s=time.time() - t0, summary=summarize(rows), rows=rows)
    base = Path(a.out) if a.out else Path(__file__).resolve().parent / "results" / f"{date}-{sha}-swe-loc"
    base.parent.mkdir(exist_ok=True); base.with_name(base.name + ".json").write_text(json.dumps(res, indent=1)); base.with_name(base.name + ".md").write_text(md(res)); print("wrote", base)
    shutil.rmtree(SCRATCH, ignore_errors=True)

if __name__ == "__main__": main()
