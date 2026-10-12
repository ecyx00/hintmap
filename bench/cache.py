"""Shared persistent bench cache (HM_BENCH_CACHE, default ~/.cache/hintmap-bench). Every bench script gets its checkouts and hintmap indexes from here.

repos/<slug>            one clone per repo, checked out at whatever commit was asked last (git fetch --depth 1 of the commit, only when missing)
idx/<code key>/<slug>@<commit>[tag]/.hintmap + build_s    index built once per (repo, commit, indexing code, build env); reused by every later run
A retrieval-only change keeps the code key, so a rerun is queries only. The key hashes hm/index.py (minus the TESTP/GENP role regexes, which `reroles` re-applies in place), hm/common.py and hm/graph.py;
other-key index dirs are deleted on the first build under a new key (disk is tight).
heavy(): re-execs the script under `lockf /tmp/heavy-bench.lock` (once); workers(n): at most 4."""
import hashlib, os, re, shutil, sqlite3, subprocess, sys, time
from pathlib import Path

CACHE = Path(os.environ.get("HM_BENCH_CACHE") or Path.home() / ".cache" / "hintmap-bench")
LOCK, MAX_WORKERS, BUILD_TIMEOUT = "/tmp/heavy-bench.lock", 4, 900
# --pilot N = the first N of these: Multilingual losses vs semble (block20 report) first, then wins/lite tasks as controls.
PILOT = """apache__druid-16875 preactjs__preact-3454 fmtlib__fmt-3272 astral-sh__ruff-15330 rubocop__rubocop-13705 apache__druid-14092 reactivex__rxjava-7597 sharkdp__bat-2393
uutils__coreutils-6690 preactjs__preact-4436 jqlang__jq-2681 vuejs__core-11870 axios__axios-5892 briannesbitt__carbon-2981 fluent__fluentd-4655 gin-gonic__gin-2121 projectlombok__lombok-3602""".split()

def sh(*a, cwd=None, **kw): return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True, **kw).stdout.strip()
def heavy():
    if os.environ.get("HM_BENCH_LOCKED") or not shutil.which("lockf"): return
    os.environ["HM_BENCH_LOCKED"] = "1"; os.execvp("nice", ["nice", "lockf", LOCK, sys.executable, *sys.argv])
def workers(n): return max(1, min(n, MAX_WORKERS))
def repo_dir(slug):
    d = CACHE / "repos" / slug.replace("/", "__"); d.mkdir(parents=True, exist_ok=True); (d / ".used").touch(); return d  # .used mtime = LRU clock
def du(p): return int(subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True).stdout.split()[0] or 0) * 1024 if Path(p).exists() else 0
def repo_bytes(d): return du(d) + sum(du(x) for x in (CACHE / "idx").glob(f"*/{d.name}@*"))
def make_room(keep, need=0.5e9):
    """Disk guard before any clone / index build. Evicts whole repos (clone + all its indexes), least recently used first, never `keep` nor a repo touched in the last 30 min (parallel workers are using it),
    until cache <= HM_BENCH_CAP_GB (default 6) and free disk after `need` bytes >= HM_BENCH_MIN_FREE_GB (default 3); SystemExit with a message when that is impossible."""
    cap, floor = float(os.environ.get("HM_BENCH_CAP_GB", 6)) * 1e9, float(os.environ.get("HM_BENCH_MIN_FREE_GB", 3)) * 1e9
    repos = sorted((x for x in (CACHE / "repos").iterdir() if x.is_dir() and x.name != keep and time.time() - ((x / ".used").stat().st_mtime if (x / ".used").exists() else 0) > 1800), key=lambda x: (x / ".used").stat().st_mtime if (x / ".used").exists() else 0)
    free = lambda: shutil.disk_usage(CACHE).free - need
    total = lambda: du(CACHE / "repos") + du(CACHE / "idx")
    while (total() > cap or free() < floor) and repos:
        v = repos.pop(0); print(f"[cache] evict {v.name} ({repo_bytes(v) / 1e9:.2f} GB)", file=sys.stderr)
        for x in (CACHE / "idx").glob(f"*/{v.name}@*"): shutil.rmtree(x, ignore_errors=True)
        shutil.rmtree(v, ignore_errors=True)
    if free() < floor: sys.exit(f"[cache] only {free() / 1e9:.1f} GB would remain free (< {floor / 1e9:.0f} GB) and nothing left to evict; free disk or lower HM_BENCH_CAP_GB")
def clean_partial():
    """drop interrupted index copies and repos whose clone died mid-fetch"""
    for x in (CACHE / "idx").glob("*/*/.hintmap.tmp"): shutil.rmtree(x, ignore_errors=True)
    for x in (CACHE / "repos").glob("*/.git/shallow.lock"): x.unlink(missing_ok=True)
def code_key(hmroot):
    h = hashlib.sha1()
    for f in ("index.py", "common.py", "graph.py"):
        h.update("".join(l for l in (Path(hmroot) / "hm" / f).read_text().splitlines(True) if not re.match(r"(TESTP|GENP) = ", l)).encode())
    return h.hexdigest()[:10]
def checkout(d, url, commit, fetch=True):
    """d at commit (shallow fetch when absent); git clean so every index sees the same tree"""
    if not (d / ".git").exists(): sh("git", "init", "-q", cwd=d); sh("git", "remote", "add", "origin", url, cwd=d)
    if subprocess.run(["git", "rev-parse", "-q", "--verify", commit + "^{commit}"], cwd=d, capture_output=True).returncode: make_room(d.name); sh("git", "fetch", "-q", "--depth", "1", "origin", commit, cwd=d, timeout=1800)
    if subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True).stdout.strip() != commit: sh("git", "checkout", "-qf", commit, cwd=d)
    sh("git", "clean", "-fdqx", cwd=d)
def reroles(root, hmroot):
    sys.path.insert(0, str(hmroot)); from hm import index as hix
    db = sqlite3.connect(Path(root) / ".hintmap" / "graph.db"); hix.sync_roles(db); db.close()
def index(d, url, commit, hmroot, env, tag="", args=(), want_checkout=True):
    """(dir holding .hintmap, build seconds, cached?). want_checkout: d is checked out at commit and gets a private copy of the index (hook / agent scripts run in it); else the cached index dir itself is returned and no git is touched when it exists."""
    hmroot = Path(hmroot); key = code_key(hmroot); o = CACHE / "idx" / key / f"{d.name}@{commit[:12]}{tag}"
    if (CACHE / "idx").exists() and not (CACHE / "idx" / key).exists():
        for x in (CACHE / "idx").iterdir(): shutil.rmtree(x, ignore_errors=True)
    hit = (o / ".hintmap" / "graph.db").exists()
    if not hit:
        make_room(d.name); checkout(d, url, commit); shutil.rmtree(d / ".hintmap", ignore_errors=True); t0 = time.perf_counter()
        subprocess.run([sys.executable, str(hmroot / "hintmap"), "graph", "init", str(d), *args], env=env, check=True, capture_output=True, text=True, timeout=BUILD_TIMEOUT)
        bs = time.perf_counter() - t0; o.mkdir(parents=True, exist_ok=True); shutil.rmtree(o / ".hintmap.tmp", ignore_errors=True)
        shutil.copytree(d / ".hintmap", o / ".hintmap.tmp"); os.rename(o / ".hintmap.tmp", o / ".hintmap"); (o / "build_s").write_text(str(bs))
        if not want_checkout: shutil.rmtree(d / ".hintmap", ignore_errors=True)
    bs = float((o / "build_s").read_text()); reroles(o, hmroot)
    if not want_checkout: return o, bs, hit
    checkout(d, url, commit)
    shutil.rmtree(d / ".hintmap", ignore_errors=True); shutil.copytree(o / ".hintmap", d / ".hintmap")
    return d, bs, hit
