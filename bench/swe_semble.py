#!/usr/bin/env python3
"""semble arm on the SWE-bench localisation sample (same seed/stratification as swe_hist.py; Lite + Multilingual).

/tmp/semble-venv/bin/python bench/swe_semble.py run [--n 40] [--workers 3] --out <prefix>      # semble 0.6.1 defaults: SembleIndex.from_path(checkout), search(q, top_k=50) chunks -> files deduped, first 10 (as semble_arm.py)
python3 bench/swe_semble.py report <master swe.py json> <semble json> --out <prefix>              # MRR/Acc/Hit next to master's hook_full/hook_300 + paired bootstrap CI
Venv: `uv venv /tmp/semble-venv && uv pip install --python /tmp/semble-venv/bin/python semble==0.6.1`."""
import cache, argparse, concurrent.futures as cf, json, random, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import swe
from swe import SCRATCH, SEED, sh, stratified, fetch, gold, REPO_LANG, DATASETS, score

def sample(n):
    rng, picked = random.Random(SEED), []
    for ds, did in DATASETS.items():
        rows = fetch(did)
        for r in rows: r["gold"] = gold(r["patch"]); r["group"] = r["repo"] if ds == "lite" else REPO_LANG[r["repo"].lower()]
        for r in stratified([r for r in rows if r["gold"]], lambda r: r["group"], n, rng): r["ds"] = ds; picked.append(r)
    return picked

def run_repo(a):
    from semble import SembleIndex
    repo, tasks = a; d = cache.repo_dir(repo); res = []
    for t in sorted(tasks, key=lambda t: t["base_commit"]):
        r = dict(instance_id=t["instance_id"], repo=repo, ds=t["ds"], group=t["group"], gold=t["gold"])
        try:
            sh("rm", "-rf", str(d / ".hintmap")); cache.checkout(d, f"https://github.com/{repo}.git", t["base_commit"])
            w = time.perf_counter(); ix = SembleIndex.from_path(d); r["build_s"] = time.perf_counter() - w
            for k, q in (("full", t["problem_statement"]), ("300", t["problem_statement"][:300])):
                w = time.perf_counter(); big = ix.search(q, top_k=50); ms = (time.perf_counter() - w) * 1000
                r[f"semble_{k}"] = dict(top=list(dict.fromkeys(str(Path(x.chunk.file_path).relative_to(d)) if Path(x.chunk.file_path).is_absolute() else x.chunk.file_path for x in big))[:10], ms=ms)
        except Exception as e: r["skip"] = f"{type(e).__name__}: {str(e)[-200:]}"
        res.append(r); print(repo, t["instance_id"], r.get("skip", "ok"), flush=True)
    return res

def boot(d, rnd=random.Random(1)):
    n = len(d); bs = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(2000)); return sum(d) / n, bs[50], bs[1949]

def report(master, semble, out):
    M = {r["instance_id"]: r for r in json.load(open(master))["rows"] if "skip" not in r}; S = {r["instance_id"]: r for r in json.load(open(semble)) if "skip" not in r}
    ids = sorted(set(M) & set(S)); L = [f"{len(ids)} tasks scored by both (master {len(M)}, semble {len(S)}); gold files from patch, file-level, top 10 files. Mean [95% bootstrap CI over tasks].", ""]
    for qn in ("full", "300"):
        for scope in ("lite", "multi", "all"):
            rs = [i for i in ids if scope in (M[i]["ds"], "all")]
            if not rs: continue
            L += [f"### query={qn} set={scope} n={len(rs)}", "| arm | Acc@1 | Acc@5 | Acc@10 | Hit@5 | Hit@10 | MRR |", "|---|---|---|---|---|---|---|"]; sc = {}
            for an, src in (("hintmap master", M), ("semble 0.6.1", S)):
                key = f"hook_{qn}" if src is M else f"semble_{qn}"; sc[an] = [score(src[i], key) for i in rs]
                cell = lambda m: "{:.3f} [{:.3f}, {:.3f}]".format(*boot([s[m] for s in sc[an]]))
                L.append(f"| {an} | " + " | ".join(cell(m) for m in ("acc@1", "acc@5", "acc@10", "hit@5", "hit@10", "mrr")) + " |")
            L.append("paired semble - master: " + "; ".join(f"{m} {boot([a[m] - b[m] for a, b in zip(sc['semble 0.6.1'], sc['hintmap master'])])[0]:+.3f} [{boot([a[m] - b[m] for a, b in zip(sc['semble 0.6.1'], sc['hintmap master'])])[1]:+.3f}, {boot([a[m] - b[m] for a, b in zip(sc['semble 0.6.1'], sc['hintmap master'])])[2]:+.3f}]" for m in ("acc@5", "hit@10", "mrr")) + "\n")
    Path(out + ".md").write_text("\n".join(L)); print("\n".join(L))

def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True); r = sp.add_parser("run"); r.add_argument("--n", type=int, default=40); r.add_argument("--workers", type=int, default=3); r.add_argument("--out", required=True); r.add_argument("--pilot", type=int, default=0)
    p = sp.add_parser("report"); p.add_argument("master"); p.add_argument("semble"); p.add_argument("--out", required=True); a = ap.parse_args()
    if a.cmd == "report": return report(a.master, a.semble, a.out)
    cache.heavy(); a.workers = cache.workers(a.workers); by = {}
    for t in sample(a.n):
        if a.pilot and t["instance_id"] not in cache.PILOT[:a.pilot]: continue
        by.setdefault(t["repo"], []).append(t)
    SCRATCH.mkdir(parents=True, exist_ok=True); rows = []
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for part in ex.map(run_repo, sorted(by.items(), key=lambda kv: -len(kv[1]))): rows += part
    Path(a.out + ".json").write_text(json.dumps(rows, indent=1)); print("wrote", a.out)
if __name__ == "__main__": main()
