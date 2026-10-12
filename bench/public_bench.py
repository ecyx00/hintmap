#!/usr/bin/env python3
"""Reproduce the public hintmap benchmark table: SWE-bench Lite issue -> gold-patch-file localisation.

    python3 bench/public_bench.py [--n 100] [--workers 3] [--no-semble] [--out bench/results/public]

Data: princeton-nlp/SWE-bench_Lite (public, fetched from the HuggingFace datasets-server; 300 tasks). Ground truth = pre-image
files of the task's gold `patch` (set by the SWE-bench authors, not by hintmap; release-note files and files the patch creates are dropped).
Query = the issue text (`full`) or its first 300 chars (`300`). Repos are cloned at each task's base_commit (shallow) into ~/.cache/hintmap-bench.
Arms: hintmap (`hintmap retrieve`, default settings), bm25 (plain SQLite FTS5 bm25 over whole files, no model), semble 0.6.1 (optional, installed with uv into a temp venv).
Needs: git, network, the hintmap venv + embedding model (~/.hintmap, created by hintmap's installer), optionally uv. Sample = stratified by repo, seed 20261005 (swe.py).
Metrics: Hit@k = any gold file in top k; Acc@k = all gold files in top k; MRR of the first gold file. 95% bootstrap CI over tasks (2000 resamples); paired = arm - hintmap.
"""
import argparse, json, os, random, statistics as st, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cache, swe, swe_semble

def boot(d, rnd=random.Random(1)):
    n = len(d); b = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(2000)); return sum(d) / n, b[50], b[1949]
def cell(d): return "{:.3f} [{:.3f}, {:.3f}]".format(*boot(d))
def p50(x): return st.median(x) if x else float("nan")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100, help="tasks sampled from Lite (stratified by repo)"); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-semble", action="store_true"); ap.add_argument("--out", default=str(HERE / "results" / "public")); a = ap.parse_args(); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    py = sys.executable; sj = out.with_name(out.name + "-hintmap-bm25.json")
    if not sj.exists(): subprocess.run([py, str(HERE / "swe.py"), "--only", "lite", "--n", str(a.n), "--bm25", "--workers", str(a.workers), "--out", str(sj.with_suffix(""))], check=True)
    M = {r["instance_id"]: r for r in json.loads(sj.read_text())["rows"] if "skip" not in r}
    S = {}
    if not a.no_semble:
        mj = out.with_name(out.name + "-semble.json"); vp = Path(tempfile.gettempdir()) / "semble-venv"
        if not mj.exists():
            if not vp.exists(): subprocess.run(["uv", "venv", str(vp)], check=True); subprocess.run(["uv", "pip", "install", "--python", str(vp / "bin/python"), "semble==0.6.1"], check=True)
            subprocess.run([str(vp / "bin/python"), str(HERE / "swe_semble.py"), "run", "--n", str(a.n), "--workers", str(a.workers), "--out", str(mj.with_suffix(""))], check=True)
        S = {r["instance_id"]: r for r in json.loads(mj.read_text()) if "skip" not in r and r["ds"] == "lite"}
    ids = sorted(set(M) & set(S)) if S else sorted(M)
    arms = [("hintmap (full issue)", M, "hook_full"), ("hintmap (first 300 chars)", M, "hook_300"), ("bm25 FTS5 (full issue)", M, "bm25_full"), ("bm25 FTS5 (first 300 chars)", M, "bm25_300")]
    if S: arms += [("semble 0.6.1 (full issue)", S, "semble_full"), ("semble 0.6.1 (first 300 chars)", S, "semble_300")]
    sc = {n: [swe.score(src[i], k) for i in ids] for n, src, k in arms}
    L = [f"SWE-bench Lite file localisation, n={len(ids)} tasks (of {a.n} sampled; seed {swe.SEED}). Mean [95% bootstrap CI over tasks].", "",
         "| arm | Hit@1 | Hit@5 | Hit@10 | Acc@5 (all gold files) | MRR | query p50 ms | index p50 s |", "|---|---|---|---|---|---|---|---|"]
    for n, src, k in arms:
        ms = [src[i][k]["ms"] for i in ids if "ms" in src[i][k]]; bs = [src[i]["build_s"] for i in ids if src[i].get("build_s")]
        L.append(f"| {n} | " + " | ".join(cell([s[m] for s in sc[n]]) for m in ("hit@1", "hit@5", "hit@10", "acc@5", "mrr")) + f" | {p50(ms):.0f} | " + (f"{p50(bs):.1f}" if k.startswith("hook") and bs else "n/a") + " |")
    L += ["", "Paired difference (arm - hintmap full issue; positive = arm better), 95% bootstrap CI:", ""]
    for n, _, _ in arms[1:]: L.append(f"- {n}: " + "; ".join(f"{m} {'{:+.3f} [{:+.3f}, {:+.3f}]'.format(*boot([x[m] - y[m] for x, y in zip(sc[n], sc['hintmap (full issue)'])]))}" for m in ("hit@1", "hit@5", "mrr")))
    sz = [cache.du(p / ".hintmap") for p in cache.CACHE.glob("idx/*/*") if (p / ".hintmap").exists()]
    L += ["", f"Index size on disk (hintmap .hintmap dir, median over {len(sz)} indexed repo@commit, checkout sizes vary): {p50(sz) / 1e6:.1f} MB." if sz else ""]
    out.with_suffix(".md").write_text("\n".join(L)); print("\n".join(L))

if __name__ == "__main__": main()
