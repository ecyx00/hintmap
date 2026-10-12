"""hintmap.report: see README Layout. `hintmap report [--since DATE]`: the live A/B of both hooks from injections.jsonl + subagent transcripts."""
import collections, glob, json, os, random, re, statistics as st, time
from hm.common import HOME
from hm.jit import grep_ident, read_cmd

H = os.path.expanduser
EXT = r"(?:swift|py|ts|tsx|js|mjs|zig|rs|go|md|json|sh|m|mm|h|c|cpp|toml|ya?ml|plist|xcstrings|strings|metal|kt)"
PATH = re.compile(r"(?<![\w/.-])((?:/|\.{0,2}/)?[\w@+.-]+(?:/[\w@+.-]+)*\." + EXT + r")\b")
CD = re.compile(r'^\s*cd\s+("?)([^"&;\s]+)\1\s*(?:&&|;)\s*')
SEARCH = re.compile(r"^(rg|grep|find|fd|ls|git\s+(grep|log|ls-files|show)|wc|hintmap)\b")
READ = re.compile(r"^(sed\s+-n|cat(?!\s*>)|head|tail|nl|awk|less|bat)\b")
AGENT_SINCE = "2026-10-06T22:24"  # first holdout record that logs `emitted`, same hook version for both arms
def norm(p, cwd, root):
    for q in ([p] if p.startswith("/") else [os.path.join(cwd, p), os.path.join(root, p)]):
        q = os.path.normpath(q)
        if q.startswith(root + "/") and os.path.isfile(q): return q[len(root) + 1:]
def isgold(p): return p and not p.startswith(("docs/", "notes/")) and not p.endswith(".md")
def rlen(c):
    if isinstance(c, str): return len(c)
    return sum(len(x.get("text", "")) for x in c if isinstance(x, dict)) if isinstance(c, list) else 0
def transcript(f, root):
    """-> dict of one subagent transcript: turns, ctx tokens, and ev = [(kind, paths, result_bytes, single_ident_grep)] with kind w=write r=read s=search o=other."""
    ev, res, seen, ctx, turns, brief, braw, ids = [], {}, set(), 0, 0, None, None, set()
    use, tools, whole, t0 = {}, collections.Counter(), set(), None  # use: last usage per message id (streamed chunks repeat); whole: tool_use ids of whole-file reads
    for l in open(f):
        try: d = json.loads(l)
        except ValueError: continue
        msg = d.get("message") or {}; c = msg.get("content"); t0 = t0 or d.get("timestamp")
        if d.get("type") == "user":
            if brief is None: braw = c if isinstance(c, str) else json.dumps(c); brief = braw.split("## Code locations")[0]
            if isinstance(c, list):
                for x in c:
                    if isinstance(x, dict) and x.get("type") == "tool_result": res[x.get("tool_use_id")] = rlen(x.get("content"))
        if d.get("type") != "assistant": continue
        if msg.get("id") not in seen:
            seen.add(msg.get("id")); turns += 1; u = msg.get("usage") or {}; ctx += u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        if msg.get("id") and msg.get("usage"): use[msg["id"]] = msg["usage"]
        for c in msg.get("content") or []:
            if not (isinstance(c, dict) and c.get("type") == "tool_use"): continue
            i = c.get("input") or {}; n = c["name"]; ids.add(c.get("id")); uid = c.get("id"); g1 = False
            if n == "Read" and not i.get("offset") and not i.get("limit"): whole.add(uid)
            if n == "LSP": tools["lsp"] += 1
            elif n == "Skill" and "code-navigation" in str(i.get("skill") or i): tools["skill"] += 1
            elif n == "Bash":
                mh = re.match(r"^\s*(?:cd\s+\S+\s*&&\s*)?(?:\S*/)?hintmap\s+(def|refs|show|impact|tests-for|edit-check|search|map|callees|changed|q)\b", i.get("command") or "")
                if mh: tools[mh.group(1)] += 1
                if read_cmd(i.get("command") or ""): whole.add(uid)
            if n in ("Read", "Edit", "Write", "MultiEdit"):
                p = norm(i.get("file_path", ""), root, root); ev.append(("w" if n != "Read" else "r", {p} if p else set(), uid, False))
            elif n in ("Grep", "Glob"): ev.append(("s", set(), uid, n == "Grep" and bool(re.fullmatch(r"[A-Za-z_$][\w$]{2,}", i.get("pattern") or ""))))
            elif n == "Bash":
                s = i.get("command") or ""; cwd = root; mm = CD.match(s)
                if mm: cwd = os.path.join(root, mm.group(2)) if not mm.group(2).startswith("/") else mm.group(2); s = s[mm.end():]
                ps = {x for x in (norm(p, cwd, root) for p in PATH.findall(s)) if x}; head = s.lstrip()
                if re.search(r"python3?\s+-\s*<<", head) and re.search(r"\.write|write_text", s): k = "w"
                elif re.search(r"sed\s+-i|perl\s+-p?i|\btee\b|cat\s*>>?|(?<![<>=\d&-])>\s*[\w./]", head):
                    k = "w"; tg = re.findall(r"(?:>>?|tee(?:\s+-a)?)\s*([\w./@+-]+)", s) + [x[-1] for x in re.findall(r"(?:sed|perl)\s+-\S*i\S*\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s+((?:[\w./@+-]+\s*)+)", s) for x in [x.split()]]
                    ps = {x for x in (norm(p, cwd, root) for p in tg) if x}
                elif SEARCH.match(head): k = "s"; g1 = grep_ident(s) is not None
                elif READ.match(head): k = "r"
                else: k = "o"
                ev.append((k, ps, uid, g1))
    bill = {k: sum(u.get(f, 0) for u in use.values()) for k, f in (("in", "input_tokens"), ("cw", "cache_creation_input_tokens"), ("cr", "cache_read_input_tokens"), ("out", "output_tokens"))}
    bill["eq"] = (bill["in"] + 1.25 * bill["cw"] + 0.1 * bill["cr"] + 5 * bill["out"]) / 1e6  # Mtok at the plain input price
    return dict(ev=[(k, ps, res.get(u, 0), g1) for k, ps, u, g1 in ev], turns=turns, ctx=ctx, brief=brief or "", braw=braw or "", ids=ids, tools=tools, wbytes=sum(res.get(u, 0) for u in whole), bill=bill, day=(t0 or "")[:10])
def boot(a, b, f=st.median, n=2000):
    random.seed(1); d = sorted(f([random.choice(a) for _ in a]) - f([random.choice(b) for _ in b]) for _ in range(n)); return round(d[int(.025 * n)], 1), round(d[int(.975 * n)], 1)
def line(lab, a, b, fmt="%7.1f"):
    if not a or not b: return print(f"  {lab:14} n/a")
    print(f"  {lab:14} med {fmt % st.median(a)} vs {fmt % st.median(b)}  d95={boot(a, b)}   mean {fmt % st.mean(a)} vs {fmt % st.mean(b)} d95={boot(a, b, st.mean)}")
def load(since):
    rows = [json.loads(l) for l in open(os.path.join(HOME(), "injections.jsonl")) if l.strip()]
    meta = {}
    for m in glob.glob(H("~/.claude/projects/*/*/subagents/*.meta.json")):
        try:
            if os.path.getmtime(m[:-10] + ".jsonl") < _ts(since) - 86400: continue
            meta[m] = json.load(open(m))
        except (OSError, ValueError): pass
    return rows, meta
def _ts(s):
    import time; return time.mktime(time.strptime((s + "T00:00")[:16], "%Y-%m-%dT%H:%M"))
def agent_ab(rows, meta, since):
    P = {r["tool_use_id"]: r for r in rows if r.get("event") == "pretooluse-agent" and r.get("tool_use_id") and r["ts"] >= since}
    R = []
    for m, mt in meta.items():
        r = P.get(mt.get("toolUseId"))
        if not r or not r.get("root") or r.get("arm") not in ("inject", "holdout"): continue
        t = transcript(m[:-10] + ".jsonl", r["root"]); ev = t["ev"]
        if not t["turns"] or (r["arm"] == "inject" and "## Code locations" not in t["braw"]): continue
        gold = {p for k, ps, _, _ in ev if k == "w" for p in ps if isgold(p)}; em = r.get("emitted") or []
        first = next((j for j, (k, ps, _, _) in enumerate(ev) if ps & gold), None)
        R.append(dict(arm=r["arm"], root=r["root"].split("/")[-1], atype=r.get("agent_type"), turns=t["turns"], ctx=t["ctx"], calls=len(ev), search=sum(k == "s" for k, *_ in ev), gold=gold, em=em,
                      sb=None if first is None else sum(k == "s" for k, *_ in ev[:first]), em_touched=sum(any(e in ps for _, ps, _, _ in ev) for e in em)))
    print(f"== Agent hook A/B since {since} (inject vs holdout, 95% bootstrap CI of the difference)")
    for name, sel in (("all", lambda o: True), ("coder", lambda o: o["atype"] == "coder")):
        I = [o for o in R if o["arm"] == "inject" and sel(o)]; Hh = [o for o in R if o["arm"] == "holdout" and sel(o)]
        print(f"-- {name}: inject {len(I)}  holdout {len(Hh)}")
        for lab, k in (("turns", "turns"), ("ctx_Mtok", "ctx"), ("calls", "calls"), ("search", "search")):
            f = (lambda o, k=k: o[k] / 1e6) if k == "ctx" else (lambda o, k=k: o[k]); line(lab, [f(o) for o in I], [f(o) for o in Hh], "%7.2f")
        for lab, arm in (("inject", I), ("holdout", Hh)):
            sb = [o["sb"] for o in arm if o["sb"] is not None]; print(f"  {lab}: tasks with edits {sum(1 for o in arm if o['gold'])}, listed files touched {sum(o['em_touched'] for o in arm)}/{sum(len(o['em']) for o in arm)}, search before 1st gold med {st.median(sb) if sb else '-'}")
def jit_ab(rows, meta, since):
    J = [r for r in rows if r.get("event") == "pretooluse-bash" and r.get("arm")]
    if since: J = [r for r in J if r["ts"] >= since]
    print(f"\n== JIT hook (pretooluse-bash): {len(J)} logged actions" + (f" since {J[0]['ts']}" if J else ""))
    if not J: return print("  nothing logged yet (hook not registered, or HINTMAP_JIT=0)")
    ms = sorted(r["ms"] for r in J); cp = sorted(r.get("cpu_ms", 0) for r in J)
    print(f"  hook ms p50={ms[len(ms) // 2]} p95={ms[int(len(ms) * .95)]}  cpu p50={cp[len(cp) // 2]}  (in-process, acted calls only)")
    c = collections.Counter((r["arm"], r["kind"]) for r in J)
    for (a, k), n in sorted(c.items()): print(f"  {a:8} {k:18} {n}")
    byag, bytu = {}, {}
    for r in J:
        if r.get("agent_id"): byag[r["agent_id"]] = r["arm"]
        if r.get("tool_use_id"): bytu[r["tool_use_id"]] = r["arm"]
    T = {"inject": [], "holdout": []}
    for m in meta:
        f = m[:-10] + ".jsonl"; aid = os.path.basename(f)[6:-6]
        arm = byag.get(aid)
        if not arm and bytu:
            try: txt = open(f).read()
            except OSError: continue
            arm = next((a for u, a in bytu.items() if len(u) >= 10 and f'"{u}"' in txt), None)  # exact quoted id: a short or fake id ("ta") must not match every transcript (that put 273 tasks in one arm)
        if not arm: continue
        root = next((r["root"] for r in J if r.get("agent_id") == aid and r.get("root")), None) or "/"
        t = transcript(f, root); ev = t["ev"]
        T[arm].append(dict(turns=t["turns"], grep1=sum(g for *_, g in ev), search=sum(k == "s" for k, *_ in ev), rbytes=sum(b for k, _, b, _ in ev if k == "r"), obytes=sum(b for _, _, b, _ in ev), ctx=t["ctx"]))
    print(f"  tasks joined to a transcript: inject {len(T['inject'])}  holdout {len(T['holdout'])} (a task is in the arm of its first acted call; arm = hash(agent_id))")
    for lab, k, fmt in (("single-id greps", "grep1", "%7.1f"), ("search calls", "search", "%7.1f"), ("read bytes KB", "rbytes", "%7.1f"), ("tool out KB", "obytes", "%7.1f"), ("turns", "turns", "%7.1f"), ("ctx Mtok", "ctx", "%7.2f")):
        sc = 1e3 if k in ("rbytes", "obytes") else 1e6 if k == "ctx" else 1
        line(lab, [o[k] / sc for o in T["inject"]], [o[k] / sc for o in T["holdout"]], fmt)
def adoption(meta, since, rows=()):
    """free-choice toolbox use per subagent and per day: hintmap CLI / LSP / skill loads versus single-identifier greps and whole-file read bytes, plus billed cost (in + 1.25 cache write + 0.1 cache read + 5 output, in input-token equivalents)."""
    A = []; parm = {r["agent_id"]: r["arm"] for r in rows if r.get("event") == "subagent-start" and r.get("arm") and r.get("agent_id")}
    for m in sorted(meta):
        f = m[:-10] + ".jsonl"
        try: t = transcript(f, "/")
        except OSError: continue
        day = t["day"] or time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(f)))
        if not t["turns"] or day < (since or "")[:10]: continue
        tl = t["tools"]; A.append(dict(id=os.path.basename(f)[6:14], day=day, typ=(meta[m].get("agentType") or "?")[:10], turns=t["turns"], hm=sum(v for k, v in tl.items() if k not in ("lsp", "skill")), lsp=tl["lsp"], skill=tl["skill"], arm=parm.get(os.path.basename(f)[6:-6], "-"),
                                      g1=sum(g for *_, g in t["ev"]), wkb=t["wbytes"] / 1e3, **t["bill"]))
    print(f"\n== Free-choice adoption since {since or 'start'}: {len(A)} subagent transcripts (hm = hintmap def|refs|show|impact|tests-for|edit-check|search|map|callees|changed|q via Bash; g1 = single-identifier greps; wKB = whole-file read output; eq = Mtok at input price, in + 1.25 cw + 0.1 cr + 5 out)")
    cols = (("turns", "%5d"), ("hm", "%3d"), ("lsp", "%3d"), ("skill", "%5d"), ("g1", "%3d"), ("wkb", "%8.0f"), ("in", "%7d"), ("cw", "%9d"), ("cr", "%10d"), ("out", "%7d"), ("eq", "%7.2f"))
    print("  " + "id       day        type       " + " ".join(f"{c:>{len(f % 0)}}" for c, f in cols))
    for o in sorted(A, key=lambda o: (o["day"], o["id"])): print(f"  {o['id']:8} {o['day']} {o['typ']:10} " + " ".join(f % o[c] for c, f in cols))
    print("  per day:")
    print("  day         n  " + " ".join(f"{c:>{len(f % 0)}}" for c, f in cols))
    for d in sorted({o["day"] for o in A}):
        g = [o for o in A if o["day"] == d]; print(f"  {d} {len(g):3d}  " + " ".join(f % sum(o[c] for o in g) for c, f in cols))
    for arm in ("inject", "holdout"):
        g = [o for o in A if o["arm"] == arm]
        if g: print(f"  SubagentStart pointer arm {arm:7}: n={len(g)}  used hm {sum(1 for o in g if o['hm'])}  skill {sum(1 for o in g if o['skill'])}  mean hm calls {sum(o['hm'] for o in g) / len(g):.2f}  mean g1 {sum(o['g1'] for o in g) / len(g):.2f}  mean wKB {sum(o['wkb'] for o in g) / len(g):.0f}")
    if A: print(f"  agents using any hm/LSP/skill: {sum(1 for o in A if o['hm'] or o['lsp'] or o['skill'])}/{len(A)}")
def cmd_report(a):
    rows, meta = load(a.since or AGENT_SINCE)
    if not a.jit_only: agent_ab(rows, meta, a.since or AGENT_SINCE)
    jit_ab(rows, meta, a.since)
    adoption(meta, a.since, rows)
