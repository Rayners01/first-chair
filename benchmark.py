"""
benchmark.py — run the solvers across a grid and write a report.

For every (resort, budget, lambda) cell it runs the greedy baseline, the exact
ILP (recorded only when CBC *proves* optimality within the time limit), and the
annealer over several seeds. Every result is re-simulated with the shared
objective before it counts, so a solver bug shows up as a failed row rather
than a good-looking number.

Results append to results.csv keyed by cell, so a long run can be interrupted
and resumed; report.md is regenerated from the CSV each time.

    python benchmark.py --graphs graphs/*.json
    python benchmark.py --graphs graphs/tignes.json --hours 1 2 3 --seeds 5
    python benchmark.py --quick                 # smoke test, one cell
    python benchmark.py --report-only           # rebuild report.md from CSV
"""

import argparse
import csv
import glob
import statistics
import time
from pathlib import Path

from solve_greedy import (load_graph, compute_prizes, score_itinerary, greedy)
from solve_anneal import all_dists, solve as anneal_solve
from solve_ilp import (prune, solve_escalating, start_component, hierholzer,
                       audit)

FIELDS = ["resort", "hours", "lam", "solver", "seed", "objective", "seconds",
          "status", "vertical_m", "piste_km", "coverage_pct", "edges",
          "prize_edges", "verified"]


# ---------------------------------------------------------------- helpers

def derived(itinerary, edges, prizes):
    """Vertical, piste and coverage of a realised itinerary."""
    unique = {i for i in itinerary if i in prizes}
    v_tot = sum(edges[i]["vertical_m"] for i in prizes) or 1.0
    vert = sum(edges[i]["vertical_m"] for i in unique)
    return {"vertical_m": round(vert),
            "piste_km": round(sum(edges[i]["length_m"] for i in unique) / 1000, 1),
            "coverage_pct": round(100 * vert / v_tot, 1)}


def run_greedy(ctx, budget, prizes):
    t0 = time.time()
    itin, _, _ = greedy(ctx["edges"], ctx["fwd"], ctx["rev"], ctx["start"],
                        budget, prizes, return_to_start=True)
    secs = time.time() - t0
    ok = audit(itin, ctx["edges"], prizes, ctx["start"], budget,
               score_itinerary(itin, prizes), 0)[0]
    return [{"solver": "greedy", "seed": "", "status": "ok",
             "objective": round(score_itinerary(itin, prizes), 6),
             "seconds": round(secs, 2), "verified": int(ok),
             **derived(itin, ctx["edges"], prizes)}]


def run_anneal(ctx, budget, prizes, seeds, seconds, restarts):
    rows = []
    for seed in range(seeds):
        t0 = time.time()
        itin, val, _ = anneal_solve(ctx["edges"], ctx["fwd"], ctx["rev"], prizes,
                                    ctx["D"], ctx["start"], budget, seconds,
                                    restarts, seed=seed, closed=True)
        secs = time.time() - t0
        ok = audit(itin, ctx["edges"], prizes, ctx["start"], budget, val, 0)[0]
        rows.append({"solver": "anneal", "seed": seed, "status": "ok",
                     "objective": round(val, 6), "seconds": round(secs, 2),
                     "verified": int(ok),
                     **derived(itin, ctx["edges"], prizes)})
    return rows


def run_ilp(ctx, budget, prizes_all, time_limit):
    """Exact solve. Only reported when CBC proves optimality AND it verifies."""
    edges, start = ctx["edges"], ctx["start"]
    t0 = time.time()
    keep = prune(edges, start, budget)
    prizes = {i: prizes_all[i] for i in keep if i in prizes_all}
    if not prizes:
        return [{"solver": "ilp", "seed": "", "status": "no prizes",
                 "objective": "", "seconds": 0, "verified": 0,
                 "vertical_m": "", "piste_km": "", "coverage_pct": "",
                 "edges": len(keep), "prize_edges": 0}]

    result, status, cap = solve_escalating(edges, keep, prizes, start, budget,
                                           0.0, time_limit, False)
    secs = time.time() - t0
    base = {"solver": "ilp", "seed": "", "seconds": round(secs, 2),
            "edges": len(keep), "prize_edges": len(prizes)}
    if result is None:
        return [{**base, "status": status, "objective": "", "verified": 0,
                 "vertical_m": "", "piste_km": "", "coverage_pct": ""}]

    xval, yval = result
    claimed = sum(prizes[i] for i in prizes if yval[i] > 0.5)
    comp = start_component(edges, xval, start)
    tour = {i: c for i, c in xval.items() if c > 0 and edges[i]["u"] in comp}
    stray = sum(c for i, c in xval.items()
                if c > 0 and edges[i]["u"] not in comp)
    itin, leftover = hierholzer(edges, tour, start)
    ok = audit(itin, edges, prizes, start, budget, claimed, stray + leftover)[0]
    return [{**base, "status": "proven" if ok else "verify failed",
             "objective": round(claimed, 6), "verified": int(ok),
             **derived(itin, edges, prizes)}]


# ---------------------------------------------------------------- driver

def load_context(path):
    nodes, edges, fwd, rev, core = load_graph(path)
    if not edges:
        raise SystemExit(f"{path}: empty core")
    used = {e["u"] for e in edges} | {e["v"] for e in edges}
    lifts = [e for e in edges if e["type"] == "lift"]
    start = (max(lifts, key=lambda e: e["length_m"])["u"] if lifts
             else edges[0]["u"])
    print(f"  loading {Path(path).stem}: {len(edges)} edges, "
          f"precomputing distances\u2026", flush=True)
    return {"edges": edges, "fwd": fwd, "rev": rev, "start": start,
            "D": all_dists(edges, fwd, used),
            "name": Path(path).stem}


def read_done(csv_path):
    if not csv_path.exists():
        return set(), []
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    done = {(r["resort"], r["hours"], r["lam"], r["solver"], r["seed"])
            for r in rows}
    return done, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=["graphs/*.json"])
    ap.add_argument("--hours", nargs="+", type=float, default=[1, 3, 6])
    ap.add_argument("--lam", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--anneal-seconds", type=float, default=8.0)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--ilp-time-limit", type=int, default=300)
    ap.add_argument("--no-ilp", action="store_true")
    ap.add_argument("--out", default="bench")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.hours, args.lam, args.seeds = [3], [0.5], 2
        args.anneal_seconds, args.ilp_time_limit = 3.0, 60

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    done, rows = read_done(csv_path)

    if not args.report_only:
        paths = sorted({p for pat in args.graphs for p in glob.glob(pat)})
        if not paths:
            raise SystemExit(f"no graphs matched {args.graphs}")

        fh = open(csv_path, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not done:
            writer.writeheader()

        total = len(paths) * len(args.hours) * len(args.lam)
        cell = 0
        for path in paths:
            ctx = None
            for hours in args.hours:
                for lam in args.lam:
                    cell += 1
                    name = Path(path).stem
                    key = (name, str(hours), str(lam))
                    plan = [("greedy", [""]), ("anneal", list(range(args.seeds)))]
                    if not args.no_ilp:
                        plan.append(("ilp", [""]))
                    if all((*key, s, str(sd)) in done
                           for s, seeds in plan for sd in seeds):
                        print(f"[{cell}/{total}] {name} {hours}h lam={lam} "
                              f"\u2014 already done", flush=True)
                        continue

                    if ctx is None:
                        ctx = load_context(path)
                    budget = hours * 3600
                    prizes = compute_prizes(ctx["edges"], lam)
                    print(f"[{cell}/{total}] {name} {hours}h lam={lam}",
                          flush=True)

                    new = run_greedy(ctx, budget, prizes)
                    new += run_anneal(ctx, budget, prizes, args.seeds,
                                      args.anneal_seconds, args.restarts)
                    if not args.no_ilp:
                        new += run_ilp(ctx, budget, prizes, args.ilp_time_limit)

                    for r in new:
                        r.setdefault("edges", len(ctx["edges"]))
                        r.setdefault("prize_edges", len(prizes))
                        row = {"resort": name, "hours": hours, "lam": lam, **r}
                        if (name, str(hours), str(lam), row["solver"],
                                str(row["seed"])) in done:
                            continue
                        writer.writerow({k: row.get(k, "") for k in FIELDS})
                        rows.append({k: str(row.get(k, "")) for k in FIELDS})
                    fh.flush()
                    for r in new:
                        gap = ""
                        print(f"    {r['solver']:<7}{str(r['seed']):<3}"
                              f"{r['status']:<14}{r['objective']}{gap}")
        fh.close()

    write_report(rows, out / "report.md")
    print(f"\nwrote {csv_path} and {out / 'report.md'}")


# ---------------------------------------------------------------- report

def fnum(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def write_report(rows, path):
    if not rows:
        path.write_text("# Benchmarks\n\nNo results yet.\n")
        return

    cells = {}
    for r in rows:
        key = (r["resort"], fnum(r["hours"]), fnum(r["lam"]))
        cells.setdefault(key, []).append(r)

    lines = ["# Benchmarks", "",
             "Generated by `benchmark.py`. Every row was re-simulated against "
             "the shared objective before being recorded; ILP results appear "
             "only where CBC proved optimality.", ""]

    failed = [r for r in rows if r["objective"] and r["verified"] == "0"]
    if failed:
        lines += [f"> **{len(failed)} result(s) failed verification** — "
                  "investigate before trusting this report.", ""]

    # ---- headline: solver quality against proven optimum
    lines += ["## Solver quality", "",
              "Annealer is the mean over seeds; gap is measured against the "
              "proven optimum where one exists.", "",
              "| Resort | Budget | λ | Greedy | Anneal (mean) | Anneal (best) "
              "| Optimum | Greedy gap | Anneal gap |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (resort, hours, lam), rs in sorted(cells.items()):
        g = next((fnum(r["objective"]) for r in rs if r["solver"] == "greedy"), None)
        anneals = [fnum(r["objective"]) for r in rs
                   if r["solver"] == "anneal" and r["objective"]]
        opt = next((fnum(r["objective"]) for r in rs
                    if r["solver"] == "ilp" and r["status"] == "proven"), None)
        mean = statistics.fmean(anneals) if anneals else None
        best = max(anneals) if anneals else None
        gap = lambda v: f"{(1 - v / opt) * 100:.1f}%" if opt and v else "—"
        lines.append(
            f"| {resort} | {hours:g}h | {lam:g} | "
            f"{g:.4f} | {mean:.4f} | {best:.4f} | "
            f"{f'{opt:.4f}' if opt else '—'} | {gap(g)} | {gap(mean)} |"
            if g and mean else
            f"| {resort} | {hours:g}h | {lam:g} | … | … | … | … | … | … |")

    # ---- annealer improvement over the baseline
    lines += ["", "## Annealer vs greedy", "",
              "| Resort | Budget | λ | Improvement | Seed spread |",
              "| --- | --- | --- | --- | --- |"]
    for (resort, hours, lam), rs in sorted(cells.items()):
        g = next((fnum(r["objective"]) for r in rs if r["solver"] == "greedy"), None)
        a = [fnum(r["objective"]) for r in rs
             if r["solver"] == "anneal" and r["objective"]]
        if not g or not a:
            continue
        spread = f"{(max(a) - min(a)) / statistics.fmean(a) * 100:.1f}%" \
            if len(a) > 1 else "—"
        lines.append(f"| {resort} | {hours:g}h | {lam:g} | "
                     f"{(statistics.fmean(a) / g - 1) * 100:+.1f}% | {spread} |")

    # ---- tractability of the exact solver
    lines += ["", "## Exact solver tractability", "",
              "Where the ILP stops being usable — the reason the annealer "
              "exists.", "",
              "| Resort | Budget | λ | Edges (pruned) | Prize edges | Status "
              "| Time |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for (resort, hours, lam), rs in sorted(cells.items()):
        r = next((r for r in rs if r["solver"] == "ilp"), None)
        if not r:
            continue
        lines.append(f"| {resort} | {hours:g}h | {lam:g} | {r['edges']} | "
                     f"{r['prize_edges']} | {r['status']} | "
                     f"{fnum(r['seconds'], 0):.0f}s |")

    # ---- what lambda actually trades
    lines += ["", "## What λ trades", "",
              "Annealer means. λ=0 maximises piste covered, λ=1 vertical "
              "descended.", "",
              "| Resort | Budget | λ | Vertical (m) | Piste (km) | Coverage |",
              "| --- | --- | --- | --- | --- | --- |"]
    for (resort, hours, lam), rs in sorted(cells.items()):
        a = [r for r in rs if r["solver"] == "anneal" and r["vertical_m"]]
        if not a:
            continue
        mean = lambda k: statistics.fmean(fnum(r[k], 0) for r in a)
        lines.append(f"| {resort} | {hours:g}h | {lam:g} | "
                     f"{mean('vertical_m'):,.0f} | {mean('piste_km'):.1f} | "
                     f"{mean('coverage_pct'):.0f}% |")

    # ---- runtimes
    lines += ["", "## Runtimes", "",
              "| Solver | Median | Max |", "| --- | --- | --- |"]
    for solver in ("greedy", "anneal", "ilp"):
        secs = [fnum(r["seconds"], 0) for r in rows if r["solver"] == solver]
        if secs:
            lines.append(f"| {solver} | {statistics.median(secs):.1f}s | "
                         f"{max(secs):.1f}s |")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()