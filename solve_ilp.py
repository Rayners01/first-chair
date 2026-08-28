"""
solve_ilp.py — exact / gap-bounded solver for the optimal ski day.

Arc orienteering as a flow MILP, solved in one call:
    x_e integer  traversals of edge e        y_e binary  prize collected
    max  sum prize_e y_e
    s.t. y_e <= x_e; flow balance of x at every node (closed walk);
         sum x_e t_e <= budget; the tour leaves the start;
         single-commodity flow from the start to each collected edge's head,
         along arcs with x_e >= 1 — this is what kills disconnected subtours.

Speed: edges that cannot lie on any closed tour within the budget are pruned
first; --gap 0.02 stops once CBC has PROVEN the answer within 2% of optimal
(--gap 0 for exact ground truth); traversal caps start small and double
whenever one binds, so a cap can never silently truncate the optimum.

A result is reported only if CBC proves the requested gap, no cap binds, and
the extracted tour passes independent re-simulation.

    python solve_ilp.py graph.json --hours 3
    python solve_ilp.py graph.json --hours 6 --gap 0 --itinerary
"""

import argparse
import json
from collections import defaultdict, deque

import pulp

from solve_greedy import (load_graph, compute_prizes, score_itinerary,
                          choose_start, dijkstra, hhmm, describe)


def prune(edges, start, budget_s):
    """Drop edges no closed tour within the budget could use."""
    fwd, rev = defaultdict(list), defaultdict(list)
    for i, e in enumerate(edges):
        fwd[e["u"]].append(i)
        rev[e["v"]].append(i)
    out_t = dijkstra(start, edges, fwd)[0]
    back_t = dijkstra(start, edges, rev, backwards=True)[0]
    return [i for i, e in enumerate(edges)
            if out_t.get(e["u"]) is not None and back_t.get(e["v"]) is not None
            and out_t[e["u"]] + e["seconds"] + back_t[e["v"]] <= budget_s + 1e-6]


def build_and_solve(edges, keep, prizes, start, budget_s, cap, gap,
                    time_limit, verbose):
    n_prize = len(prizes)
    prob = pulp.LpProblem("ski_day", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x{i}", 0, cap, cat="Integer") for i in keep}
    y = {i: pulp.LpVariable(f"y{i}", 0, 1, cat="Binary") for i in prizes}
    f = {i: pulp.LpVariable(f"f{i}", 0, n_prize) for i in keep}

    prob += pulp.lpSum(prizes[i] * y[i] for i in prizes)

    outs, ins, demand = defaultdict(list), defaultdict(list), defaultdict(list)
    for i in keep:
        outs[edges[i]["u"]].append(i)
        ins[edges[i]["v"]].append(i)
    for i in prizes:
        demand[edges[i]["v"]].append(i)

    for i in prizes:
        prob += y[i] <= x[i]
    for i in keep:
        prob += f[i] <= n_prize * x[i]
    prob += pulp.lpSum(x[i] * edges[i]["seconds"] for i in keep) <= budget_s
    prob += pulp.lpSum(x[i] for i in outs[start]) >= 1

    for v in set(outs) | set(ins):
        prob += (pulp.lpSum(x[i] for i in ins[v])
                 == pulp.lpSum(x[i] for i in outs[v]))
        net = pulp.lpSum(f[i] for i in ins[v]) - pulp.lpSum(f[i] for i in outs[v])
        if v == start:
            # emit one unit per collected edge, less any consumed at the start
            prob += -net == (pulp.lpSum(y[i] for i in prizes)
                             - pulp.lpSum(y[i] for i in demand[start]))
        else:
            prob += net == pulp.lpSum(y[i] for i in demand[v])

    status = prob.solve(pulp.PULP_CBC_CMD(
        msg=1 if verbose else 0, timeLimit=time_limit,
        gapRel=gap if gap > 0 else None))
    if pulp.LpStatus[status] != "Optimal":
        return None, pulp.LpStatus[status]
    return ({i: int(round(x[i].value() or 0)) for i in keep},
            {i: (y[i].value() or 0.0) for i in prizes}), "Optimal"


def solve_escalating(edges, keep, prizes, start, budget_s, gap, time_limit,
                     verbose, cap=4, cap_max=32):
    while True:
        result, status = build_and_solve(edges, keep, prizes, start, budget_s,
                                         cap, gap, time_limit, verbose)
        if result is None:
            return None, status, cap
        binding = sum(1 for v in result[0].values() if v >= cap)
        if not binding:
            return result, status, cap
        if cap >= cap_max:
            return None, "CapLimit", cap
        print(f"  cap {cap} binds on {binding} edge(s); escalating to {cap * 2}")
        cap *= 2


def start_component(edges, xval, start):
    adj = defaultdict(set)
    for i, c in xval.items():
        if c > 0:
            adj[edges[i]["u"]].add(edges[i]["v"])
            adj[edges[i]["v"]].add(edges[i]["u"])
    comp, seen, queue = set(), {start}, deque([start])
    while queue:
        node = queue.popleft()
        comp.add(node)
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return comp


def hierholzer(edges, xval, start):
    """Eulerian circuit from integer traversal counts."""
    count = {i: c for i, c in xval.items() if c > 0}
    ready = defaultdict(list)
    for i in count:
        ready[edges[i]["u"]].append(i)

    circuit, stack = [], [(start, None)]
    while stack:
        node, via = stack[-1]
        outs = ready[node]
        while outs and count.get(outs[-1], 0) == 0:
            outs.pop()
        if outs:
            count[outs[-1]] -= 1
            stack.append((edges[outs[-1]]["v"], outs[-1]))
        else:
            stack.pop()
            if via is not None:
                circuit.append(via)
    return circuit[::-1], sum(count.values())


def audit(itinerary, edges, prizes, start, budget_s, claimed, stray):
    """Independent re-simulation of the extracted tour."""
    lines, ok, node, clock = [], True, start, 0.0
    for i in itinerary:
        if edges[i]["u"] != node:
            lines.append(f"FAIL discontinuity at edge {i}")
            ok = False
            break
        node, clock = edges[i]["v"], clock + edges[i]["seconds"]
    if node != start:
        lines.append("FAIL does not return to start")
        ok = False
    if clock > budget_s + 1e-6:
        lines.append(f"FAIL over budget: {clock:.0f}s > {budget_s:.0f}s")
        ok = False
    simulated = score_itinerary(itinerary, prizes)
    if abs(simulated - claimed) > 1e-6:
        lines.append(f"FAIL simulated {simulated:.6f} != claimed {claimed:.6f}")
        ok = False
    if stray:
        lines.append(f"note: {stray} stray flow unit(s) discarded (prize-free)")
    lines.append("VERIFIED: tour matches claimed objective and budget" if ok
                 else "VERIFICATION FAILED — do not use this result")
    return ok, clock, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--start-node", type=int)
    ap.add_argument("--start-name")
    ap.add_argument("--gap", type=float, default=0.02,
                    help="0 proves exact optimality")
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--itinerary", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    nodes, edges, fwd, rev, core = load_graph(args.graph)
    start = choose_start(nodes, edges, args)
    budget = args.hours * 3600
    all_prizes = compute_prizes(edges, args.lam)
    keep = prune(edges, start, budget)
    prizes = {i: all_prizes[i] for i in keep if i in all_prizes}

    print(f"instance   {len(keep)}/{len(edges)} edges after pruning, "
          f"{len(prizes)} prize edges")
    print(f"settings   lam={args.lam} budget={args.hours}h start={start} "
          f"gap={args.gap:.0%} limit={args.time_limit}s")

    result, status, cap = solve_escalating(edges, keep, prizes, start, budget,
                                           args.gap, args.time_limit,
                                           args.verbose)
    if result is None:
        print(f"\nNO RESULT ({status}). Raise --time-limit or --gap, or "
              f"shorten --hours. Nothing unproven is ever reported.")
        return

    xval, yval = result
    claimed = sum(prizes[i] for i in prizes if yval[i] > 0.5)
    comp = start_component(edges, xval, start)
    tour = {i: c for i, c in xval.items() if c > 0 and edges[i]["u"] in comp}
    stray = sum(c for i, c in xval.items()
                if c > 0 and edges[i]["u"] not in comp)

    itinerary, leftover = hierholzer(edges, tour, start)
    ok, clock, lines = audit(itinerary, edges, prizes, start, budget, claimed,
                             stray + leftover)

    guarantee = ("proven optimal" if args.gap == 0
                 else f"proven within {args.gap:.0%} of optimal")
    print(f"\nobjective  {claimed:.4f}  ({guarantee}, cap {cap})")
    print("\n".join(lines) + "\n")

    if args.itinerary:
        describe(itinerary, edges, clock, budget, prizes)
    else:
        unique = {i for i in itinerary if i in prizes}
        print(f"unique vertical {sum(edges[i]['vertical_m'] for i in unique):,.0f} m"
              f" | unique piste {sum(edges[i]['length_m'] for i in unique) / 1000:.1f} km"
              f" | {len(itinerary)} legs | home {hhmm(clock)}")

    if ok and args.out:
        with open(args.out, "w") as fh:
            json.dump({"start_node": start, "budget_s": budget,
                       "objective": claimed, "lam": args.lam,
                       "gap_guarantee": args.gap,
                       "edges": [edges[i] for i in itinerary]}, fh)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()