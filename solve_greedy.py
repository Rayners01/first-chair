"""
solve_greedy.py — shared objective, graph helpers, and the greedy baseline.

Objective (used by every solver): each run edge carries a prize collected on
FIRST descent only,
    prize(e) = lam * vert_e / V_total + (1 - lam) * len_e / L_total
Re-skiing is allowed as transit but scores nothing. Lifts/walks/skates carry no
prize. lam=1 maximises unique vertical, lam=0 unique piste length. Coverage is
measured by length, not run count, so the score can't be inflated by how finely
the graph builder happened to split a way.

Greedy repeatedly takes the shortest path to the best uncollected run by prize
per second: deliberately myopic, and the baseline the ILP must beat.

    python solve_greedy.py graph.json --hours 6.5 --return-to-start
"""

import argparse
import heapq
import json
import math
from collections import defaultdict


def compute_prizes(edges, lam):
    runs = [i for i, e in enumerate(edges) if e["type"] == "run"]
    v_tot = sum(edges[i]["vertical_m"] for i in runs) or 1.0
    l_tot = sum(edges[i]["length_m"] for i in runs) or 1.0
    return {i: lam * edges[i]["vertical_m"] / v_tot
            + (1 - lam) * edges[i]["length_m"] / l_tot for i in runs}


def score_itinerary(itinerary, prizes):
    return sum(prizes[i] for i in set(itinerary) if i in prizes)


def load_graph(path):
    with open(path) as fh:
        g = json.load(fh)
    core = set(g["core"])
    edges = [e for e in g["edges"] if e["u"] in core and e["v"] in core]
    fwd, rev = defaultdict(list), defaultdict(list)
    for i, e in enumerate(edges):
        fwd[e["u"]].append(i)
        rev[e["v"]].append(i)
    return g["nodes"], edges, fwd, rev, core


def dijkstra(source, edges, adj, backwards=False):
    """Shortest times from source, or to source when backwards."""
    dist, prev, heap = {source: 0.0}, {}, [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, math.inf):
            continue
        for ei in adj[node]:
            nxt = edges[ei]["u"] if backwards else edges[ei]["v"]
            nd = d + edges[ei]["seconds"]
            if nd < dist.get(nxt, math.inf):
                dist[nxt], prev[nxt] = nd, ei
                heapq.heappush(heap, (nd, nxt))
    return dist, prev


def extract_path(target, source, prev, edges, backwards=False):
    path, node = [], target
    while node != source:
        ei = prev.get(node)
        if ei is None:
            return None
        path.append(ei)
        node = edges[ei]["v"] if backwards else edges[ei]["u"]
    return path if backwards else path[::-1]


def choose_start(nodes, edges, args):
    if args.start_node is not None:
        return args.start_node
    if args.start_name:
        name = args.start_name.lower()
        for e in edges:
            if e["type"] == "lift" and name in (e.get("name") or "").lower():
                return e["u"]
        raise SystemExit(f"no lift matching {args.start_name!r} in the core")
    lifts = [e for e in edges if e["type"] == "lift"]
    return max(lifts, key=lambda e: e["length_m"])["u"] if lifts else edges[0]["u"]


def greedy(edges, fwd, rev, start, budget_s, prizes, return_to_start=False):
    time_back = dijkstra(start, edges, rev, backwards=True)[0] \
        if return_to_start else {}
    node, clock, itinerary, collected = start, 0.0, [], set()

    while True:
        dist, prev = dijkstra(node, edges, fwd)
        best, best_score = None, 0.0
        for ei, prize in prizes.items():
            if ei in collected:
                continue
            reach = dist.get(edges[ei]["u"])
            if reach is None:
                continue
            total = reach + edges[ei]["seconds"]
            if clock + total > budget_s:
                continue
            if return_to_start:
                back = time_back.get(edges[ei]["v"])
                if back is None or clock + total + back > budget_s:
                    continue
            score = prize / max(total, 1.0)
            if score > best_score:
                best_score, best = score, ei
        if best is None:
            break

        for step in extract_path(edges[best]["u"], node, prev, edges) + [best]:
            itinerary.append(step)
            clock += edges[step]["seconds"]
        collected.add(best)
        node = edges[best]["v"]

    if return_to_start and node != start:
        prev_b = dijkstra(start, edges, rev, backwards=True)[1]
        home = extract_path(node, start, prev_b, edges, backwards=True)
        if home:
            for step in home:
                itinerary.append(step)
                clock += edges[step]["seconds"]
            node = start
    return itinerary, clock, node


# ---------------------------------------------------------------- output

def hhmm(seconds, day_start=9 * 3600):
    t = int(day_start + seconds)
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}"


def describe(itinerary, edges, clock, budget_s, prizes=None):
    ridden = [edges[i] for i in itinerary]
    unique = {i for i in itinerary if edges[i]["type"] == "run"}

    print(f"day plan: {hhmm(0)} -> {hhmm(clock)} "
          f"(budget {hhmm(budget_s)}, used {clock / budget_s:.0%})")
    if prizes:
        v_tot = sum(edges[i]["vertical_m"] for i in prizes) or 1.0
        l_tot = sum(edges[i]["length_m"] for i in prizes) or 1.0
        uv = sum(edges[i]["vertical_m"] for i in unique)
        ul = sum(edges[i]["length_m"] for i in unique)
        print(f"objective score     {score_itinerary(itinerary, prizes):.4f} "
              f"(of 1.0 for the whole resort)")
        print(f"unique vertical     {uv:,.0f} m ({uv / v_tot:.0%} of resort)")
        print(f"unique piste dist   {ul / 1000:,.1f} km ({ul / l_tot:.0%} of resort)")
    print(f"vertical skied      "
          f"{sum(e['vertical_m'] for e in ridden if e['type'] == 'run'):,.0f} m")
    print(f"distance covered    {sum(e['length_m'] for e in ridden) / 1000:,.1f} km")
    print(f"run descents        {sum(1 for e in ridden if e['type'] == 'run')} "
          f"({len(unique)} unique segments)")
    print(f"lift rides          {sum(1 for e in ridden if e['type'] == 'lift')}\n")

    icons = {"run": "v", "lift": "^", "walk": "w", "skate": "~"}
    t, group = 0.0, None
    for e in ridden:
        key = (e["type"], e.get("name") or f"({e['type']})")
        if group and group[0] == key:            # collapse consecutive segments
            group[2] += e["seconds"]
            group[3] += e["vertical_m"] if e["type"] == "run" else 0
        else:
            if group:
                print_leg(icons, *group)
            group = [key, t, e["seconds"],
                     e["vertical_m"] if e["type"] == "run" else 0]
        t += e["seconds"]
    if group:
        print_leg(icons, *group)


def print_leg(icons, key, start_s, dur_s, vert):
    kind, label = key
    extra = f"  -{vert:.0f} m" if kind == "run" and vert else ""
    print(f"  {hhmm(start_s)}  {icons[kind]} {label:<38s} "
          f"{dur_s / 60:4.1f} min{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--hours", type=float, default=6.5)
    ap.add_argument("--lam", type=float, default=0.5,
                    help="1=vertical, 0=coverage")
    ap.add_argument("--start-node", type=int)
    ap.add_argument("--start-name", help="start at this lift's bottom station")
    ap.add_argument("--return-to-start", action="store_true")
    ap.add_argument("--out", help="write the route as JSON for the UI")
    args = ap.parse_args()

    nodes, edges, fwd, rev, core = load_graph(args.graph)
    if not edges:
        raise SystemExit("empty core — rebuild the graph first")
    start = choose_start(nodes, edges, args)
    budget = args.hours * 3600
    prizes = compute_prizes(edges, args.lam)

    print(f"objective: lam={args.lam}  start node {start}\n")
    itinerary, clock, end = greedy(edges, fwd, rev, start, budget, prizes,
                                   args.return_to_start)
    describe(itinerary, edges, clock, budget, prizes)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"start_node": start, "end_node": end,
                       "budget_s": budget, "used_s": clock,
                       "edges": [edges[i] for i in itinerary]}, fh)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()