"""
solve_anneal.py — simulated annealing for the optimal ski day.

Same objective as every other solver (solve_greedy.compute_prizes). A solution
is an ordered list of prize edges to collect; the tour is realised by shortest
paths between them and back to the start. Moves add, drop, relocate or swap
targets; infeasible (over-budget) proposals are rejected. Initialised from the
greedy solution, so it can only match or beat the baseline.

The reported score always comes from auditing the REALISED itinerary — prizes
picked up incidentally while transiting between targets count, and the audit
(continuity, budget, re-scoring) is the same one the ILP must pass.

    python solve_anneal.py graph.json --hours 6 --seconds 10
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict

from solve_greedy import (load_graph, compute_prizes, score_itinerary,
                          choose_start, dijkstra, extract_path, greedy,
                          describe, hhmm)
from solve_ilp import audit


def all_dists(edges, fwd, nodes_used):
    return {n: dijkstra(n, edges, fwd)[0] for n in nodes_used}


def tour_time(targets, D, edges, start, closed=True):
    """Time of the realised tour, or None if disconnected."""
    t, cur = 0.0, start
    for ei in targets:
        d = D[cur].get(edges[ei]["u"])
        if d is None:
            return None
        t += d + edges[ei]["seconds"]
        cur = edges[ei]["v"]
    if not closed:
        return t
    back = D[cur].get(start)
    return None if back is None else t + back


def realise(targets, edges, fwd, start, closed=True):
    """Expand the target order into a full edge itinerary."""
    itinerary, cur = [], start
    for ei in targets:
        _, prev = dijkstra(cur, edges, fwd)
        leg = extract_path(edges[ei]["u"], cur, prev, edges)
        if leg is None:
            return None
        itinerary += leg + [ei]
        cur = edges[ei]["v"]
    if not closed:
        return itinerary
    _, prev = dijkstra(cur, edges, fwd)
    home = extract_path(start, cur, prev, edges)
    if home is None:
        return None
    return itinerary + home


ELITE = 6   # snapshots kept per run, ranked later by REALISED score


def anneal(edges, fwd, prizes, D, start, budget_s, init, seconds, rng,
           closed=True):
    """Add/drop/relocate targets with O(1) incremental time deltas.

    Returns several elite target lists: the search ranks them by target-prize
    sum, but the true score comes from the realised itinerary (which also
    collects runs incidentally in transit), so the caller re-ranks.
    """
    inf = math.inf

    def d(a, b):
        return D[a].get(b, inf)

    def leg(prev_node, ei):
        return d(prev_node, edges[ei]["u"]) + edges[ei]["seconds"]

    def exit_of(k):
        return start if k < 0 else edges[targets[k]]["v"]

    def home_of(node):
        return d(node, start) if closed else 0.0

    def entry_after(k):
        """Node the tour heads to after position k (next target's u, or home)."""
        return edges[targets[k + 1]]["u"] if k + 1 < len(targets) else start

    targets = [ei for ei in init if ei in prizes]
    total = tour_time(targets, D, edges, start, closed)
    if total is None:
        targets, total = [], 0.0
    score = sum(prizes[i] for i in targets)
    elite = [(score, list(targets))]
    in_set = set(targets)

    avail = [ei for ei in prizes if ei not in in_set]
    where = {ei: k for k, ei in enumerate(avail)}

    def take(k):
        """Remove avail[k], keeping the index map consistent."""
        ei = avail[k]
        last = avail.pop()
        if k < len(avail):
            avail[k] = last
            where[last] = k
        del where[ei]
        return ei

    def give_back(ei):
        where[ei] = len(avail)
        avail.append(ei)

    mean_prize = sum(prizes.values()) / len(prizes)
    T0, deadline = 3.0 * mean_prize, time.time() + seconds

    def insert_delta(k, ei):
        prev = exit_of(k - 1)
        if k < len(targets):
            nxt = edges[targets[k]]["u"]
            return leg(prev, ei) + d(edges[ei]["v"], nxt) - d(prev, nxt)
        return leg(prev, ei) + home_of(edges[ei]["v"]) - home_of(prev)

    def drop_delta(k):
        prev = exit_of(k - 1)
        if k + 1 < len(targets):
            nxt = edges[targets[k + 1]]["u"]
            return d(prev, nxt) - leg(prev, targets[k]) \
                - d(edges[targets[k]]["v"], nxt)
        return home_of(prev) - leg(prev, targets[k]) \
            - home_of(edges[targets[k]]["v"])

    while time.time() < deadline:
        frac = max(1e-9, (deadline - time.time()) / seconds)
        T = max(T0 * frac * frac, 1e-12)
        r = rng.random()

        if r < 0.5 or not targets:                                   # add
            # Sample only from runs not already in the tour: once most of
            # the resort is selected, sampling the whole pool wastes nearly
            # every iteration on rejected duplicates.
            if not avail:
                continue
            ai = rng.randrange(len(avail))
            ei = avail[ai]
            spots = range(len(targets) + 1) if len(targets) < 14 else \
                [rng.randint(0, len(targets)) for _ in range(14)]
            k, dt = min(((k, insert_delta(k, ei)) for k in spots),
                        key=lambda p: p[1])
            if dt == inf or total + dt > budget_s:
                continue
            take(ai)
            targets.insert(k, ei)
            in_set.add(ei)
            total += dt
            score += prizes[ei]
        elif r < 0.75:                                               # drop
            k = rng.randrange(len(targets))
            dp = -prizes[targets[k]]
            if rng.random() >= math.exp(dp / T):
                continue
            total += drop_delta(k)
            in_set.discard(targets[k])
            give_back(targets[k])
            score += dp
            targets.pop(k)
        else:                                                        # relocate
            k = rng.randrange(len(targets))
            dt1 = drop_delta(k)
            ei = targets.pop(k)
            spots = range(len(targets) + 1) if len(targets) < 14 else \
                [rng.randint(0, len(targets)) for _ in range(14)]
            j, dt2 = min(((j, insert_delta(j, ei)) for j in spots),
                         key=lambda p: p[1])
            if dt2 == inf or total + dt1 + dt2 > budget_s or \
                    (dt1 + dt2 > 0 and rng.random() >=
                     math.exp(-(dt1 + dt2) / max(budget_s * frac * 0.01, 1.0))):
                targets.insert(k, ei)                                # revert
                continue
            targets.insert(j, ei)
            total += dt1 + dt2

        if score > elite[-1][0] or len(elite) < ELITE:
            elite.append((score, list(targets)))
            elite.sort(key=lambda p: -p[0])
            del elite[ELITE:]
    return [t for _, t in elite]


def solve(edges, fwd, rev, prizes, D, start, budget_s, seconds, restarts=3,
          seed=0, closed=True):
    """Full pipeline: greedy init, annealing restarts, pick best REALISED."""
    g_itin, _, _ = greedy(edges, fwd, rev, start, budget_s, prizes,
                          return_to_start=closed)
    init = [i for i in dict.fromkeys(g_itin) if i in prizes]

    best, best_val = g_itin, score_itinerary(g_itin, prizes)
    for r in range(restarts):
        for targets in anneal(edges, fwd, prizes, D, start, budget_s, init,
                              seconds / restarts,
                              random.Random(seed * 1000 + r), closed):
            itin = realise(targets, edges, fwd, start, closed)
            if itin is None:
                continue
            val = score_itinerary(itin, prizes)
            if val > best_val:
                best, best_val = itin, val
    return best, best_val, g_itin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--start-node", type=int)
    ap.add_argument("--start-name")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="total annealing time budget")
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--itinerary", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    nodes, edges, fwd, rev, core = load_graph(args.graph)
    start = choose_start(nodes, edges, args)
    budget = args.hours * 3600
    prizes = compute_prizes(edges, args.lam)
    rng = random.Random(args.seed)

    nodes_used = {e["u"] for e in edges} | {e["v"] for e in edges}
    D = all_dists(edges, fwd, nodes_used)

    itinerary, final, greedy_itin = solve(
        edges, fwd, rev, prizes, D, start, budget, args.seconds,
        args.restarts, args.seed)
    greedy_score = score_itinerary(greedy_itin, prizes)
    ok, clock, lines = audit(itinerary, edges, prizes, start, budget,
                             score_itinerary(itinerary, prizes), 0)

    print(f"annealed   {final:.4f}   (greedy {greedy_score:.4f}, "
          f"{(final - greedy_score) / max(greedy_score, 1e-9):+.1%} vs greedy)")
    print("\n".join(lines) + "\n")
    if args.itinerary:
        describe(itinerary, edges, clock, budget, prizes)
    else:
        unique = {i for i in itinerary if i in prizes}
        print(f"unique vertical {sum(edges[i]['vertical_m'] for i in unique):,.0f} m"
              f" | unique piste {sum(edges[i]['length_m'] for i in unique) / 1000:.1f} km"
              f" | home {hhmm(clock)}")

    if ok and args.out:
        with open(args.out, "w") as fh:
            json.dump({"start_node": start, "budget_s": budget,
                       "objective": final, "lam": args.lam,
                       "edges": [edges[i] for i in itinerary]}, fh)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()