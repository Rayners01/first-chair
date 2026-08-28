"""
build_graph.py — build a routable ski graph from OpenSkiMap runs/lifts GeoJSON.

Runs are split at run-run shared vertices; lifts are never split (a lift
crossing a piste is not a station). Endpoints are hub-snapped, then any node
that dangles or is touched only by lifts is projected onto the nearest run and
joined with a walk edge. Gondolas/cable cars/funiculars are bidirectional;
flat pieces become skate edges; gentle pieces are walkable in reverse.

    python build_graph.py runs.geojson lifts.geojson --area "Val d'Isere|Tignes"
    python build_graph.py runs.geojson lifts.geojson --area-id <skiAreaId>
    python build_graph.py runs.geojson lifts.geojson --catalog      # list areas
"""

import argparse
import unicodedata
import json
import math
from collections import defaultdict, deque
from pathlib import Path

EARTH_R = 6371000.0
RUN_USES = {"downhill", "connection"}
WALK_SPEED, SKATE_SPEED = 1.0, 2.5
FLAT_DROP_M, GENTLE_DROP_M = 0.5, 3.0
BIDIRECTIONAL_LIFTS = {"gondola", "cable_car", "funicular", "mixed_lift"}
LIFT_RULES = defaultdict(int)   # how each lift's ride time was determined

LIFT_SPEED = {"gondola": 6.0, "chair_lift": 2.6, "chair_lift_detachable": 5.0,
              "cable_car": 10.0, "t-bar": 2.5, "j-bar": 2.5, "platter": 2.8,
              "drag_lift": 2.5, "magic_carpet": 0.8, "rope_tow": 2.0,
              "funicular": 8.0}
SKI_SPEED = {"novice": 4.0, "easy": 6.0, "intermediate": 8.0,
             "advanced": 9.0, "expert": 8.0, "freeride": 6.0}
DEFAULT_LIFT_SPEED, DEFAULT_SKI_SPEED, BOARD_TIME = 3.5, 7.0, 60.0

# Sanity window for a trusted `duration`: outside this the value is wrong or in
# the wrong unit, so fall through to the speed model rather than believe it.
MIN_CABLE_SPEED, MAX_CABLE_SPEED = 0.8, 14.0
DETACHABLE_OCCUPANCY = 6      # 6-seaters and up are detachable in practice
DETACHABLE_CAPACITY = 2400    # persons/hour a fixed grip cannot sustain


def lift_seconds(props, length):
    """
    Ride time in seconds, plus the rule used. Preference order:

      1. `duration` from OSM — the actual measured ride time, no model at all;
      2. explicit `detachable`;
      3. `occupancy` / `capacity` / `bubble`, which imply a detachable chair
         (a fixed grip cannot carry 6 across, sustain 2400 p/h, or bubble);
      4. the per-type speed table.

    OpenSkiMap leaves `detachable` null on most lifts, so relying on it alone
    makes every high-speed chair ~1.7x too slow.
    """
    kind = (props.get("liftType") or props.get("aerialway") or "").lower()

    duration = props.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        speed = length / duration
        if MIN_CABLE_SPEED <= speed <= MAX_CABLE_SPEED:
            return duration + BOARD_TIME, "duration"

    if kind == "chair_lift":
        if props.get("detachable"):
            return length / LIFT_SPEED["chair_lift_detachable"] + BOARD_TIME, \
                "detachable tag"
        occ, cap = props.get("occupancy"), props.get("capacity")
        if (isinstance(occ, (int, float)) and occ >= DETACHABLE_OCCUPANCY) \
                or (isinstance(cap, (int, float)) and cap >= DETACHABLE_CAPACITY) \
                or props.get("bubble") or props.get("heating"):
            return length / LIFT_SPEED["chair_lift_detachable"] + BOARD_TIME, \
                "inferred detachable"

    return length / LIFT_SPEED.get(kind, DEFAULT_LIFT_SPEED) + BOARD_TIME, \
        "speed model"


# ---------------------------------------------------------------- geometry

def haversine(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R * math.asin(math.sqrt(h))


def ele(pt):
    return pt[2] if len(pt) > 2 else 0.0


def slope_length(coords):
    return sum(math.hypot(haversine(p, q), ele(q) - ele(p))
               for p, q in zip(coords, coords[1:]))


def lerp(a, b, t):
    return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]),
            ele(a) + t * (ele(b) - ele(a))]


def project_onto_polyline(pt, coords):
    """Nearest point on a polyline: (distance_m, arc_position_m, point)."""
    kx = 111320.0 * math.cos(math.radians(pt[1]))
    best, arc = (math.inf, 0.0, None), 0.0
    for a, b in zip(coords, coords[1:]):
        ax, ay = (a[0] - pt[0]) * kx, (a[1] - pt[1]) * 111320.0
        bx, by = (b[0] - pt[0]) * kx, (b[1] - pt[1]) * 111320.0
        vx, vy = bx - ax, by - ay
        n2 = vx * vx + vy * vy
        t = 0.0 if n2 == 0 else min(1.0, max(0.0, -(ax * vx + ay * vy) / n2))
        d = math.hypot(ax + t * vx, ay + t * vy)
        seg = haversine(a, b)
        if d < best[0]:
            best = (d, arc + t * seg, lerp(a, b, t))
        arc += seg
    return best


def split_geometry(coords, cut_arcs):
    """Split a polyline at arc positions (metres from start)."""
    cuts = sorted(cut_arcs)
    pieces, cur, ci, arc = [], [coords[0]], 0, 0.0
    for a, b in zip(coords, coords[1:]):
        seg = haversine(a, b)
        while ci < len(cuts) and seg > 0 and arc + seg >= cuts[ci]:
            p = lerp(a, b, min(1.0, max(0.0, (cuts[ci] - arc) / seg)))
            cur.append(p)
            pieces.append(cur)
            cur, ci = [p], ci + 1
        cur.append(b)
        arc += seg
    pieces.append(cur)
    return pieces


class PointIndex:
    """Spatial hash merging points within `tol` metres."""

    def __init__(self, tol):
        self.tol, self.cell = tol, max(tol, 1e-9) / 111_320.0
        self.buckets, self.points, self.merges = defaultdict(list), [], 0

    def add(self, pt):
        ci, cj = int(pt[0] // self.cell), int(pt[1] // self.cell)
        best, best_d = None, self.tol
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for pid in self.buckets.get((ci + di, cj + dj), ()):
                    d = haversine(self.points[pid], pt)
                    if d < best_d:
                        best, best_d = pid, d
        if best is not None:
            self.merges += 1
            return best
        self.points.append([pt[0], pt[1], ele(pt)])
        self.buckets[(ci, cj)].append(len(self.points) - 1)
        return len(self.points) - 1


# ---------------------------------------------------------------- ingest

def linestrings(feature):
    """Coordinate lists of a feature, flattening MultiLineStrings."""
    geom = feature.get("geometry") or {}
    if geom.get("type") == "LineString":
        return [geom["coordinates"]]
    if geom.get("type") == "MultiLineString":
        return geom["coordinates"]
    return []


def feature_areas(feature):
    """(id, name) of every ski area a feature belongs to."""
    out = []
    for entry in (feature.get("properties") or {}).get("skiAreas") or []:
        if isinstance(entry, str):
            out.append((entry, entry))
            continue
        props = entry.get("properties", entry) or {}
        aid, name = props.get("id"), props.get("name")
        if aid or name:
            out.append((aid or name, name or aid))
    return out


def slugify(text):
    text = unicodedata.normalize("NFKD", text or "area")
    text = "".join(c for c in text if not unicodedata.combining(c))
    slug = "".join(c.lower() if c.isascii() and c.isalnum() else "-" for c in text)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "area"


def is_skiable_run(props):
    uses = props.get("uses")
    return not uses or bool(set(uses) & RUN_USES)


def load(path, bbox=None, area=None, area_ids=None, run_filter=False):
    """Read features, optionally filtered by bbox, area name, or ski area id."""
    wanted = set(area_ids) if area_ids else None
    names = [a.strip().lower() for a in area.split("|")] if area else None

    def matches(feat):
        if wanted is not None:
            return any(aid in wanted for aid, _ in feature_areas(feat))
        if names is None:
            return True
        blob = json.dumps(feat["properties"].get("skiAreas") or []).lower()
        return any(n in blob for n in names)

    def inside(coords):
        if bbox is None:
            return True
        x0, y0, x1, y1 = bbox
        return any(x0 <= c[0] <= x1 and y0 <= c[1] <= y1 for c in coords)

    with open(path) as fh:
        data = json.load(fh)
    out = []
    for feat in data.get("features", []):
        props = feat.setdefault("properties", {})
        if run_filter and not is_skiable_run(props):
            continue
        if not matches(feat):
            continue
        out += [(props, c) for c in linestrings(feat)
                if len(c) >= 2 and inside(c)]
    return out


def stream_features(path):
    """
    Yield features one at a time. Uses ijson when available so a multi-hundred-MB
    source file never has to sit in memory as parsed Python objects; falls back
    to json.load otherwise (fine for regional extracts, not for the planet).
    """
    try:
        import ijson
    except ImportError:
        with open(path) as fh:
            yield from json.load(fh).get("features", [])
        return
    with open(path, "rb") as fh:
        # use_float avoids Decimal objects, which are slow and break arithmetic
        yield from ijson.items(fh, "features.item", use_float=True)


def flat_length(coords):
    """Fast planar length in metres — for catalog stats only, not routing."""
    if len(coords) < 2:
        return 0.0
    kx = 111320.0 * math.cos(math.radians(coords[0][1]))
    total = 0.0
    for a, b in zip(coords, coords[1:]):
        total += math.hypot((b[0] - a[0]) * kx, (b[1] - a[1]) * 111320.0)
    return total


def catalog(runs_path, lifts_path, min_lifts=2, min_runs=3):
    """
    One streaming pass over both files: every ski area with enough terrain to
    plan a day, with the stats the resort picker shows. Areas below the
    thresholds are dropped — a single drag lift makes no itinerary.
    """
    areas = {}

    def touch(aid, name):
        a = areas.get(aid)
        if a is None:
            a = areas[aid] = {"id": aid, "name": name or aid, "slug": None,
                              "runs": 0, "lifts": 0, "piste_m": 0.0}
        elif name and a["name"] == aid:
            a["name"] = name
        return a

    for feat in stream_features(runs_path):
        if not is_skiable_run(feat.get("properties") or {}):
            continue
        areas_here = feature_areas(feat)
        if not areas_here:
            continue
        length = sum(flat_length(c) for c in linestrings(feat))
        for aid, name in areas_here:
            a = touch(aid, name)
            a["runs"] += 1
            a["piste_m"] += length

    for feat in stream_features(lifts_path):
        for aid, name in feature_areas(feat):
            touch(aid, name)["lifts"] += 1

    out = []
    for a in areas.values():
        if a["lifts"] < min_lifts or a["runs"] < min_runs:
            continue
        a["slug"] = slugify(a["name"])
        a["piste_km"] = round(a["piste_m"] / 1000, 1)
        del a["piste_m"]
        out.append(a)
    out.sort(key=lambda a: (-a["piste_km"], a["name"]))

    seen = {}
    for a in out:                      # keep slugs unique for filenames
        n = seen.get(a["slug"], 0)
        seen[a["slug"]] = n + 1
        if n:
            a["slug"] = f"{a['slug']}-{n + 1}"
    return out


# ---------------------------------------------------------------- build

def split_at_junctions(runs, lifts, tol):
    idx = PointIndex(tol)
    ways, touched = [], defaultdict(set)
    for props, coords in runs:
        ids = [idx.add(p) for p in coords]
        ways.append((props, ids, coords))
        for vid in set(ids):
            touched[vid].add(len(ways) - 1)
    shared = {v for v, ws in touched.items() if len(ws) > 1}

    segments = []
    for props, ids, coords in ways:
        cuts = sorted({0, len(ids) - 1}
                      | {i for i in range(1, len(ids) - 1) if ids[i] in shared})
        segments += [("run", props, coords[a:b + 1])
                     for a, b in zip(cuts, cuts[1:]) if b > a]
    return segments + [("lift", p, c) for p, c in lifts], len(shared)


def assemble(runs, lifts, tol_junction, tol_hub):
    segments, nshared = split_at_junctions(runs, lifts, tol_junction)
    hubs, raw, warnings = PointIndex(tol_hub), [], defaultdict(int)

    for kind, props, seg in segments:
        uphill = ele(seg[0]) < ele(seg[-1])
        if (kind == "run") == uphill:
            seg = seg[::-1]
        if kind == "run":
            mode = "skate" if ele(seg[0]) - ele(seg[-1]) < FLAT_DROP_M else "run"
        else:
            mode = "lift"
        u, v = hubs.add(seg[0]), hubs.add(seg[-1])
        if u == v:
            warnings[f"degenerate_{mode}"] += 1
            continue
        raw.append({"mode": mode, "props": props, "u": u, "v": v, "geometry": seg})
    return hubs, raw, nshared, warnings


def link_danglers(hubs, raw, radius, end_snap=30.0, dedupe=5.0):
    """Join unreachable nodes to the nearest run via walk connectors."""
    degree, modes = defaultdict(int), defaultdict(set)
    for e in raw:
        for n in (e["u"], e["v"]):
            degree[n] += 1
            modes[n].add(e["mode"])
    danglers = sorted({n for n in degree
                       if degree[n] == 1 or modes[n] == {"lift"}})

    proposals, connectors, linked = defaultdict(list), [], 0
    for node in danglers:
        pt = hubs.points[node]
        best = (radius, None, None)
        for ei, e in enumerate(raw):
            if e["mode"] in ("run", "skate") and node not in (e["u"], e["v"]):
                d, arc, _ = project_onto_polyline(pt, e["geometry"])
                if d < best[0]:
                    best = (d, ei, arc)
        gap, ei, arc = best
        if ei is None:
            continue
        linked += 1
        total = slope_length(raw[ei]["geometry"])
        if arc <= end_snap:
            connectors.append((node, raw[ei]["u"]))
        elif arc >= total - end_snap:
            connectors.append((node, raw[ei]["v"]))
        else:
            proposals[ei].append((arc, node))

    out = []
    for ei, e in enumerate(raw):
        if ei not in proposals:
            out.append(e)
            continue
        merged = []
        for arc, node in sorted(proposals[ei]):
            if merged and arc - merged[-1][0] < dedupe:
                merged[-1][1].append(node)
            else:
                merged.append((arc, [node]))
        pieces = split_geometry(e["geometry"], [m[0] for m in merged])
        chain = [e["u"]]
        for (_, attached), piece in zip(merged, pieces):
            hubs.points.append(piece[-1][:3])
            chain.append(len(hubs.points) - 1)
            connectors += [(n, chain[-1]) for n in attached]
        chain.append(e["v"])
        out += [{**e, "u": chain[i], "v": chain[i + 1], "geometry": seg}
                for i, seg in enumerate(pieces)]

    out += [{"mode": "walk", "props": {}, "u": a, "v": b,
             "geometry": [hubs.points[a], hubs.points[b]]} for a, b in connectors]
    return out, len(connectors), linked


def finalize(raw):
    """Expand internal edges into directed edges with times."""
    edges = []
    for e in raw:
        seg, props = e["geometry"], e["props"]
        length = slope_length(seg)
        name = props.get("name")

        if e["mode"] == "run":
            drop = ele(seg[0]) - ele(seg[-1])
            speed = SKI_SPEED.get((props.get("difficulty") or "").lower(),
                                  DEFAULT_SKI_SPEED)
            edges.append({"type": "run", "u": e["u"], "v": e["v"], "name": name,
                          "difficulty": props.get("difficulty"),
                          "length_m": round(length, 1),
                          "vertical_m": round(max(drop, 0.0), 1),
                          "seconds": round(length / speed, 1), "geometry": seg})
            if drop < GENTLE_DROP_M:   # short gentle pieces would wall off hubs
                edges.append({"type": "walk", "u": e["v"], "v": e["u"],
                              "name": name, "length_m": round(length, 1),
                              "vertical_m": 0.0,
                              "seconds": round(max(length, 1.0) / WALK_SPEED, 1),
                              "geometry": seg[::-1]})

        elif e["mode"] == "lift":
            kind = (props.get("liftType") or props.get("aerialway") or "").lower()
            secs, rule = lift_seconds(props, length)
            secs = round(secs, 1)
            LIFT_RULES[rule] += 1
            base = {"type": "lift", "name": name, "timing": rule,
                    "lift_type": props.get("liftType"),
                    "length_m": round(length, 1), "seconds": secs}
            edges.append({**base, "u": e["u"], "v": e["v"], "geometry": seg,
                          "vertical_m": round(ele(seg[-1]) - ele(seg[0]), 1)})
            if kind in BIDIRECTIONAL_LIFTS:   # cabins carry riders down too
                edges.append({**base, "u": e["v"], "v": e["u"],
                              "geometry": seg[::-1], "vertical_m": 0.0})

        else:   # skate / walk
            speed = SKATE_SPEED if e["mode"] == "skate" else WALK_SPEED
            base = {"type": e["mode"], "name": name, "length_m": round(length, 1),
                    "vertical_m": 0.0, "seconds": round(max(length, 1.0) / speed, 1)}
            edges.append({**base, "u": e["u"], "v": e["v"], "geometry": seg})
            edges.append({**base, "u": e["v"], "v": e["u"], "geometry": seg[::-1]})
    return edges


def build(runs, lifts, tol_junction, tol_hub, radius):
    hubs, raw, nshared, warnings = assemble(runs, lifts, tol_junction, tol_hub)
    raw, nconn, nlinked = link_danglers(hubs, raw, radius)
    return hubs.points, finalize(raw), {
        "nshared": nshared, "merges": hubs.merges,
        "connectors": nconn, "linked": nlinked, "warnings": warnings}


# ---------------------------------------------------------------- analysis

def strong_components(n, edges):
    """Kosaraju, iterative."""
    fwd, rev = defaultdict(list), defaultdict(list)
    for e in edges:
        fwd[e["u"]].append(e["v"])
        rev[e["v"]].append(e["u"])

    seen, order = set(), []
    for s in range(n):
        if s in seen:
            continue
        seen.add(s)
        stack = [(s, iter(fwd[s]))]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                order.append(node)
                stack.pop()
            elif nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(fwd[nxt])))

    done, comps = set(), []
    for s in reversed(order):
        if s in done:
            continue
        done.add(s)
        group, queue = [], deque([s])
        while queue:
            node = queue.popleft()
            group.append(node)
            for prev in rev[node]:
                if prev not in done:
                    done.add(prev)
                    queue.append(prev)
        comps.append(group)
    return sorted(comps, key=len, reverse=True)


def analyse(nodes, edges, meta=None, verbose=True):
    scc = strong_components(len(nodes), edges)
    core = set(scc[0]) if scc else set()

    def split(kind):
        allk = [e for e in edges if e["type"] == kind]
        return allk, [e for e in allk if e["u"] in core and e["v"] in core]

    runs_all, runs_in = split("run")
    lifts_all, lifts_in = split("lift")
    vert_all = sum(e["vertical_m"] for e in runs_all) or 1.0
    vert_in = sum(e["vertical_m"] for e in runs_in)

    if verbose:
        m = meta or {}
        print(f"nodes                 {len(nodes)}")
        print(f"shared vertices       {m.get('nshared', '-')}")
        print(f"hub merges            {m.get('merges', '-')}")
        print(f"danglers linked       {m.get('linked', '-')} "
              f"({m.get('connectors', '-')} walk connectors)")
        print(f"edges                 {len(edges)} directed")
        print(f"runs in core          {len(runs_in)}/{len(runs_all)} "
              f"({len(runs_in) / max(len(runs_all), 1):.0%})")
        print(f"lifts in core         {len(lifts_in)}/{len(lifts_all)} "
              f"({len(lifts_in) / max(len(lifts_all), 1):.0%})")
        print(f"vertical in core      {vert_in:,.0f} m / {vert_all:,.0f} m "
              f"({vert_in / vert_all:.0%})   <-- the number that matters")
        print(f"top SCC sizes         {[len(c) for c in scc[:5]]}")
        if LIFT_RULES:
            tot = sum(LIFT_RULES.values())
            print("lift ride times       " + ", ".join(
                f"{k} {v} ({v / tot:.0%})"
                for k, v in sorted(LIFT_RULES.items(), key=lambda p: -p[1])))
        if m.get("warnings"):
            print("warnings              "
                  + ", ".join(f"{k}={v}" for k, v in m["warnings"].items()))

        in_core = {id(e) for e in lifts_in}
        missing = sorted((e for e in lifts_all if id(e) not in in_core),
                         key=lambda e: -e["length_m"])[:8]
        if missing:
            print("\nLongest lifts still NOT in core:")
            for e in missing:
                print(f"  {e['length_m']:>7.0f} m  {e.get('lift_type') or '?':<14}"
                      f" {e.get('name') or '(unnamed)'}")
    return core, len(lifts_in), len(lifts_all), vert_in / vert_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("lifts")
    ap.add_argument("--bbox", help="minlon,minlat,maxlon,maxlat")
    ap.add_argument("--area", help="ski area name; '|' separates alternatives")
    ap.add_argument("--area-id", action="append",
                    help="ski area id (repeatable); exact, unlike --area")
    ap.add_argument("--catalog", action="store_true",
                    help="list every ski area in the data and exit")
    ap.add_argument("--tol-junction", type=float, default=2.0)
    ap.add_argument("--tol-hub", type=float, default=30.0)
    ap.add_argument("--link-radius", type=float, default=100.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="graph.json")
    args = ap.parse_args()

    if args.catalog:
        rows = catalog(args.runs, args.lifts)
        print(f"{len(rows)} ski areas\n")
        print(f"{'piste km':>9} {'lifts':>6} {'runs':>6}  name")
        for a in rows:
            print(f"{a['piste_km']:>9} {a['lifts']:>6} {a['runs']:>6}  "
                  f"{a['name']}  [{a['id']}]")
        return

    bbox = [float(x) for x in args.bbox.split(",")] if args.bbox else None
    runs = load(args.runs, bbox, args.area, args.area_id, run_filter=True)
    lifts = load(args.lifts, bbox, args.area, args.area_id)
    print(f"loaded {len(runs)} run geometries, {len(lifts)} lift geometries\n")

    if args.sweep:
        print(f"{'hub':>5} {'radius':>7} {'lifts':>9} {'vert%':>7}")
        for hub in (15, 30, 45):
            for rad in (50, 100, 150):
                nodes, edges, _ = build(runs, lifts, args.tol_junction, hub, rad)
                _, li, la, vf = analyse(nodes, edges, verbose=False)
                print(f"{hub:>5} {rad:>7} {li:>4}/{la:<4} {vf:>6.0%}")
        return

    nodes, edges, meta = build(runs, lifts, args.tol_junction,
                               args.tol_hub, args.link_radius)
    core, *_ = analyse(nodes, edges, meta)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"nodes": nodes, "edges": edges, "core": sorted(core)}, fh)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()