"""
app.py — First Chair server: wraps the solvers behind a small API and serves the UI.

Resorts come from two places:
  * graphs/<slug>.json — already built, instantly available;
  * the source runs.geojson / lifts.geojson — every other ski area in the data,
    built on demand by /api/build and then cached in graphs/ like the rest.

    uvicorn app:app --port 8000     ->  http://localhost:8000

Every served route is audited (continuity, budget, re-scored with the shared
objective) before it leaves the server; unverifiable routes 500 instead.
"""

import hashlib
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from build_graph import analyse, build, catalog, load
from solve_greedy import load_graph, compute_prizes, score_itinerary, hhmm
from solve_anneal import all_dists, solve as solve_day

BASE_DIR = Path(__file__).parent
GRAPH_DIR = BASE_DIR / "graphs"
CACHE_DIR = BASE_DIR / "cache"
STATIC_DIR = BASE_DIR / "static"
SRC_RUNS = BASE_DIR / "runs.geojson"
SRC_LIFTS = BASE_DIR / "lifts.geojson"
CATALOG_FILE = CACHE_DIR / "catalog.json"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SOLVE_SECONDS = 8.0
RESTARTS = 4          # more restarts = less run-to-run variance in the plan
MIN_CORE_VERTICAL = 0.35   # reject a built graph this disconnected

app = FastAPI(title="First Chair")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_resorts = {}                 # slug -> loaded graph + precomputed distances
_stats = {}                   # slug -> (mtime, lightweight stats)
_build_locks = {}             # slug -> Lock, so one build runs at a time
_locks_guard = threading.Lock()

_catalog = None               # None until the source scan finishes
_catalog_state = "idle"       # idle | scanning | ready | none
_catalog_lock = threading.Lock()


# ------------------------------------------------------------ resort catalog

def _scan_catalog():
    """Scan the source data once. Slow on big extracts, so never inline."""
    global _catalog, _catalog_state
    try:
        stamp = max(SRC_RUNS.stat().st_mtime, SRC_LIFTS.stat().st_mtime)
        if CATALOG_FILE.exists():
            cached = json.loads(CATALOG_FILE.read_text())
            if cached.get("stamp") == stamp:
                _catalog, _catalog_state = cached["areas"], "ready"
                return
        areas = catalog(SRC_RUNS, SRC_LIFTS)
        CATALOG_FILE.write_text(json.dumps({"stamp": stamp, "areas": areas}))
        _catalog, _catalog_state = areas, "ready"
    except Exception as exc:                       # never take the server down
        print(f"catalog scan failed: {exc}")
        _catalog, _catalog_state = [], "none"


def start_catalog_scan():
    """Kick the scan off in the background; requests never wait on it."""
    global _catalog_state
    with _catalog_lock:
        if _catalog_state in ("scanning", "ready", "none"):
            return
        if not (SRC_RUNS.exists() and SRC_LIFTS.exists()):
            _catalog_state = "none"
            return
        _catalog_state = "scanning"
    threading.Thread(target=_scan_catalog, daemon=True).start()


@app.on_event("startup")
def on_startup():
    start_catalog_scan()


def source_catalog():
    """Areas found so far — empty while the background scan is still running."""
    return _catalog or []


def built_slugs():
    return sorted(p.stem for p in GRAPH_DIR.glob("*.json"))


def graph_stats(slug):
    """
    Stats for the picker, WITHOUT the expensive parts of get_resort: no
    all-pairs distances, no adjacency. Cached against the file's mtime.
    """
    path = GRAPH_DIR / f"{slug}.json"
    mtime = path.stat().st_mtime
    hit = _stats.get(slug)
    if hit and hit[0] == mtime:
        return hit[1]
    with open(path) as fh:
        g = json.load(fh)
    core = set(g["core"])
    edges = [e for e in g["edges"] if e["u"] in core and e["v"] in core]
    runs = [e for e in edges if e["type"] == "run"]
    stats = {"runs": len(runs),
             "lifts": sum(1 for e in edges if e["type"] == "lift"),
             "vertical_m": round(sum(e["vertical_m"] for e in runs)),
             "piste_km": round(sum(e["length_m"] for e in runs) / 1000, 1)}
    _stats[slug] = (mtime, stats)
    return stats


def get_resort(slug):
    """Load and cache a built graph, precomputing all-pairs travel times."""
    if slug in _resorts:
        return _resorts[slug]
    path = GRAPH_DIR / f"{slug}.json"
    if not path.exists():
        raise HTTPException(404, f"{slug!r} has not been built yet")
    nodes, edges, fwd, rev, core = load_graph(path)
    if not edges:
        raise HTTPException(500, f"{slug!r} has an empty core")

    used = {e["u"] for e in edges} | {e["v"] for e in edges}
    runs = [e for e in edges if e["type"] == "run"]
    lifts = {}
    for e in edges:
        name = e.get("name")
        if e["type"] == "lift" and name and name not in lifts:
            lifts[name] = e["u"]

    _resorts[slug] = {
        "nodes": nodes, "edges": edges, "fwd": fwd, "rev": rev,
        "D": all_dists(edges, fwd, used),
        "starts": [{"node": n, "name": name} for name, n in sorted(lifts.items())],
        "stats": {
            "runs": len(runs),
            "lifts": sum(1 for e in edges if e["type"] == "lift"),
            "vertical_m": round(sum(e["vertical_m"] for e in runs)),
            "piste_km": round(sum(e["length_m"] for e in runs) / 1000, 1),
        },
    }
    return _resorts[slug]


def display_name(slug):
    for a in source_catalog():
        if a["slug"] == slug:
            return a["name"]
    return slug.replace("-", " ").title()


def wait_for_catalog(timeout=180):
    """Block only where a caller genuinely needs the catalog (i.e. building)."""
    start_catalog_scan()
    deadline = time.time() + timeout
    while _catalog_state == "scanning" and time.time() < deadline:
        time.sleep(0.25)
    return source_catalog()


# ------------------------------------------------------------ requests

class SolveRequest(BaseModel):
    resort: str
    hours: float = 6.0
    lam: float = 0.5
    start_node: int | None = None
    return_to_start: bool = True


class BuildRequest(BaseModel):
    area_id: str


def audit_route(itinerary, edges, start, budget_s, closed):
    """Independent check before anything is served. Returns (clock, error)."""
    node, clock = start, 0.0
    for i in itinerary:
        if edges[i]["u"] != node:
            return None, "route discontinuity"
        node, clock = edges[i]["v"], clock + edges[i]["seconds"]
    if closed and node != start:
        return None, "route does not return to start"
    if clock > budget_s + 1e-6:
        return None, "route exceeds the time budget"
    return clock, None


def collapse_legs(itinerary, edges):
    """Merge consecutive segments of the same way into readable legs."""
    fallback = {"walk": "Short walk", "skate": "Flat traverse",
                "run": "Unnamed run", "lift": "Lift"}
    legs, t = [], 0.0
    for i in itinerary:
        e = edges[i]
        label = e.get("name") or fallback[e["type"]]
        if legs and legs[-1]["type"] == e["type"] and legs[-1]["label"] == label:
            legs[-1]["seconds"] += e["seconds"]
            legs[-1]["vert"] += e["vertical_m"] if e["type"] == "run" else 0
        else:
            legs.append({"t0": t, "type": e["type"], "label": label,
                         "seconds": e["seconds"],
                         "difficulty": e.get("difficulty"),
                         "vert": e["vertical_m"] if e["type"] == "run" else 0})
        t += e["seconds"]
    for leg in legs:
        leg["time"] = hhmm(leg["t0"])
        leg["mins"] = round(leg["seconds"] / 60, 1)
        leg["vert"] = round(leg["vert"])
    return legs


# ------------------------------------------------------------ endpoints

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.ico")

@app.get("/api/resorts")
def list_resorts():
    """
    Built resorts plus, once the background scan finishes, everything else in
    the source data. Deliberately cheap: no graph is fully loaded here.
    """
    start_catalog_scan()
    ready_slugs = set(built_slugs())
    ready = sorted(
        ({"slug": s, "name": display_name(s), "stats": graph_stats(s)}
         for s in ready_slugs),
        key=lambda r: -r["stats"]["piste_km"])

    available = [{"id": a["id"], "slug": a["slug"], "name": a["name"],
                  "piste_km": a["piste_km"], "lifts": a["lifts"]}
                 for a in source_catalog() if a["slug"] not in ready_slugs]
    return {"ready": ready, "available": available,
            "catalog": _catalog_state}


@app.get("/api/resorts/{slug}")
def resort_detail(slug: str):
    r = get_resort(slug)
    return {"slug": slug, "name": display_name(slug),
            "starts": r["starts"], "stats": r["stats"]}


@app.post("/api/build")
def build_resort(req: BuildRequest):
    """Build a graph for one ski area from the source data, then cache it."""
    area = next((a for a in wait_for_catalog() if a["id"] == req.area_id), None)
    if area is None:
        raise HTTPException(404, f"unknown ski area {req.area_id!r}")

    slug, out = area["slug"], GRAPH_DIR / f"{area['slug']}.json"
    with _locks_guard:
        lock = _build_locks.setdefault(slug, threading.Lock())

    with lock:                       # a second request waits, then reuses it
        if out.exists():
            return {"slug": slug, "name": area["name"], "reused": True,
                    "stats": graph_stats(slug)}

        t0 = time.time()
        runs = load(SRC_RUNS, area_ids=[req.area_id], run_filter=True)
        lifts = load(SRC_LIFTS, area_ids=[req.area_id])
        if not runs or not lifts:
            raise HTTPException(422, f"{area['name']} has no mapped terrain")

        nodes, edges, _ = build(runs, lifts, 2.0, 30.0, 100.0)
        core, _, _, vert_frac = analyse(nodes, edges, verbose=False)
        if vert_frac < MIN_CORE_VERTICAL:
            raise HTTPException(
                422, f"{area['name']} is too disconnected to plan a day "
                     f"(only {vert_frac:.0%} of its pistes link up)")

        out.write_text(json.dumps({"nodes": nodes, "edges": edges,
                                   "core": sorted(core)}))
        _stats.pop(slug, None)
        return {"slug": slug, "name": area["name"], "reused": False,
                "build_seconds": round(time.time() - t0, 1),
                "core_vertical_pct": round(vert_frac * 100),
                "stats": graph_stats(slug)}


@app.post("/api/solve")
def plan_day(req: SolveRequest):
    r = get_resort(req.resort)
    edges, fwd, rev, D = r["edges"], r["fwd"], r["rev"], r["D"]
    budget = req.hours * 3600
    start = req.start_node if req.start_node is not None \
        else max((e for e in edges if e["type"] == "lift"),
                 key=lambda e: e["length_m"])["u"]

    key = hashlib.sha1(json.dumps(
        [req.resort, req.hours, round(req.lam, 2), start,
         req.return_to_start]).encode()).hexdigest()[:16]
    cached = CACHE_DIR / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    t0 = time.time()
    prizes = compute_prizes(edges, req.lam)
    itinerary, objective, g_itin = solve_day(
        edges, fwd, rev, prizes, D, start, budget,
        SOLVE_SECONDS, RESTARTS, seed=0, closed=req.return_to_start)
    if not itinerary:
        raise HTTPException(500, "no feasible route found for these settings")

    clock, err = audit_route(itinerary, edges, start, budget,
                             req.return_to_start)
    if err:
        raise HTTPException(500, f"route failed verification: {err}")

    unique = {i for i in itinerary if i in prizes}
    v_tot = sum(edges[i]["vertical_m"] for i in prizes) or 1.0
    resp = {
        "objective": round(objective, 4),
        "greedy_objective": round(score_itinerary(g_itin, prizes), 4),
        "solve_seconds": round(time.time() - t0, 1),
        "start_node": start,
        "totals": {
            "vertical_m": round(sum(edges[i]["vertical_m"] for i in unique)),
            "distance_km": round(sum(edges[i]["length_m"]
                                     for i in itinerary) / 1000, 1),
            "unique_runs": len({edges[i].get("name") or i for i in unique}),
            "coverage_pct": round(100 * sum(edges[i]["vertical_m"]
                                            for i in unique) / v_tot),
            "home": hhmm(clock),
            "lift_rides": sum(1 for i in itinerary
                              if edges[i]["type"] == "lift"),
        },
        "legs": collapse_legs(itinerary, edges),
        "edges": [{k: edges[i][k] for k in
                   ("type", "name", "difficulty", "seconds", "vertical_m",
                    "length_m", "geometry") if k in edges[i]}
                  for i in itinerary],
    }
    cached.write_text(json.dumps(resp))
    return resp


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")