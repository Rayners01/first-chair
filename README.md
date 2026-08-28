# First Chair

Plans the mathematically optimal ski day: given a resort, a starting lift and
how long you've got, it works out which runs to ski in which order, and which
lifts to catch between them.

"Best" is a time-budgeted **arc orienteering problem**. Every piste carries a
prize collected only on its first descent, and the objective blends vertical
descended against piste covered — one slider moves the plan from *see
everything* to *most vertical*. The problem is NP-hard, so it's solved three
ways and the results are measured against each other:

| Solver | Role |
| --- | --- |
| `solve_greedy.py` | Myopic baseline. Defines the shared objective every solver uses. |
| `solve_ilp.py` | Exact flow MILP. Proves optimality where the instance allows it. |
| `solve_anneal.py` | Simulated annealing. Fast enough to run interactively. |

Pistes, lifts and elevations come from OpenStreetMap via
[OpenSkiMap](https://openskimap.org).
## Running it

```bash
pip install fastapi uvicorn pulp ijson

# build a resort graph from OpenSkiMap runs.geojson / lifts.geojson
python build_graph.py runs.geojson lifts.geojson --area "Tignes|Val d'Isere" \
    --out graphs/tignes-val-disere.json

uvicorn app:app --port 8000        # then open http://localhost:8000
```

Drop the source `runs.geojson` and `lifts.geojson` at the project root and any
other ski area in the data can be built on demand from the resort picker.

## Status

Working end to end. Benchmarks, write-up and a proper account of how the graph
is built are still to come.