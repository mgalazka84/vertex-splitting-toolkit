# Vertex Splitting Toolkit

Dependency-free Python tools for vertex splitting into chordal, interval,
and unit interval graphs.

## Requirements

Python 3.10 or newer. No third-party packages are required.

## Run

```bash
python demo.py
python verify.py --output verification_results.json
```

The demo reduces a ten-vertex graph to seven vertices, finds a one-split
solution, and lifts it to a checked certificate on the original graph.
The verification command checks graph recognition, structural bounds,
reduction equivalence, and reconstructed solutions. Random inputs use a
fixed seed. The JSON report contains counts and timings from your run.

## Files

- `vertex_splitting.py`: graph operations, recognition, twin-class reductions,
  exact splitting search, and certificate reconstruction.
- `verify.py`: exhaustive checks on small graphs, structured examples,
  and seeded weighted cases.
- `demo.py`: a complete reduction, search, and reconstruction example.

## Supported options

| Option | Values |
| --- | --- |
| Target | `chordal`, `interval`, `unit_interval` |
| Exclusive splitting | `False` or `True` |
| Shallow splitting | `False` or `True` |
| Vertex costs | Positive integers; defaults to unit costs |

A general split assigns each neighbor to one or both new vertices. An
exclusive split assigns it to exactly one. In shallow mode, a newly created
vertex cannot be split again. Both descendants inherit the original cost.

Graphs are tuples of open-neighborhood bit masks. Use `graph_from_edges`
to construct a graph from numbered vertices and undirected edges.

## API notes

`kernelize(graph, budget, costs=None, target="interval", shallow=False)`
returns a `Kernel` with the reduced graph, costs, retained original vertex
indices, budget, and a rejection flag. Pass `shallow` as a keyword.

`solve_exact` returns a tuple of split operations or `None` when no solution
exists within the budget. An empty tuple is a valid solution. Set
`max_nodes` to bound search effort: reaching the limit raises `SearchLimit`
and does not count as an infeasible instance.

`lift_kernel_witness` restores deleted vertices. `check_certificate` verifies
the resulting graph, descendant sets, original costs, and variant constraints.
When combining these functions, use the same target and splitting options
throughout, as in `demo.py`.

Exact search and the independent exhaustive recognizers are intended for
small instances. The test suite provides finite consistency checks.
