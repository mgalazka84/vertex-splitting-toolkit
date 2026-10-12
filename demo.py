"""Reduce an interval-splitting instance and lift an exact solution."""

from vertex_splitting import (
    check_certificate,
    graph_from_edges,
    kernelize,
    lift_kernel_witness,
    solve_exact,
)


def main():
    # A four-cycle with six additional leaves at vertex 0.
    graph = graph_from_edges(
        10,
        [(0, 1), (1, 2), (2, 3), (3, 0)] + [(0, v) for v in range(4, 10)],
    )
    budget = 1
    target = "interval"
    exclusive = True
    shallow = True
    costs = (1,) * len(graph)

    reduced = kernelize(graph, budget, costs, target, shallow=shallow)
    print(f"Original vertices: {len(graph)}")
    if reduced.rejected:
        print("No solution: preprocessing rejected the instance.")
        return
    print(f"Retained vertices: {len(reduced.graph)}")
    witness = solve_exact(
        reduced.graph, reduced.budget, target, reduced.costs,
        exclusive=exclusive, shallow=shallow, max_nodes=500000,
    )
    # An empty tuple is a valid zero-split solution; only None means infeasible.
    if witness is None:
        print("No solution within the budget.")
        return
    certificate = lift_kernel_witness(
        graph, reduced, witness, target, costs,
        exclusive=exclusive, shallow=shallow,
    )
    assert check_certificate(
        graph, budget, certificate, target, costs,
        exclusive=exclusive, shallow=shallow,
    )
    print(f"Splits used: {len(witness)}")
    print(f"Vertices after lifting: {len(certificate.graph)}")
    print("Certificate valid: True")


if __name__ == "__main__":
    main()
