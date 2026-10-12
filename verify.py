"""Deterministic and seeded checks for graph recognition and vertex splitting.

Run: python verify.py --output verification_results.json
All randomness is seeded. The exact oracle never invokes kernelization.
"""

import argparse
import json
import platform
import random
import time
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

from vertex_splitting import (
    TARGETS, Certificate, SearchLimit, Split, blowup, check_certificate, check_witness, chordal_by_induced_cycles,
    graph_from_edges, induced, interval_by_clique_order, is_chordal, is_clique,
    is_independent, is_interval, kernelize, labeled_graphs, solve_exact,
    lift_kernel_witness, replay_certificate, split_graph, twin_partition,
    unit_interval_by_order, vertices,
)


def check_equivalence(g, budget, target, costs=None, exclusive=False, shallow=False):
    reduced = kernelize(g, budget, costs, target, shallow=shallow)
    witness = solve_exact(g, budget, target, costs, exclusive, shallow, max_nodes=500000)
    full_yes = witness is not None
    if full_yes:
        assert check_witness(g, budget, witness, target, costs, exclusive, shallow)
        assert check_certificate(g, budget, replay_certificate(g, witness), target,
                                 costs, exclusive, shallow)
    if reduced.rejected:
        reduced_yes = False
    else:
        witness2 = solve_exact(reduced.graph, budget, target, reduced.costs,
                               exclusive, shallow, max_nodes=500000)
        reduced_yes = witness2 is not None
        if reduced_yes:
            assert check_witness(reduced.graph, budget, witness2, target, reduced.costs,
                                 exclusive, shallow)
    if full_yes != reduced_yes:
        raise AssertionError({"graph": g, "budget": budget, "target": target,
                              "costs": costs, "exclusive": exclusive, "shallow": shallow,
                              "full_yes": full_yes, "reduced_yes": reduced_yes,
                              "reduced": reduced.__dict__})
    changed = len(reduced.graph) < len(g) or reduced.rejected
    lifted = changed and reduced_yes
    if lifted:
        cert = lift_kernel_witness(g, reduced, witness2, target, costs, exclusive, shallow)
        assert check_certificate(g, budget, cert, target, costs, exclusive, shallow)
    return changed, lifted


@lru_cache(maxsize=None)
def set_partitions(mask):
    """Every unordered partition of a bit-mask set into nonempty blocks."""
    if not mask:
        return ((),)
    first = mask & -mask
    result = []
    for partition in set_partitions(mask ^ first):
        result.append((first,) + partition)
        for i in range(len(partition)):
            result.append(partition[:i] + (partition[i] | first,) + partition[i + 1:])
    return tuple(result)


def run():
    start = time.perf_counter()
    stats = {"python": platform.python_version(), "seed": 20271011,
             "recognition_checks": {}, "boundary_lemma_checks": 0,
             "covering_boundary_lemma_checks": 0,
             "lifted_changed_yes_instances": 0, "sharpness_checks": {},
             "kernel_equivalence_checks": {}, "changed_instances": {},
             "search_limits": 0, "scope": "finite deterministic and seeded regression checks"}

    recognition = {"chordal_vs_induced_cycles": 0,
                   "interval_vs_consecutive_clique_order": 0,
                   "unit_interval_vs_umbrella_order": 0}
    for n in range(7):
        for g in labeled_graphs(n):
            assert is_chordal(g) == chordal_by_induced_cycles(g), ("chordal", g)
            recognition["chordal_vs_induced_cycles"] += 1
            assert is_interval(g) == interval_by_clique_order(g), ("interval", g)
            recognition["interval_vs_consecutive_clique_order"] += 1
            if n <= 5:
                assert TARGETS["unit_interval"](g) == unit_interval_by_order(g), ("unit", g)
                recognition["unit_interval_vs_umbrella_order"] += 1
            if is_chordal(g):
                for subset in range(1 << n):
                    if subset.bit_count() < 2 or not is_independent(g, subset):
                        continue
                    selected = tuple(vertices(subset))
                    common = (1 << n) - 1
                    for u in selected:
                        common &= g[u]
                    assert is_clique(g, common)
                    any_neighbor = 0
                    for u in selected:
                        any_neighbor |= g[u]
                    # The smallest admissible boundary consists only of vertices
                    # with mixed adjacency to S; uniform nonneighbors belong to Z.
                    boundary = any_neighbor & ~common & ~subset
                    nonsimplicial = sum(not is_clique(g, g[u]) for u in selected)
                    assert nonsimplicial <= boundary.bit_count(), ("boundary", g, subset)
                    stats["boundary_lemma_checks"] += 1
                    if n <= 5:
                        # All admissible boundaries, including uniformly adjacent
                        # descendants, and every partition that covers S per block.
                        optional = ((1 << n) - 1) & ~subset & ~boundary
                        extra = optional
                        while True:
                            bmask = boundary | extra
                            if bmask:
                                for partition in set_partitions(bmask):
                                    if all(g[x] & block for x in selected for block in partition):
                                        assert nonsimplicial <= bmask.bit_count() - len(partition) + 1, (
                                            "covering_boundary", g, subset, partition)
                                        stats["covering_boundary_lemma_checks"] += 1
                            if not extra:
                                break
                            extra = (extra - 1) & optional
        print(f"recognition and boundary: all labeled graphs n={n} passed", flush=True)
    stats["recognition_checks"] = recognition

    def record(family, g, budget, target, costs=None, exclusive=False, shallow=False):
        try:
            changed, lifted = check_equivalence(g, budget, target, costs, exclusive, shallow)
        except SearchLimit:
            stats["search_limits"] += 1
            return
        stats["kernel_equivalence_checks"][family] = stats["kernel_equivalence_checks"].get(family, 0) + 1
        stats["changed_instances"][family] = stats["changed_instances"].get(family, 0) + int(changed)
        stats["lifted_changed_yes_instances"] += int(lifted)

    for n in range(6):
        for g in labeled_graphs(n):
            for target in TARGETS:
                for budget in (0, 1):
                    for exclusive in (False, True):
                        record("all_labeled_n_le_5_k_le_1", g, budget, target, exclusive=exclusive)
        print(f"unweighted equivalence: all labeled graphs n={n} passed", flush=True)

    # Force deletion from large false-twin classes, rather than rely on random graphs.
    for q in labeled_graphs(3):
        for types in product((False, True), repeat=3):
            for sizes in ((5, 1, 1), (5, 2, 1), (3, 2, 2), (1, 1, 5)):
                g = blowup(q, sizes, types)
                for target in TARGETS:
                    for exclusive in (False, True):
                        record("structured_twin_blowups_k_1", g, 1, target, exclusive=exclusive)
    print("structured twin blowups passed", flush=True)

    # YES instances outside the target with a false-twin class that is truncated.
    # A split on the cycle opens it into a caterpillar; large leaf twins survive.
    for cycle_length in (4, 5):
        for leaves in (5, 7):
            edges = [(v, (v + 1) % cycle_length) for v in range(cycle_length)]
            edges.extend((0, v) for v in range(cycle_length, cycle_length + leaves))
            g = graph_from_edges(cycle_length + leaves, edges)
            assert not is_interval(g)
            for target in TARGETS:
                for exclusive in (False, True):
                    for shallow in (False, True):
                        record("cycle_with_large_leaf_twins_K_1", g, 1, target,
                               exclusive=exclusive, shallow=shallow)

    rng = random.Random(stats["seed"])
    for trial in range(100):
        n = rng.randrange(2, 7)
        edges = [(u, v) for u, v in combinations(range(n), 2) if rng.randrange(2)]
        g = graph_from_edges(n, edges)
        costs = tuple(rng.randrange(1, 5) for _ in range(n))
        budget = rng.randrange(3)
        for target in TARGETS:
            for exclusive in (False, True):
                for shallow in (False, True):
                    record("random_weighted_all_variants_K_le_2", g, budget, target,
                           costs, exclusive, shallow)
    # Explicit nonuniform costs in a large clique or independent twin class.
    for q in labeled_graphs(3):
        for types in ((True, False, False), (False, False, False)):
            g = blowup(q, (5, 1, 1), types)
            costs = (4, 2, 1, 3, 1, 1, 2)
            for target in TARGETS:
                for exclusive in (False, True):
                    for shallow in (False, True):
                        record("weighted_twin_deletion_K_1", g, 1, target, costs,
                               exclusive, shallow)
    print("weighted and shallow variants passed", flush=True)

    # Isolated vertices, true clones and a center with too many independent leaves.
    unit_exact = 0
    for leaves in range(1, 9):
        star = graph_from_edges(leaves + 1, [(0, v) for v in range(1, leaves + 1)])
        for budget in range(3):
            for exclusive in (False, True):
                for shallow in (False, True):
                    record("unit_interval_star_bound", star, budget, "unit_interval",
                           exclusive=exclusive, shallow=shallow)
                    expected = (leaves <= 2 * min(budget + 1, 2) if shallow
                                else leaves <= 2 * (budget + 1))
                    witness = solve_exact(star, budget, "unit_interval", exclusive=exclusive,
                                          shallow=shallow, max_nodes=500000)
                    assert (witness is not None) == expected, ("star_capacity", leaves, budget,
                                                               exclusive, shallow)
                    unit_exact += 1

    # Witnesses at the exact capacity: one center descendant per leaf pair.
    # Certificate checking is independent of the kernel and of split search.
    unit_witnesses = 0
    for budget in range(9):
        for shallow in (False, True):
            centers = min(budget + 1, 2) if shallow else budget + 1
            leaves = 2 * centers
            star = graph_from_edges(leaves + 1, ((0, v) for v in range(1, leaves + 1)))
            h = graph_from_edges(centers + leaves,
                                 ((i, centers + 2 * i + j) for i in range(centers)
                                  for j in (0, 1)))
            cert = Certificate(h, (0,) * centers + tuple(range(1, leaves + 1)))
            for exclusive in (False, True):
                assert check_certificate(star, budget, cert, "unit_interval",
                                         exclusive=exclusive, shallow=shallow)
                unit_witnesses += 1
    stats["sharpness_checks"]["unit_interval_star_exact_capacity_checks_K_le_2"] = unit_exact
    stats["sharpness_checks"]["unit_interval_capacity_certificates_K_le_8"] = unit_witnesses

    # Test the forbidden smaller cap K+1 independently on K_{2,K+2}.
    # The arbitrary-K proof is the edge-count argument, not these finite tests.
    exact_sharpness = 0
    for budget in range(3):
        full = graph_from_edges(budget + 4,
                                ((u, c) for u in (0, 1) for c in range(2, budget + 4)))
        smaller = induced(full, range(budget + 3))
        for target in ("chordal", "interval"):
            for exclusive in (False, True):
                for shallow in (False, True):
                    assert solve_exact(full, budget, target, exclusive=exclusive, shallow=shallow,
                                       max_nodes=500000) is None
                    witness = solve_exact(smaller, budget, target, exclusive=exclusive,
                                          shallow=shallow, max_nodes=500000)
                    assert witness is not None
                    assert check_witness(smaller, budget, witness, target,
                                         exclusive=exclusive, shallow=shallow)
                    exact_sharpness += 1
    stats["sharpness_checks"]["K_2_Kplus2_exact_NO_vs_truncated_YES_K_le_2"] = exact_sharpness

    # Explicit shallow-exclusive witness for K_{2,K+1}, up to K=8.
    constructive_sharpness = 0
    for budget in range(9):
        g = graph_from_edges(budget + 3,
                             ((u, c) for u in (0, 1) for c in range(2, budget + 3)))
        state, owners, witness = g, tuple(range(len(g))), []
        for original in range(3, len(g)):
            i = owners.index(original)
            action = Split(i, (0,), (1,))
            witness.append(action)
            state = split_graph(state, action)
            owners = tuple(x for j, x in enumerate(owners) if j != i) + (original, original)
        for target in ("chordal", "interval"):
            for exclusive in (False, True):
                for shallow in (False, True):
                    assert check_witness(g, budget, witness, target,
                                         exclusive=exclusive, shallow=shallow)
                    constructive_sharpness += 1
    stats["sharpness_checks"]["explicit_caterpillar_witnesses_K_le_8"] = constructive_sharpness
    print("sharpness witnesses and lifting checks passed", flush=True)

    stats["total_kernel_equivalence_checks"] = sum(stats["kernel_equivalence_checks"].values())
    stats["total_changed_instances"] = sum(stats["changed_instances"].values())
    stats["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    stats["status"] = "PASS"
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="verification_results.json")
    args = parser.parse_args()
    results = run()
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
