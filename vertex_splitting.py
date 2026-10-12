"""Vertex splitting, twin-class reductions, and exact graph solvers (Python 3.10+).

Graphs are tuples of open-neighborhood bit masks. Exact search is intended for small graphs and may take exponential time.
No third-party packages are needed. Positive integer costs are inherited by
both children of a split. Newly created vertices cannot be split in shallow mode.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product

Graph = tuple[int, ...]


def vertices(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def graph_from_edges(n: int, edges) -> Graph:
    adj = [0] * n
    for u, v in edges:
        if not 0 <= u < n or not 0 <= v < n or u == v:
            raise ValueError("invalid edge")
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return tuple(adj)


def labeled_graphs(n: int):
    pairs = tuple(combinations(range(n), 2))
    for mask in range(1 << len(pairs)):
        yield graph_from_edges(n, (e for i, e in enumerate(pairs) if mask >> i & 1))


def induced(g: Graph, keep) -> Graph:
    keep = tuple(keep)
    return tuple(sum(1 << j for j, v in enumerate(keep) if g[u] >> v & 1)
                 for u in keep)


def is_clique(g: Graph, subset: int) -> bool:
    return all((subset ^ (1 << v)) & ~g[v] == 0 for v in vertices(subset))


def is_independent(g: Graph, subset: int) -> bool:
    return all(g[v] & subset == 0 for v in vertices(subset))


def twin_partition(g: Graph) -> tuple[tuple[tuple[int, ...], bool], ...]:
    """Return maximal twin classes; bool indicates a clique class.

    Singleton classes are treated as independent, which does not affect safety.
    """
    classes = []
    unseen = set(range(len(g)))
    while unseen:
        u = min(unseen)
        group = tuple(v for v in sorted(unseen)
                      if (g[u] & ~(1 << v)) == (g[v] & ~(1 << u)))
        unseen.difference_update(group)
        clique = len(group) >= 2 and bool(g[group[0]] >> group[1] & 1)
        classes.append((group, clique))
    return tuple(classes)


@lru_cache(maxsize=300000)
def chordal_peo(g: Graph):
    """Maximum cardinality search, followed by direct PEO verification."""
    n = len(g)
    remaining = (1 << n) - 1
    weights = [0] * n
    selection = []
    while remaining:
        u = max(vertices(remaining), key=lambda v: (weights[v], -v))
        selection.append(u)
        remaining ^= 1 << u
        for v in vertices(g[u] & remaining):
            weights[v] += 1
    peo = tuple(reversed(selection))
    remaining = (1 << n) - 1
    for u in peo:
        remaining ^= 1 << u
        if not is_clique(g, g[u] & remaining):
            return None
    return peo


def is_chordal(g: Graph) -> bool:
    return chordal_peo(g) is not None


def chordal_by_induced_cycles(g: Graph) -> bool:
    """Independent exponential recognizer, used only for small-graph checks."""
    for size in range(4, len(g) + 1):
        for selected in combinations(range(len(g)), size):
            mask = sum(1 << v for v in selected)
            if not all((g[v] & mask).bit_count() == 2 for v in selected):
                continue
            seen, frontier = 0, 1 << selected[0]
            while frontier:
                seen |= frontier
                neighbors = 0
                for v in vertices(frontier):
                    neighbors |= g[v]
                frontier = neighbors & mask & ~seen
            if seen == mask:
                return False
    return True


def has_asteroidal_triple(g: Graph) -> bool:
    n = len(g)
    full = (1 << n) - 1
    components = []
    for forbidden in range(n):
        remaining = full & ~(g[forbidden] | (1 << forbidden))
        ids = [-1] * n
        label = 0
        while remaining:
            start = remaining & -remaining
            component, frontier = 0, start
            while frontier:
                component |= frontier
                neighbors = 0
                for v in vertices(frontier):
                    neighbors |= g[v]
                frontier = neighbors & remaining & ~component
            for v in vertices(component):
                ids[v] = label
            remaining &= ~component
            label += 1
        components.append(ids)
    for a, b, c in combinations(range(n), 3):
        if g[a] & ((1 << b) | (1 << c)) or g[b] >> c & 1:
            continue
        if (components[a][b] == components[a][c] >= 0
                and components[b][a] == components[b][c] >= 0
                and components[c][a] == components[c][b] >= 0):
            return True
    return False


@lru_cache(maxsize=300000)
def is_interval(g: Graph) -> bool:
    return is_chordal(g) and not has_asteroidal_triple(g)


def interval_by_clique_order(g: Graph) -> bool:
    """Independent exact consecutive-ones recognizer for small graphs.

    Enumerates maximal cliques and searches for a consecutive clique ordering.
    It does not call either chordal or asteroidal-triple recognition.
    """
    n = len(g)
    cliques = []
    for mask in range(1, 1 << n):
        if not is_clique(g, mask):
            continue
        if any(is_clique(g, mask | (1 << v)) for v in range(n) if not mask >> v & 1):
            continue
        cliques.append(mask)

    @lru_cache(maxsize=None)
    def order(remaining, previous, closed):
        if not remaining:
            return True
        for i in vertices(remaining):
            current = cliques[i]
            if current & closed:
                continue
            if order(remaining ^ (1 << i), current, closed | (previous & ~current)):
                return True
        return False

    return order((1 << len(cliques)) - 1, 0, 0)


def is_claw_free(g: Graph) -> bool:
    for u in range(len(g)):
        for a, b, c in combinations(vertices(g[u]), 3):
            if not (g[a] & ((1 << b) | (1 << c)) or g[b] >> c & 1):
                return False
    return True


@lru_cache(maxsize=300000)
def is_unit_interval(g: Graph) -> bool:
    return is_interval(g) and is_claw_free(g)


def unit_interval_by_order(g: Graph) -> bool:
    """Exhaustive proper-interval ordering test, independent of the above test."""
    for order in permutations(range(len(g))):
        ok = True
        for i, j, h in combinations(range(len(g)), 3):
            a, b, c = order[i], order[j], order[h]
            if g[a] >> c & 1 and not (g[a] >> b & 1 and g[b] >> c & 1):
                ok = False
                break
        if ok:
            return True
    return False


TARGETS = {"chordal": is_chordal, "interval": is_interval,
           "unit_interval": is_unit_interval}


@dataclass(frozen=True)
class Kernel:
    graph: Graph
    costs: tuple[int, ...]
    budget: int
    retained: tuple[int, ...]
    rejected: bool = False


def kernelize(g: Graph, budget: int, costs=None, target="interval", *, shallow=False) -> Kernel:
    """Reduce twin classes using the split budget and target graph class.

    Chordal/interval: keep K+1 clique twins and K+2 independent twins.
    Unit interval: reject nonisolated independent classes larger than 2K+2,
    or larger than 2*min(K+1, 2) under the shallow restriction;
    keep only one isolated vertex per independent class.
    Retain cheapest vertices and cap costs at K+1. K is numeric budget, not its
    logarithm. A rejected result is a distinguished NO certificate.
    """
    if target not in TARGETS or budget < 0:
        raise ValueError("invalid target or budget")
    costs = tuple(costs if costs is not None else [1] * len(g))
    if len(costs) != len(g) or any(type(c) is not int or c <= 0 for c in costs):
        raise ValueError("positive integer vertex costs required")
    retained = []
    for group, clique in twin_partition(g):
        cap = budget + 1 if clique else budget + 2
        if target == "unit_interval" and not clique:
            cap = 2 * min(budget + 1, 2) if shallow else 2 * budget + 2
            if g[group[0]] and len(group) > cap:
                return Kernel((), (), budget, (), rejected=True)
            if not g[group[0]]:
                cap = 1
        retained.extend(sorted(group, key=lambda v: (costs[v], v))[:cap])
    retained.sort()
    return Kernel(induced(g, retained), tuple(min(costs[v], budget + 1) for v in retained),
                  budget, tuple(retained))


@dataclass(frozen=True)
class Split:
    vertex: int
    left: tuple[int, ...]
    right: tuple[int, ...]


def split_graph(g: Graph, action: Split) -> Graph:
    u = action.vertex
    if not 0 <= u < len(g):
        raise ValueError("invalid split vertex")
    left, right = set(action.left), set(action.right)
    if left | right != set(vertices(g[u])):
        raise ValueError("split must cover exactly the old neighborhood")
    keep = tuple(v for v in range(len(g)) if v != u)
    index = {v: i for i, v in enumerate(keep)}
    adj = list(induced(g, keep)) + [0, 0]
    a, b = len(keep), len(keep) + 1
    for old, new in index.items():
        if old in left:
            adj[new] |= 1 << a
            adj[a] |= 1 << new
        if old in right:
            adj[new] |= 1 << b
            adj[b] |= 1 << new
    return tuple(adj)


def all_splits(g: Graph, u: int, exclusive=False):
    neighbors = tuple(vertices(g[u]))
    for choices in product((1, 2) if exclusive else (1, 2, 3), repeat=len(neighbors)):
        # Children are interchangeable; one representative per swapped pair.
        swapped = tuple(2 if x == 1 else 1 if x == 2 else 3 for x in choices)
        if choices > swapped:
            continue
        yield Split(u, tuple(v for v, c in zip(neighbors, choices) if c & 1),
                    tuple(v for v, c in zip(neighbors, choices) if c & 2))


class SearchLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class Certificate:
    """Resulting graph and original owner of each descendant."""
    graph: Graph
    origins: tuple[int, ...]


def replay_certificate(g: Graph, witness, origins=None) -> Certificate:
    owners = tuple(range(len(g))) if origins is None else tuple(origins)
    if len(owners) != len(g):
        raise ValueError("one owner required for every vertex")
    for action in witness:
        u = action.vertex
        owner = owners[u]
        owners = tuple(x for i, x in enumerate(owners) if i != u) + (owner, owner)
        g = split_graph(g, action)
    return Certificate(g, owners)


def check_certificate(g: Graph, budget: int, certificate: Certificate,
                      target="interval", costs=None, exclusive=False, shallow=False):
    """Check the quotient, fibers, original costs and target directly.

    This checker does not call kernelization, lifting, or split-sequence search.
    """
    h, owners = certificate.graph, certificate.origins
    weights = tuple(costs if costs is not None else [1] * len(g))
    if len(owners) != len(h) or len(weights) != len(g):
        return False
    if any(not 0 <= x < len(g) for x in owners):
        return False
    fibers = [sum(1 << i for i, x in enumerate(owners) if x == v)
              for v in range(len(g))]
    if any(not fiber or not is_independent(h, fiber) for fiber in fibers):
        return False
    if shallow and any(fiber.bit_count() > 2 for fiber in fibers):
        return False
    cost = sum(w * (fiber.bit_count() - 1) for w, fiber in zip(weights, fibers))
    if cost > budget:
        return False
    for u, v in combinations(range(len(g)), 2):
        incidences = sum((h[x] & fibers[v]).bit_count() for x in vertices(fibers[u]))
        if bool(incidences) != bool(g[u] >> v & 1):
            return False
        if exclusive and incidences > 1:
            return False
    return TARGETS[target](h)


def add_clone(g: Graph, representative: int, true_twin=False) -> Graph:
    neighbors = g[representative]
    if true_twin:
        neighbors |= 1 << representative
    return tuple(mask | ((1 << len(g)) if neighbors >> i & 1 else 0)
                 for i, mask in enumerate(g)) + (neighbors,)


def lift_kernel_witness(g: Graph, kernel: Kernel, witness, target="interval",
                        costs=None, exclusive=False, shallow=False) -> Certificate:
    """Construct a certificate for the original graph from a kernel solution.

    Restores classes successively, finding an untouched simplicial representative
    again for each independent class. Original uncapped costs are checked at end.
    """
    if kernel.rejected or witness is None:
        raise ValueError("a successful kernel solution is required")
    if not check_witness(kernel.graph, kernel.budget, witness, target, kernel.costs,
                         exclusive, shallow):
        raise ValueError("invalid kernel witness")
    cert = replay_certificate(kernel.graph, witness, kernel.retained)
    h, owners = cert.graph, cert.origins
    retained = set(kernel.retained)
    for group, clique in twin_partition(g):
        missing = [v for v in group if v not in retained]
        if not missing:
            continue
        candidates = [i for i, owner in enumerate(owners)
                      if owner in group and owners.count(owner) == 1]
        if not clique:
            candidates = [i for i in candidates if is_clique(h, h[i])]
        if not candidates:
            raise AssertionError("no untouched representative for restoration")
        representative = candidates[0]
        for original in missing:
            h = add_clone(h, representative, true_twin=clique)
            owners += (original,)
    result = Certificate(h, owners)
    if not check_certificate(g, kernel.budget, result, target, costs, exclusive, shallow):
        raise AssertionError("lifted certificate violates the original instance")
    return result


def solve_exact(g: Graph, budget: int, target="interval", costs=None,
                exclusive=False, shallow=False, max_nodes=None):
    """Return a sequence of splits or None. A limit raises, never means NO.

    The only pruning is target recognition and, for one remaining affordable
    split, the necessary hereditary condition that G-v belongs to the target.
    Kernelization is NOT called, making this an independent reduction oracle.
    """
    recognize = TARGETS[target]
    costs = tuple(costs if costs is not None else [1] * len(g))
    if budget < 0 or len(costs) != len(g) or any(type(c) is not int or c <= 0 for c in costs):
        raise ValueError("invalid budget or costs")
    counter = [0]

    @lru_cache(maxsize=None)
    def search(state, weights, splittable, remaining):
        counter[0] += 1
        if max_nodes is not None and counter[0] > max_nodes:
            raise SearchLimit(f"search exceeded {max_nodes} states")
        if recognize(state):
            return ()
        feasible = [v for v in range(len(state))
                    if weights[v] <= remaining and splittable[v]]
        if not feasible:
            return None
        for u in feasible:
            rest = tuple(v for v in range(len(state)) if v != u)
            if remaining - weights[u] < min(weights, default=remaining + 1):
                if not recognize(induced(state, rest)):
                    continue
            for action in all_splits(state, u, exclusive):
                next_g = split_graph(state, action)
                next_weights = tuple(weights[v] for v in rest) + (weights[u], weights[u])
                next_allowed = tuple(splittable[v] for v in rest) + (not shallow, not shallow)
                suffix = search(next_g, next_weights, next_allowed, remaining - weights[u])
                if suffix is not None:
                    return (action,) + suffix
        return None

    witness = search(g, costs, (True,) * len(g), budget)
    return witness


def check_witness(g: Graph, budget: int, witness, target="interval", costs=None,
                  exclusive=False, shallow=False):
    weights = tuple(costs if costs is not None else [1] * len(g))
    allowed = (True,) * len(g)
    spent = 0
    for action in witness:
        u = action.vertex
        if exclusive and set(action.left) & set(action.right):
            return False
        if not allowed[u]:
            return False
        spent += weights[u]
        keep = tuple(v for v in range(len(g)) if v != u)
        next_weights = tuple(weights[v] for v in keep) + (weights[u], weights[u])
        next_allowed = tuple(allowed[v] for v in keep) + (not shallow, not shallow)
        g = split_graph(g, action)
        weights, allowed = next_weights, next_allowed
    return spent <= budget and TARGETS[target](g)


def blowup(quotient: Graph, sizes, clique_types) -> Graph:
    classes, index = [], 0
    for size in sizes:
        classes.append(tuple(range(index, index + size)))
        index += size
    edges = []
    for i, group in enumerate(classes):
        if clique_types[i]:
            edges.extend(combinations(group, 2))
        for j in range(i):
            if quotient[i] >> j & 1:
                edges.extend((u, v) for u in group for v in classes[j])
    return graph_from_edges(index, edges)
