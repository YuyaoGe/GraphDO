"""
Graph generator for GraphDO tasks.

Follows the paper's graph generation methodology (ACL 2025):
  - Erdős-Rényi (ER) model: n ∈ [5, 15], connection probability p = 0.3
  - Generated graphs are filtered to ensure each instance has a valid, well-defined solution
  - Reference: Appendix §GraphDO / Graph Generation

Usage:
  python graph_generator.py --num_graphs 280          # paper-scale
  python graph_generator.py --tasks connectivity      # single task
  python graph_generator.py --seed 123 --num_graphs 100 --output_dir ./my_graphs

Output: {output_dir}/{task}.jsonl  (overwrites existing files)
"""

import argparse
import json
import os
import random
import networkx as nx
from typing import Dict, Any, List, Optional


TASKS = ["connectivity", "cycle", "shortest_path", "hamilton", "topology",
         "node_classification"]

# Paper parameters (Appendix §GraphDO)
N_MIN = 5
N_MAX = 15
P_ER = 0.3   # Erdős-Rényi connection probability


# ---------------------------------------------------------------------------
# ER graph primitives
# ---------------------------------------------------------------------------

def _er_undirected(n: int, p: float, rng: random.Random) -> nx.Graph:
    """Erdős-Rényi undirected graph G(n, p)."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                G.add_edge(i, j)
    return G


def _er_dag(n: int, p: float, rng: random.Random) -> nx.DiGraph:
    """
    ER directed acyclic graph.
    A random permutation defines the topological order;
    directed edges go from lower-rank to higher-rank nodes with probability p.
    """
    perm = list(range(n))
    rng.shuffle(perm)          # perm[i] = node at topological rank i
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                G.add_edge(perm[i], perm[j])
    return G


# ---------------------------------------------------------------------------
# Hamilton path check (exact backtracking, tractable for n ≤ N_MAX = 15)
# ---------------------------------------------------------------------------

def _has_ham_path(G: nx.Graph) -> bool:
    """Return True if G has a Hamiltonian path (visits every node exactly once)."""
    n = G.number_of_nodes()
    if n == 0:
        return False
    nodes = list(G.nodes())

    def backtrack(path: List[int], visited: set) -> bool:
        if len(path) == n:
            return True
        for nb in G.neighbors(path[-1]):
            if nb not in visited:
                visited.add(nb)
                path.append(nb)
                if backtrack(path, visited):
                    return True
                path.pop()
                visited.remove(nb)
        return False

    for start in nodes:
        if backtrack([start], {start}):
            return True
    return False


# ---------------------------------------------------------------------------
# Task-specific generators (each retries until a valid instance is found)
# ---------------------------------------------------------------------------

def gen_connectivity(rng: random.Random) -> Dict[str, Any]:
    """
    ER undirected graph with one connectivity query (u, v).

    Filter: graph must have at least 1 edge.
    The natural ER distribution gives a mix of connected and disconnected query answers
    (small n graphs are often disconnected; larger n graphs are usually connected).
    """
    for _ in range(200):
        n = rng.randint(N_MIN, N_MAX)
        G = _er_undirected(n, P_ER, rng)
        if G.number_of_edges() == 0:
            continue
        # Pick one random query pair
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        while v == u:
            v = rng.randint(0, n - 1)
        edges = [[a, b] for a, b in G.edges()]
        return {"num_nodes": n, "num_edges": len(edges), "edges": edges,
                "questions": [[u, v]]}
    raise RuntimeError("gen_connectivity: could not generate a valid graph")


def gen_cycle(rng: random.Random) -> Dict[str, Any]:
    """
    ER undirected graph for cycle detection.
    Filter: graph must have at least 1 edge.
    """
    for _ in range(200):
        n = rng.randint(N_MIN, N_MAX)
        G = _er_undirected(n, P_ER, rng)
        if G.number_of_edges() == 0:
            continue
        edges = [[a, b] for a, b in G.edges()]
        return {"num_nodes": n, "num_edges": len(edges), "edges": edges}
    raise RuntimeError("gen_cycle: could not generate a valid graph")


def gen_shortest_path(rng: random.Random) -> Dict[str, Any]:
    """
    ER weighted undirected graph with one source→target query.
    Filter: source and target must be connected (path exists).
    Weights are random integers in [1, 10].
    """
    for _ in range(500):
        n = rng.randint(N_MIN, N_MAX)
        G = _er_undirected(n, P_ER, rng)
        if G.number_of_edges() == 0:
            continue
        # Assign random integer weights
        for a, b in G.edges():
            G[a][b]["weight"] = rng.randint(1, 10)
        # Pick a connected query pair
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        while v == u:
            v = rng.randint(0, n - 1)
        if not nx.has_path(G, u, v):
            continue
        edges = [[a, b, G[a][b]["weight"]] for a, b in G.edges()]
        return {"num_nodes": n, "num_edges": len(edges), "edges": edges,
                "query": [u, v]}
    raise RuntimeError("gen_shortest_path: could not generate a valid graph")


def gen_hamilton(rng: random.Random) -> Dict[str, Any]:
    """
    ER undirected graph for Hamilton path detection.
    Filter: graph must have at least 1 edge.
    Note: Hamiltonian path existence is checked lazily (not pre-enforced);
    the evaluator computes the ground truth at test time.
    Node count is capped at 12 to keep backtracking tractable.
    """
    n_max_ham = min(12, N_MAX)   # backtracking is NP-hard; 12 is safe in practice
    for _ in range(200):
        n = rng.randint(N_MIN, n_max_ham)
        G = _er_undirected(n, P_ER, rng)
        if G.number_of_edges() == 0:
            continue
        edges = [[a, b] for a, b in G.edges()]
        return {"num_nodes": n, "num_edges": len(edges), "edges": edges}
    raise RuntimeError("gen_hamilton: could not generate a valid graph")


def gen_topology(rng: random.Random) -> Dict[str, Any]:
    """
    ER DAG using perm-based forward edges (guaranteed acyclic by construction).
    Filter: graph must have at least 1 edge.
    """
    for _ in range(200):
        n = rng.randint(N_MIN, N_MAX)
        G = _er_dag(n, P_ER, rng)
        if G.number_of_edges() == 0:
            continue
        edges = [[a, b] for a, b in G.edges()]
        return {"num_nodes": n, "num_edges": len(edges), "edges": edges}
    raise RuntimeError("gen_topology: could not generate a valid graph")


# ---------------------------------------------------------------------------
# Node classification (T6) — Planetoid datasets with ego / forest fire sampling
# ---------------------------------------------------------------------------

# Lazily-loaded cache so we only load each Planetoid dataset once per process.
_PLANETOID_CACHE: Dict[str, Any] = {}

NC_DATASETS = ["cora", "citeseer", "pubmed"]
NC_SAMPLINGS = ["ego", "forest_fire"]
NC_SAMPLE_SIZE = 50   # Paper: 50-node subgraphs
NC_EGO_HOPS = 3       # Paper: 3-hop ego graphs


def _load_planetoid(name: str):
    """Load a Planetoid dataset (cora/citeseer/pubmed) via torch_geometric, cached."""
    if name in _PLANETOID_CACHE:
        return _PLANETOID_CACHE[name]
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_networkx
    pretty = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "Pubmed"}[name]
    dataset = Planetoid(root="./data", name=pretty)
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    _PLANETOID_CACHE[name] = (data, G)
    return data, G


def _ego_sample(G, data, rng: random.Random, sample_size: int = NC_SAMPLE_SIZE,
                hops: int = NC_EGO_HOPS) -> Optional[Dict[str, Any]]:
    """Sample one ego subgraph with exactly `sample_size` nodes (paper requirement)."""
    node_list = list(G.nodes())
    rng.shuffle(node_list)
    for ego_node in node_list:
        sub = nx.ego_graph(G, ego_node, radius=hops)
        if sub.number_of_nodes() == sample_size:
            # Pick a random test node
            test_node = rng.choice(list(sub.nodes()))
            edges = [[int(u), int(v)] for u, v in sub.edges()]
            node_labels = {int(n): int(data.y[n].item()) for n in sub.nodes()}
            return {
                "dataset": "unknown",  # filled by caller
                "sampling": "ego",
                "num_nodes": sub.number_of_nodes(),
                "num_edges": sub.number_of_edges(),
                "edges": edges,
                "node_labels": node_labels,
                "test_node": int(test_node),
            }
    return None


def _forest_fire_sample(G, data, rng: random.Random,
                        sample_size: int = NC_SAMPLE_SIZE) -> Optional[Dict[str, Any]]:
    """Sample one forest-fire subgraph with `sample_size` nodes (paper: p=0.3)."""
    import time
    from littleballoffur import ForestFireSampler
    seed = rng.randint(0, 2**31)
    sampler = ForestFireSampler(number_of_nodes=sample_size, p=0.3, seed=seed)
    try:
        sub = sampler.sample(G)
    except Exception:
        return None
    if sub.number_of_nodes() != sample_size:
        return None
    test_node = rng.choice(list(sub.nodes()))
    edges = [[int(u), int(v)] for u, v in sub.edges()]
    node_labels = {int(n): int(data.y[n].item()) for n in sub.nodes()}
    return {
        "dataset": "unknown",
        "sampling": "forest_fire",
        "num_nodes": sub.number_of_nodes(),
        "num_edges": sub.number_of_edges(),
        "edges": edges,
        "node_labels": node_labels,
        "test_node": int(test_node),
    }


def gen_node_classification(num_graphs: int, rng: random.Random,
                            datasets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Generate node classification subgraphs for all datasets × samplings.

    Returns a flat list of graph dicts (num_graphs per dataset per sampling).
    """
    if datasets is None:
        datasets = NC_DATASETS
    graphs: List[Dict[str, Any]] = []
    graph_id = 0
    for ds_name in datasets:
        print(f"  Loading {ds_name}...")
        data, G = _load_planetoid(ds_name)
        for sampling in NC_SAMPLINGS:
            sampler_fn = _ego_sample if sampling == "ego" else _forest_fire_sample
            count = 0
            attempts = 0
            while count < num_graphs and attempts < num_graphs * 50:
                attempts += 1
                result = sampler_fn(G, data, rng)
                if result is None:
                    continue
                result["dataset"] = ds_name
                result["task"] = "node_classification"
                result["graph_id"] = graph_id
                graphs.append(result)
                graph_id += 1
                count += 1
            if count < num_graphs:
                print(f"    WARNING: only generated {count}/{num_graphs} for {ds_name}/{sampling}")
            else:
                print(f"    {ds_name}/{sampling}: {count} graphs")
    return graphs


# ---------------------------------------------------------------------------
# Task dispatch and I/O
# ---------------------------------------------------------------------------

GENERATORS = {
    "connectivity": gen_connectivity,
    "cycle": gen_cycle,
    "shortest_path": gen_shortest_path,
    "hamilton": gen_hamilton,
    "topology": gen_topology,
}


def generate_task(task: str, num_graphs: int, rng: random.Random,
                  datasets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if task == "node_classification":
        return gen_node_classification(num_graphs, rng, datasets)
    fn = GENERATORS[task]
    graphs = []
    for i in range(num_graphs):
        g = fn(rng)
        g["task"] = task
        g["graph_id"] = i
        graphs.append(g)
    return graphs


def save_jsonl(graphs: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for g in graphs:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate graph datasets for GraphDO tasks (ER model, paper-consistent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Graph generation follows the ACL 2025 paper:
  - Erdős-Rényi model: n ∈ [5, 15], p = 0.3
  - Filtered to ensure each instance has a valid, well-defined solution

Examples:
  python graph_generator.py --num_graphs 280          # paper-scale (~1400 graphs)
  python graph_generator.py --tasks connectivity      # single task only
  python graph_generator.py --seed 0 --num_graphs 50 --output_dir ./my_graphs

After generation, run:
  python generate.py --graph_root ./my_graphs         # build descriptions
  python main.py --model Llama-2-7b-chat-hf ...       # run experiments
        """,
    )
    parser.add_argument(
        "--tasks", nargs="+", default=TASKS, choices=TASKS,
        help="Tasks to generate (default: all 6)",
    )
    parser.add_argument(
        "--num_graphs", type=int, default=10,
        help="Graphs per task (default: 10; use 280 for paper-scale)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=NC_DATASETS, choices=NC_DATASETS,
        help="Datasets for node_classification (default: cora citeseer pubmed)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./graph",
        help="Output directory (default: ./graph)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(
        f"Generating graphs: ER(n=[{N_MIN},{N_MAX}], p={P_ER}), "
        f"seed={args.seed}, {args.num_graphs} graphs/task → {args.output_dir}/"
    )

    for task in args.tasks:
        ds = args.datasets if task == "node_classification" else None
        graphs = generate_task(task, args.num_graphs, rng, datasets=ds)
        path = os.path.join(args.output_dir, f"{task}.jsonl")
        save_jsonl(graphs, path)
        print(f"  {task}: {len(graphs)} graphs → {path}")

    print("Done.")


if __name__ == "__main__":
    main()
