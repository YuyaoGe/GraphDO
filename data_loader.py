"""
Graph data loader for GraphDO.
Reads from ./graph/{task}.jsonl — one JSONL file per task, 100 pre-sampled graphs each.

JSONL schema per line:
  connectivity:  {"graph_id", "task", "num_nodes", "num_edges", "edges": [[u,v],...], "questions": [[u,v],...]}
  shortest_path: {"graph_id", "task", "num_nodes", "num_edges", "edges": [[u,v,w],...], "query": [src, tgt]}
  others:        {"graph_id", "task", "num_nodes", "num_edges", "edges": [[u,v],...]}
"""

import json
import os
import networkx as nx
from typing import List, Dict, Any, Optional


class GraphDataLoader:
    """
    Loader for the local graph dataset stored as JSONL files in ./graph/.
    Each file ./graph/{task}.jsonl contains 100 pre-sampled, shuffled graphs.
    """

    TASK_CONFIGS = {
        "connectivity": {
            "graph_type": "undirected",
            "weighted": False,
        },
        "cycle": {
            "graph_type": "undirected",
            "weighted": False,
        },
        "shortest_path": {
            "graph_type": "undirected",
            "weighted": True,
        },
        "hamilton": {
            "graph_type": "undirected",
            "weighted": False,
        },
        "topology": {
            "graph_type": "directed",
            "weighted": False,
        },
    }

    def __init__(self, graph_root: str = "./graph"):
        self.graph_root = graph_root

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _jsonl_path(self, task: str) -> str:
        return os.path.join(self.graph_root, f"{task}.jsonl")

    def _build_networkx(self, graph_data: Dict[str, Any], task: str) -> nx.Graph:
        config = self.TASK_CONFIGS[task]
        G = nx.DiGraph() if config["graph_type"] == "directed" else nx.Graph()
        G.add_nodes_from(range(graph_data["num_nodes"]))
        for edge in graph_data["edges"]:
            u, v = edge[0], edge[1]
            w = edge[2] if config["weighted"] and len(edge) >= 3 else 1
            G.add_edge(u, v, weight=w)
        return G

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_graphs_for_task(self, task: str,
                                max_graphs: int = None) -> List[Dict[str, Any]]:
        """
        Load graphs for a task from ./graph/{task}.jsonl.

        Returns a list of dicts, each containing:
          graph_data      – raw dict parsed from JSONL
          networkx_graph  – nx.Graph / nx.DiGraph
          graph_id        – int (from JSONL field)
          filename        – "{task}.jsonl line {graph_id}"
          task, is_weighted, is_directed, file_path
        """
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task}. Available: {list(self.TASK_CONFIGS)}")

        jsonl_path = self._jsonl_path(task)
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

        config = self.TASK_CONFIGS[task]
        is_weighted = config["weighted"]
        is_directed = config["graph_type"] == "directed"

        graphs = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                graph_data = json.loads(line)
                graph_data.setdefault("task", task)

                networkx_graph = self._build_networkx(graph_data, task)
                graphs.append({
                    "file_path": jsonl_path,
                    "filename": f"graph{graph_data['graph_id']}.jsonl",
                    "graph_id": str(graph_data["graph_id"]),
                    "task": task,
                    "graph_data": graph_data,
                    "networkx_graph": networkx_graph,
                    "is_weighted": is_weighted,
                    "is_directed": is_directed,
                })
                if max_graphs and len(graphs) >= max_graphs:
                    break

        return graphs

    def get_task_summary(self) -> Dict[str, Any]:
        """Return graph counts and metadata per task."""
        summary = {}
        for task, config in self.TASK_CONFIGS.items():
            path = self._jsonl_path(task)
            count = 0
            if os.path.exists(path):
                with open(path) as f:
                    count = sum(1 for l in f if l.strip())
            summary[task] = {
                "graph_type": config["graph_type"],
                "weighted": config["weighted"],
                "graph_count": count,
            }
        return summary
