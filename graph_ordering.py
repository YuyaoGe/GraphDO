"""
Graph ordering algorithms for GraphDO.
Implements different methods to order graph edges for description generation.
"""

import networkx as nx
import random
from collections import deque
from typing import List, Tuple, Dict, Any


class GraphOrdering:
    """
    Class implementing different graph edge ordering algorithms.
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize with a NetworkX graph.
        
        Args:
            graph: NetworkX graph object
        """
        self.graph = graph
    
    def random_order(self) -> List[Tuple[int, int]]:
        """Generate random order of edges."""
        edges = list(self.graph.edges())
        random.shuffle(edges)
        return edges
    
    def bfs_order(self, start_node: int = None) -> List[Tuple[int, int]]:
        """
        Generate BFS order of edges via traversal on the line graph (dual graph G*).

        Per the GraphDO paper: "we perform a traversal on the dual graph G* of G
        to ensure that the resulting edge list includes all the edges in G."

        Args:
            start_node: Starting node in G. If None, choose randomly.

        Returns:
            List of edges in BFS-on-line-graph order.
        """
        return self._linegraph_order('bfs', start_node)
    
    def dfs_order(self, start_node: int = None) -> List[Tuple[int, int]]:
        """
        Generate DFS order of edges via traversal on the line graph (dual graph G*).

        Per the GraphDO paper: "we perform a traversal on the dual graph G* of G
        to ensure that the resulting edge list includes all the edges in G."

        Args:
            start_node: Starting node in G. If None, choose randomly.

        Returns:
            List of edges in DFS-on-line-graph order.
        """
        return self._linegraph_order('dfs', start_node)

    def _linegraph_order(self, mode: str, start_node: int = None) -> List[Tuple[int, int]]:
        """
        BFS or DFS traversal on the line graph (dual graph G*) of self.graph.

        In L(G) each edge of G is a node; two edge-nodes are adjacent if the
        corresponding edges share an endpoint.  This guarantees every edge of G
        appears in the output, including cross/back edges that a node-level BFS/DFS
        spanning tree would miss.

        For disconnected graphs the traversal restarts from the first unvisited
        edge-node until all edges are covered (paper: "root node reselected until
        the graph is fully described").

        Args:
            mode:       'bfs' or 'dfs'
            start_node: Starting node in G whose incident edge begins the traversal.
                        If None, chosen randomly.

        Returns:
            All edges of G in line-graph traversal order.
        """
        edges = list(self.graph.edges())
        if not edges:
            return []

        n_edges = len(edges)

        # Build L(G) adjacency via shared-endpoint lookup.
        # For both undirected and directed G we use vertex sharing (undirected
        # adjacency in L(G)), so all edges reachable from any connected component
        # of edges are covered in one traversal component.
        node_to_edge_indices: Dict[int, List[int]] = {}
        for i, (u, v) in enumerate(edges):
            node_to_edge_indices.setdefault(u, []).append(i)
            node_to_edge_indices.setdefault(v, []).append(i)

        lg_adj: List[List[int]] = [[] for _ in range(n_edges)]
        for i, (u, v) in enumerate(edges):
            neighbors: set = set()
            for j in node_to_edge_indices.get(u, []):
                if j != i:
                    neighbors.add(j)
            for j in node_to_edge_indices.get(v, []):
                if j != i:
                    neighbors.add(j)
            lg_adj[i] = list(neighbors)

        # Starting edge-node: first edge incident to start_node.
        if start_node is None:
            start_node = random.choice(list(self.graph.nodes()))
        incident = node_to_edge_indices.get(start_node, [])
        start_idx = incident[0] if incident else 0   # fallback: first edge

        visited = [False] * n_edges
        ordered: List[Tuple[int, int]] = []

        def traverse(seed: int) -> None:
            visited[seed] = True
            if mode == 'bfs':
                q = deque([seed])
                while q:
                    idx = q.popleft()
                    ordered.append(edges[idx])
                    for nb in lg_adj[idx]:
                        if not visited[nb]:
                            visited[nb] = True
                            q.append(nb)
            else:  # dfs
                stk = [seed]
                while stk:
                    idx = stk.pop()
                    ordered.append(edges[idx])
                    for nb in reversed(lg_adj[idx]):
                        if not visited[nb]:
                            visited[nb] = True
                            stk.append(nb)

        traverse(start_idx)

        # Cover edges in disconnected components.
        for i in range(n_edges):
            if not visited[i]:
                traverse(i)

        return ordered
    
    def pagerank_order(self, alpha: float = 0.85) -> List[Tuple[int, int]]:
        """
        Generate PageRank order of edges.
        
        Args:
            alpha: Damping factor for PageRank
            
        Returns:
            List of edges in PageRank order
        """
        # Calculate PageRank scores
        pagerank_scores = nx.pagerank(self.graph, alpha=alpha)
        
        # Sort nodes by PageRank scores in descending order
        sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        
        visited_edges = set()
        ordered_edges = []
        
        # Add edges in order of node importance
        for node, score in sorted_nodes:
            neighbors = list(self.graph.neighbors(node))
            for neighbor in neighbors:
                edge = tuple(sorted([node, neighbor]))  # Normalize edge representation
                if edge not in visited_edges:
                    ordered_edges.append((node, neighbor))
                    visited_edges.add(edge)
        
        return ordered_edges
    
    def personalized_pagerank_order(self, personalization: Dict[int, float], alpha: float = 0.85) -> List[Tuple[int, int]]:
        """
        Generate Personalized PageRank order of edges.
        
        Args:
            personalization: Dictionary mapping nodes to personalization values
            alpha: Damping factor for PageRank
            
        Returns:
            List of edges in Personalized PageRank order
        """
        # Ensure personalization vector is properly normalized
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v/total for k, v in personalization.items()}
        
        # Calculate Personalized PageRank scores
        ppr_scores = nx.pagerank(self.graph, alpha=alpha, personalization=personalization)
        
        # Sort nodes by PPR scores in descending order
        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        
        visited_edges = set()
        ordered_edges = []
        
        # Add edges in order of personalized importance
        for node, score in sorted_nodes:
            neighbors = list(self.graph.neighbors(node))
            for neighbor in neighbors:
                edge = tuple(sorted([node, neighbor]))  # Normalize edge representation
                if edge not in visited_edges:
                    ordered_edges.append((node, neighbor))
                    visited_edges.add(edge)
        
        return ordered_edges


    def _path_based_order(self, source: int, target: int, select: str) -> List[Tuple[int, int]]:
        """
        Base implementation for path-based orderings.
        Enumerates all simple paths between source and target, sorts by total weight,
        then selects the shortest or longest. That path's edges come first; remaining
        edges are appended afterward.

        Args:
            source: Source node
            target: Target node
            select: 'shortest' or 'longest'

        Returns:
            List of edges ordered by the selected path first
        """
        all_paths = list(nx.all_simple_paths(self.graph, source, target))
        if not all_paths:
            return list(self.graph.edges())

        def path_weight(path):
            w = 0
            for i in range(len(path) - 1):
                w += self.graph[path[i]][path[i + 1]].get('weight', 1)
            return w

        sorted_paths = sorted(all_paths, key=path_weight)
        select_path = sorted_paths[0] if select == 'shortest' else sorted_paths[-1]

        ordered_edges = []
        visited = set()  # frozenset of node pairs for undirected dedup

        for i in range(len(select_path) - 1):
            u, v = select_path[i], select_path[i + 1]
            key = frozenset([u, v])
            if key not in visited:
                ordered_edges.append((u, v))
                visited.add(key)

        for u, v in self.graph.edges():
            key = frozenset([u, v])
            if key not in visited:
                ordered_edges.append((u, v))
                visited.add(key)

        return ordered_edges

    def shortest_path_order(self, source: int, target: int) -> List[Tuple[int, int]]:
        """
        Generate Shortest Path Order of edges (paper's 'Deeper Exploration' ordering).
        Edges on the shortest path from source to target appear first.
        Only applicable to shortest_path task (requires source/target).

        Args:
            source: Source node
            target: Target node

        Returns:
            List of edges with shortest-path edges first
        """
        return self._path_based_order(source, target, 'shortest')

    def longest_path_order(self, source: int, target: int) -> List[Tuple[int, int]]:
        """
        Generate Longest Path Order of edges (paper's 'Deeper Exploration' ordering).
        Edges on the longest path from source to target appear first.
        Only applicable to shortest_path task (requires source/target).

        Args:
            source: Source node
            target: Target node

        Returns:
            List of edges with longest-path edges first
        """
        return self._path_based_order(source, target, 'longest')


class PersonalizationStrategy:
    """
    Factory class for creating task-specific personalization vectors.
    """
    
    @staticmethod
    def create_personalization(graph: nx.Graph, task_type: str, task_params: Dict[str, Any] = None) -> Dict[int, float]:
        """
        Create task-specific personalization vectors for PPR as described in the GraphDO paper.
        
        Args:
            graph: NetworkX graph
            task_type: Type of task (connectivity, cycle, shortest_path, etc.)
            task_params: Task-specific parameters
            
        Returns:
            Personalization vector dictionary
        """
        num_nodes = graph.number_of_nodes()
        personalization = {node: 0.0 for node in graph.nodes()}
        
        if task_type == "connectivity":
            # For connectivity: emphasize the two query nodes equally
            if task_params and "query_nodes" in task_params:
                u, v = task_params["query_nodes"]
                personalization[u] = 0.5
                personalization[v] = 0.5
            else:
                # Default: uniform distribution
                for node in graph.nodes():
                    personalization[node] = 1.0 / num_nodes
        
        elif task_type == "cycle":
            # For cycle detection: if cycle exists, emphasize cycle nodes
            if task_params and "cycle_nodes" in task_params:
                cycle_nodes = task_params["cycle_nodes"]
                if cycle_nodes:
                    for node in cycle_nodes:
                        personalization[node] = 1.0 / len(cycle_nodes)
                else:
                    # No cycle: uniform distribution
                    for node in graph.nodes():
                        personalization[node] = 1.0 / num_nodes
            else:
                # Default: uniform distribution
                for node in graph.nodes():
                    personalization[node] = 1.0 / num_nodes
        
        elif task_type == "shortest_path":
            # For shortest path: emphasize nodes on the shortest path
            if task_params and "path_nodes" in task_params:
                path_nodes = task_params["path_nodes"]
                if path_nodes:
                    for node in path_nodes:
                        personalization[node] = 1.0 / len(path_nodes)
                else:
                    # No path: emphasize source and target
                    if "source" in task_params and "target" in task_params:
                        personalization[task_params["source"]] = 0.5
                        personalization[task_params["target"]] = 0.5
            else:
                # Default: uniform distribution
                for node in graph.nodes():
                    personalization[node] = 1.0 / num_nodes
        
        elif task_type == "hamiltonian":
            # For Hamiltonian path: emphasize nodes on the Hamiltonian path
            if task_params and "hamiltonian_nodes" in task_params:
                ham_nodes = task_params["hamiltonian_nodes"]
                if ham_nodes:
                    for node in ham_nodes:
                        personalization[node] = 1.0 / len(ham_nodes)
                else:
                    # No Hamiltonian path: uniform distribution
                    for node in graph.nodes():
                        personalization[node] = 1.0 / num_nodes
            else:
                # Default: uniform distribution
                for node in graph.nodes():
                    personalization[node] = 1.0 / num_nodes
        
        elif task_type == "node_classification":
            # Distance-inverse personalization from the paper:
            #   δ(v) = shortest path distance from test_node
            #   e_v = (Δ - δ(v) + 1) / Σ(Δ - δ(u) + 1)
            if task_params and "test_node" in task_params:
                test_node = task_params["test_node"]
                shortest_paths = nx.single_source_shortest_path_length(graph, test_node)
                all_nodes = list(graph.nodes())
                max_distance = max(shortest_paths.values()) if shortest_paths else 0
                for node in all_nodes:
                    if node in shortest_paths:
                        distance = shortest_paths[node]
                        personalization[node] = (max_distance - distance + 1)
                    else:
                        personalization[node] = 1e-5
                total = sum(personalization.values())
                personalization = {k: v / total for k, v in personalization.items()}
            else:
                for node in graph.nodes():
                    personalization[node] = 1.0 / num_nodes

        elif task_type == "topology":
            # For topological sort: emphasize nodes with in-degree 0
            if isinstance(graph, nx.DiGraph):
                zero_indegree_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]
                if zero_indegree_nodes:
                    for node in zero_indegree_nodes:
                        personalization[node] = 1.0 / len(zero_indegree_nodes)
                else:
                    # Fallback: uniform distribution
                    for node in graph.nodes():
                        personalization[node] = 1.0 / num_nodes
            else:
                # Not a directed graph: uniform distribution
                for node in graph.nodes():
                    personalization[node] = 1.0 / num_nodes
        
        else:
            # Default: uniform distribution
            for node in graph.nodes():
                personalization[node] = 1.0 / num_nodes
        
        return personalization