"""
Answer evaluators for GraphDO graph reasoning tasks.
Uses NetworkX-based ground truth computation.
"""

import re
import networkx as nx
from typing import Dict, Any, Optional


def build_graph(graph_data: Dict[str, Any], is_weighted: bool, is_directed: bool) -> nx.Graph:
    """Reconstruct NetworkX graph from stored graph_data dict."""
    G = nx.DiGraph() if is_directed else nx.Graph()
    G.add_nodes_from(range(graph_data['num_nodes']))
    for edge in graph_data['edges']:
        u, v = edge[0], edge[1]
        if is_weighted and len(edge) >= 3:
            G.add_edge(u, v, weight=edge[2])
        else:
            G.add_edge(u, v, weight=1)
    return G


def evaluate_connectivity(response: str, graph_data: Dict[str, Any], query_params: Dict[str, Any]) -> int:
    """
    Evaluate connectivity answer.
    Ground truth: nx.has_path(G, u, v)
    Returns 1 if correct, 0 if wrong.
    """
    u, v = query_params['source'], query_params['target']
    G = build_graph(graph_data, is_weighted=False, is_directed=False)
    expected_yes = nx.has_path(G, u, v)

    ans = response.lower()
    answered_yes = (
        "the answer is yes" in ans or
        f"there is a path between node {u} and node {v}" in ans
    )
    return 1 if (expected_yes == answered_yes) else 0


def evaluate_cycle(response: str, graph_data: Dict[str, Any]) -> int:
    """
    Evaluate cycle detection answer.
    Ground truth: whether the graph has a cycle (via nx.cycle_basis).
    Returns 1 if correct, 0 if wrong.
    """
    G = build_graph(graph_data, is_weighted=False, is_directed=False)
    has_cycle = len(nx.cycle_basis(G)) > 0

    ans = response.lower()
    p_no = ans.find("there is no cycle")
    p_yes = ans.find("there is a cycle")

    # Neither keyword found
    if p_no == -1 and p_yes == -1:
        return 0

    p_no = float('inf') if p_no == -1 else p_no
    p_yes = float('inf') if p_yes == -1 else p_yes

    # Whichever phrase appears first determines the answer
    answered_yes = p_yes < p_no
    return 1 if (has_cycle == answered_yes) else 0


def evaluate_shortest_path(response: str, graph_data: Dict[str, Any], query_params: Dict[str, Any]) -> int:
    """
    Evaluate shortest path answer.
    Ground truth: nx.shortest_path_length(G, source, target, weight='weight')
    Returns 1 if the path is valid and has the correct total weight, 0 otherwise.
    """
    source, target = query_params['source'], query_params['target']
    G = build_graph(graph_data, is_weighted=True, is_directed=False)

    if not nx.has_path(G, source, target):
        return 0
    expected_length = nx.shortest_path_length(G, source, target, weight='weight')

    ans = response.lower()
    mode_str = f"the shortest path from node {source} to node {target}"
    pos = ans.find(mode_str)
    if pos == -1:
        return 0

    pos += len(mode_str) + 1
    solution = []
    num, flag, done = 0, 0, False
    for i in range(pos, len(ans)):
        if ans[i].isdigit():
            num = num * 10 + int(ans[i])
            flag = 1
        else:
            if flag == 1:
                solution.append(num)
                if num == target:
                    done = True
                    break
                flag = 0
            num = 0
    if flag == 1 and not done:
        solution.append(num)

    if not solution:
        return 0

    total_weight = 0
    for i in range(len(solution) - 1):
        if not G.has_edge(solution[i], solution[i + 1]):
            return 0
        total_weight += G[solution[i]][solution[i + 1]]['weight']

    return 1 if total_weight == expected_length else 0


def evaluate_hamilton(response: str, graph_data: Dict[str, Any]) -> int:
    """
    Evaluate Hamiltonian path answer.
    Validates purely against graph structure (path length == n, edges exist, no repeated nodes).
    Returns 1 if valid, 0 if wrong.
    """
    G = build_graph(graph_data, is_weighted=False, is_directed=False)
    n = G.number_of_nodes()

    ans = response.lower()
    pos_no = ans.find("no ")
    pos_path = ans.find("the path")

    if pos_path == -1:
        return 0
    if pos_no != -1 and pos_no < pos_path:
        return 0

    solution = []
    num, flag = 0, 0
    for i in range(pos_path, len(ans)):
        if ans[i].isdigit():
            num = num * 10 + int(ans[i])
            flag = 1
        else:
            if flag == 1:
                solution.append(num)
                if len(solution) == n:
                    break
                flag = 0
            num = 0
    if flag == 1 and len(solution) < n:
        solution.append(num)

    if len(solution) != n:
        return 0

    # Check all edges exist
    for i in range(len(solution) - 1):
        if not G.has_edge(solution[i], solution[i + 1]):
            return 0

    # Check no repeated nodes
    for i in range(len(solution) - 1):
        for j in range(i + 1, len(solution)):
            if solution[i] == solution[j]:
                return 0

    return 1


def evaluate_topology(response: str, graph_data: Dict[str, Any]) -> int:
    """
    Evaluate topological sort answer.
    Validates purely against DAG structure using in-degree reduction.
    Returns 1 if valid, 0 if wrong.
    """
    G = build_graph(graph_data, is_weighted=False, is_directed=True)
    n = G.number_of_nodes()

    ans = response.lower()

    def parse_solution(start_pos: int):
        solution = []
        num, flag = 0, 0
        for i in range(start_pos, len(ans)):
            if ans[i].isdigit():
                num = num * 10 + int(ans[i])
                flag = 1
            else:
                if flag == 1:
                    solution.append(num)
                    if len(solution) == n:
                        break
                    flag = 0
                num = 0
        if flag == 1 and len(solution) < n:
            solution.append(num)
        return solution

    def check_topo(solution) -> int:
        if not solution:
            return 0
        deg = {node: G.in_degree(node) for node in G.nodes()}
        for node in solution:
            if node not in deg or deg[node] > 0:
                return 0
            for neighbor in G[node]:
                deg[neighbor] -= 1
        return 1 if all(d == 0 for d in deg.values()) else 0

    # Find keyword position
    pos = ans.find("solution")
    if pos == -1:
        p1 = ans.find("yes")
        p2 = ans.find("in the following order")
        if p1 == -1 and p2 == -1:
            pos = -1
        elif p1 == -1:
            pos = p2
        elif p2 == -1:
            pos = p1
        else:
            pos = min(p1, p2)

    flag1 = check_topo(parse_solution(pos)) if pos != -1 else 0
    flag2 = check_topo(parse_solution(0))
    return 1 if (flag1 or flag2) else 0


def evaluate_node_classification(response: str, graph_data: Dict[str, Any],
                                  query_params: Dict[str, Any]) -> int:
    """
    Evaluate node classification answer.
    Ground truth: graph_data["node_labels"][test_node]
    Parses the response for a predicted label (integer).
    Returns 1 if correct, 0 if wrong.
    """
    test_node = query_params.get("test_node")
    if test_node is None:
        return 0
    node_labels = graph_data.get("node_labels", {})
    expected = int(node_labels.get(str(test_node), node_labels.get(test_node, -1)))

    ans = response.strip()

    # Try common patterns: "label of node X is Y", "label Y", "the answer is Y"
    patterns = [
        rf'label\s+of\s+node\s+{test_node}\s*(?:is|=|:)\s*(\d+)',
        rf'node\s+{test_node}\s*(?:is|=|:)\s*(?:label\s+)?(\d+)',
        r'the\s+(?:predicted\s+)?label\s+is\s+(\d+)',
        r'the\s+answer\s+is\s+(\d+)',
        r'label\s*(?:is|=|:)\s*(\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, ans, re.IGNORECASE)
        if m:
            predicted = int(m.group(1))
            return 1 if predicted == expected else 0

    # Fallback: find the last integer in the response
    digits = re.findall(r'\d+', ans)
    if digits:
        predicted = int(digits[-1])
        return 1 if predicted == expected else 0

    return 0


def evaluate_answer(task: str, response: str, graph_data: Dict[str, Any],
                    is_weighted: bool, is_directed: bool,
                    query_params: Optional[Dict[str, Any]] = None) -> int:
    """
    Dispatch to task-specific evaluator.
    Returns 1 if correct, 0 if wrong, -1 if task unknown.
    """
    if task == "connectivity":
        return evaluate_connectivity(response, graph_data, query_params or {})
    elif task == "cycle":
        return evaluate_cycle(response, graph_data)
    elif task == "shortest_path":
        return evaluate_shortest_path(response, graph_data, query_params or {})
    elif task == "hamilton":
        return evaluate_hamilton(response, graph_data)
    elif task == "topology":
        return evaluate_topology(response, graph_data)
    elif task == "node_classification":
        return evaluate_node_classification(response, graph_data, query_params or {})
    else:
        return -1
