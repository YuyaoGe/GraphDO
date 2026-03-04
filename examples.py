"""
Few-shot examples for each graph task.

Used by main.py when running few_shot / cot / cot_bag prompt styles.

Format expected by graphdo._construct_prompt():
  {
    "graph_description": str,   # matches text_encoder output format
    "question": str,
    "answer": str,              # concise answer, used by few_shot
    "reasoning": str,           # step-by-step + final answer, used by cot / cot_bag
  }
"""

TASK_EXAMPLES = {

    "connectivity": [
        {
            "graph_description": (
                "In an undirected graph, (i, j) means that node i and node j are "
                "connected with an edge, and the edges are: [(0, 1), (1, 2), (2, 3)]."
            ),
            "question": "Is there a path between node 0 and node 3?",
            "answer": "Yes, the answer is yes.",
            "reasoning": (
                "Starting from node 0: node 0 connects to node 1 via (0, 1). "
                "From node 1, we reach node 2 via (1, 2). "
                "From node 2, we reach node 3 via (2, 3). "
                "The path 0 -> 1 -> 2 -> 3 exists. The answer is yes."
            ),
        },
        {
            "graph_description": (
                "In an undirected graph, (i, j) means that node i and node j are "
                "connected with an edge, and the edges are: [(0, 1), (2, 3)]."
            ),
            "question": "Is there a path between node 0 and node 2?",
            "answer": "No, the answer is no.",
            "reasoning": (
                "From node 0, we can only reach node 1 via edge (0, 1). "
                "Nodes 2 and 3 form a separate component connected by edge (2, 3). "
                "There is no edge linking component {0, 1} to component {2, 3}. "
                "So there is no path from node 0 to node 2. The answer is no."
            ),
        },
    ],

    "cycle": [
        {
            "graph_description": (
                "In an undirected graph, (i, j) means that node i and node j are "
                "connected with an edge, and the edges are: [(0, 1), (1, 2), (2, 0)]."
            ),
            "question": "Is there a cycle in this graph?",
            "answer": "Yes, the answer is yes.",
            "reasoning": (
                "Starting from node 0: follow edge (0, 1) to node 1, "
                "then edge (1, 2) to node 2, "
                "then edge (2, 0) back to node 0. "
                "The sequence 0 -> 1 -> 2 -> 0 forms a cycle. The answer is yes."
            ),
        },
        {
            "graph_description": (
                "In an undirected graph, (i, j) means that node i and node j are "
                "connected with an edge, and the edges are: [(0, 1), (0, 2), (0, 3)]."
            ),
            "question": "Is there a cycle in this graph?",
            "answer": "No, the answer is no.",
            "reasoning": (
                "The graph is a star: node 0 connects to nodes 1, 2, and 3, "
                "but nodes 1, 2, and 3 have no edges among themselves. "
                "From any leaf node (1, 2, or 3), the only neighbor is node 0. "
                "There is no way to return to a visited node without reusing an edge, "
                "so no cycle exists. The answer is no."
            ),
        },
    ],

    "shortest_path": [
        {
            "graph_description": (
                "In an undirected graph, (i, j, w) means that node i and node j are "
                "connected by an edge with weight w, and the edges are: "
                "[(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 3), (2, 3, 7)]."
            ),
            "question": "Give the shortest path from node 0 to node 3.",
            "answer": "0 -> 2 -> 1 -> 3, and the shortest path length is 6.",
            "reasoning": (
                "Candidate paths from node 0 to node 3:\n"
                "  0 -> 1 -> 3: weight 4 + 3 = 7\n"
                "  0 -> 2 -> 1 -> 3: weight 1 + 2 + 3 = 6\n"
                "  0 -> 2 -> 3: weight 1 + 7 = 8\n"
                "The minimum weight is 6 via path 0 -> 2 -> 1 -> 3. "
                "The answer is: 0 -> 2 -> 1 -> 3, and the shortest path length is 6."
            ),
        },
        {
            "graph_description": (
                "In an undirected graph, (i, j, w) means that node i and node j are "
                "connected by an edge with weight w, and the edges are: "
                "[(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 10)]."
            ),
            "question": "Give the shortest path from node 0 to node 3.",
            "answer": "0 -> 1 -> 2 -> 3, and the shortest path length is 3.",
            "reasoning": (
                "Candidate paths from node 0 to node 3:\n"
                "  0 -> 3: weight 10\n"
                "  0 -> 1 -> 2 -> 3: weight 1 + 1 + 1 = 3\n"
                "The minimum weight is 3 via path 0 -> 1 -> 2 -> 3. "
                "The answer is: 0 -> 1 -> 2 -> 3, and the shortest path length is 3."
            ),
        },
    ],

    "hamilton": [
        {
            "graph_description": (
                "In an undirected graph, (i, j) means that node i and node j are "
                "connected with an edge, and the edges are: "
                "[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]."
            ),
            "question": (
                "Is there a path in this graph that visits every node exactly once? "
                "If yes, give the path."
            ),
            "answer": "Yes, the path is: 0 1 2 3.",
            "reasoning": (
                "We need to visit all 4 nodes (0, 1, 2, 3) exactly once. "
                "Try starting from node 0: go to node 1 via (0, 1), "
                "then to node 2 via (1, 2), then to node 3 via (2, 3). "
                "All 4 nodes are visited exactly once with no repeats. "
                "The answer is yes. The path is: 0 1 2 3."
            ),
        },
        {
            "graph_description": (
                "In an undirected graph, (i, j) means that node i and node j are "
                "connected with an edge, and the edges are: [(0, 1), (2, 3)]."
            ),
            "question": (
                "Is there a path in this graph that visits every node exactly once? "
                "If yes, give the path."
            ),
            "answer": "No, the answer is no.",
            "reasoning": (
                "The graph has two disconnected components: {0, 1} and {2, 3}. "
                "A Hamiltonian path must visit all nodes (0, 1, 2, 3) in sequence. "
                "To move from component {0, 1} to component {2, 3} we would need an "
                "edge between them, but none exists. "
                "Therefore no Hamiltonian path exists. The answer is no."
            ),
        },
    ],

    "topology": [
        {
            "graph_description": (
                "In a directed graph, (i, j) means that node i and node j are "
                "connected with an edge, and the edges are: "
                "[(0, 1), (0, 2), (1, 3), (2, 3)]."
            ),
            "question": "Give any topological sorting of the graph.",
            "answer": "0, 1, 2, 3",
            "reasoning": (
                "Use Kahn's algorithm:\n"
                "In-degrees: node 0 -> 0, node 1 -> 1, node 2 -> 1, node 3 -> 2.\n"
                "Queue (in-degree 0): [0].\n"
                "Process 0: output 0; decrement neighbors 1 and 2. Queue: [1, 2].\n"
                "Process 1: output 1; decrement neighbor 3 (in-degree -> 1). Queue: [2].\n"
                "Process 2: output 2; decrement neighbor 3 (in-degree -> 0). Queue: [3].\n"
                "Process 3: output 3.\n"
                "Topological sort: 0, 1, 2, 3."
            ),
        },
        {
            "graph_description": (
                "In a directed graph, (i, j) means that node i and node j are "
                "connected with an edge, and the edges are: [(2, 0), (2, 1), (1, 0)]."
            ),
            "question": "Give any topological sorting of the graph.",
            "answer": "2, 1, 0",
            "reasoning": (
                "Use Kahn's algorithm:\n"
                "In-degrees: node 0 -> 2, node 1 -> 1, node 2 -> 0.\n"
                "Queue (in-degree 0): [2].\n"
                "Process 2: output 2; decrement neighbors 0 (in-degree -> 1) "
                "and 1 (in-degree -> 0). Queue: [1].\n"
                "Process 1: output 1; decrement neighbor 0 (in-degree -> 0). Queue: [0].\n"
                "Process 0: output 0.\n"
                "Topological sort: 2, 1, 0."
            ),
        },
    ],
    "node_classification": [
        {
            "graph_description": (
                "Adjacency list: [(10, 20), (10, 30), (20, 30), (20, 40), (30, 50)]\n\n"
                "Node to label mapping: node 10: label 2 | node 20: label 2 | "
                "node 30: label ? | node 40: label 0 | node 50: label 1"
            ),
            "question": "What is the label of node 30?",
            "answer": "The label of node 30 is 2.",
            "reasoning": (
                "Node 30 is connected to nodes 10 and 20 (both label 2) and node 50 (label 1). "
                "The majority of its neighbors have label 2. "
                "Therefore the label of node 30 is 2."
            ),
        },
        {
            "graph_description": (
                "Adjacency list: [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]\n\n"
                "Node to label mapping: node 1: label 0 | node 2: label 1 | "
                "node 3: label 1 | node 4: label ? | node 5: label 1"
            ),
            "question": "What is the label of node 4?",
            "answer": "The label of node 4 is 1.",
            "reasoning": (
                "Node 4 is connected to nodes 2 (label 1) and 5 (label 1). "
                "Both neighbors have label 1. "
                "Therefore the label of node 4 is 1."
            ),
        },
    ],
}


def get_examples(task: str):
    """Return the list of few-shot examples for the given task.

    Returns an empty list for unknown tasks (zero_shot / zero_shot_cot will still work).
    """
    return TASK_EXAMPLES.get(task, [])
