"""
Text encoder for converting graph structures to natural language descriptions.
"""

from typing import List, Tuple, Dict, Any


class GraphTextEncoder:
    """
    Encodes graph edge lists into natural language descriptions.
    """
    
    def __init__(self, graph_type: str = "undirected"):
        """
        Initialize graph encoder.
        
        Args:
            graph_type: Type of graph ("undirected", "directed")
        """
        self.graph_type = graph_type
    
    def encode_unweighted_edges(self, edges: List[Tuple[int, int]]) -> str:
        """
        Encode unweighted graph edges to natural language.
        
        Args:
            edges: List of edges
            
        Returns:
            Natural language description
        """
        graph_type_str = "undirected" if self.graph_type == "undirected" else "directed"
        edge_list_str = ", ".join([f"({u}, {v})" for u, v in edges])
        
        description = (
            f"In an {graph_type_str} graph, (i, j) means that node i and node j are "
            f"connected with an edge, and the edges are: [{edge_list_str}]."
        )
        
        return description
    
    def encode_weighted_edges(self, edges: List[Tuple[int, int, float]]) -> str:
        """
        Encode weighted graph edges to natural language.
        
        Args:
            edges: List of edges with weights
            
        Returns:
            Natural language description
        """
        graph_type_str = "undirected" if self.graph_type == "undirected" else "directed"
        edge_list_str = ", ".join([f"({u}, {v}, {w})" for u, v, w in edges])
        
        description = (
            f"In an {graph_type_str} graph, (i, j, w) means that node i and node j are "
            f"connected by an edge with weight w, and the edges are: [{edge_list_str}]."
        )
        
        return description
    
    def encode_labeled_graph(self, edges: List[Tuple[int, int]], node_labels: Dict[int, Any]) -> str:
        """
        Encode graph with node labels for node classification tasks.
        
        Args:
            edges: List of edges
            node_labels: Dictionary mapping nodes to labels
            
        Returns:
            Natural language description
        """
        edge_list_str = ", ".join([f"({u}, {v})" for u, v in edges])
        
        # Create node to label mapping string
        label_mappings = []
        for node, label in node_labels.items():
            label_str = str(label) if label is not None else "?"
            label_mappings.append(f"node {node}: label {label_str}")
        
        label_mapping_str = " | ".join(label_mappings)
        
        description = f"Adjacency list: [{edge_list_str}]\n\nNode to label mapping: {label_mapping_str}"
        
        return description