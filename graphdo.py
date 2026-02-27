"""
GraphDO: Graph Description with Order
Classes for graph description generation and LLM testing.
"""

import json
from typing import Dict, List, Any
from datetime import datetime

# GraphDO class is now focused only on model testing
# For description generation, use description_generator.GraphDescriptionGenerator directly
from model_loader import ModelLoader


class GraphDO:
    """
    Main GraphDO class for LLM testing on graph descriptions.
    For description generation, use GraphDescriptionGenerator directly.
    """
    
    def __init__(self, model_name: str, gpu_id: int = 0):
        """
        Initialize GraphDO for model testing.

        Args:
            model_name: Name of the model to use
            gpu_id: GPU device ID for local models
        """
        self.model_name = model_name
        self.model_loader = ModelLoader(
            model_name=model_name,
            gpu_id=gpu_id,
        )
        print(f"GraphDO initialized with model: {model_name}")
    # === Model Testing Methods ===
    
    def load_descriptions_from_file(self, description_file: str) -> List[Dict[str, Any]]:
        """
        Load graph descriptions from JSON file.
        
        Args:
            description_file: Path to JSON file with descriptions
            
        Returns:
            List of description data
        """
        with open(description_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "descriptions" in data:
            return data["descriptions"]
        else:
            # Assume the file contains the descriptions directly
            return data
    
    def solve_graph_problem(self, graph_description: str, question: str, 
                           prompt_style: str = "zero_shot", examples: List[Dict] = None,
                           max_tokens: int = 400, temperature: float = 0.0) -> str:
        """
        Solve a graph problem using LLM.
        
        Args:
            graph_description: Natural language graph description
            question: Question about the graph
            prompt_style: Style of prompting ("zero_shot", "zero_shot_cot", "few_shot", "cot")
            examples: Few-shot examples for prompting
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Model's response
        """
        if not self.model_loader:
            raise ValueError("Model not initialized. Cannot solve graph problems.")
        
        # Construct prompt based on style
        prompt = self._construct_prompt(graph_description, question, prompt_style, examples)
        
        # Generate response
        response = self.model_loader.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response
    
    def _construct_prompt(self, graph_description: str, question: str, 
                         prompt_style: str, examples: List[Dict] = None) -> str:
        """
        Construct prompt based on the specified style.
        
        Args:
            graph_description: Graph description
            question: Question about the graph
            prompt_style: Prompting style
            examples: Few-shot examples
            
        Returns:
            Constructed prompt
        """
        if prompt_style == "zero_shot":
            return f"Graph: {graph_description}\\nQuestion: {question}\\nAnswer:"
        
        elif prompt_style == "zero_shot_cot":
            return f"Graph: {graph_description}\\nQuestion: {question} Let's think step by step.\\nAnswer:"
        
        elif prompt_style == "few_shot":
            if not examples:
                raise ValueError("Examples required for few-shot prompting")
            
            prompt_parts = []
            for example in examples:
                prompt_parts.append(
                    f"Graph: {example['graph_description']}\\n"
                    f"Question: {example['question']}\\n"
                    f"Answer: {example['answer']}"
                )
            
            prompt_parts.append(
                f"Graph: {graph_description}\\nQuestion: {question}\\nAnswer:"
            )
            
            return "\\n\\n".join(prompt_parts)
        
        elif prompt_style == "cot":
            if not examples:
                raise ValueError("Examples with reasoning required for CoT prompting")
            
            prompt_parts = []
            for example in examples:
                prompt_parts.append(
                    f"Graph: {example['graph_description']}\\n"
                    f"Question: {example['question']}\\n"
                    f"Answer: {example['reasoning']}"
                )
            
            prompt_parts.append(
                f"Graph: {graph_description}\\nQuestion: {question}\\nAnswer:"
            )
            
            return "\\n\\n".join(prompt_parts)
        
        elif prompt_style == "cot_bag":
            if not examples:
                raise ValueError("Examples with reasoning required for CoT-BAG prompting")
            
            prompt_parts = []
            for example in examples:
                prompt_parts.append(
                    f"Graph: {example['graph_description']}\\n"
                    f"Question: {example['question']}\\n"
                    f"Answer: {example['reasoning']}"
                )
            
            prompt_parts.append(
                f"Graph: {graph_description}\\nQuestion: {question}\\n"
                f"Let's construct a graph with the nodes and edges first\\nAnswer:"
            )
            
            return "\\n\\n".join(prompt_parts)
        
        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")

