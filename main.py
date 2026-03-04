"""
Script to run model tests using pre-generated graph descriptions.
This script loads JSON files containing graph descriptions and tests LLMs on them.
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from graphdo import GraphDO
from info import list_available_models
from evaluator import evaluate_answer
from examples import get_examples


def extract_questions_from_descriptions(descriptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract questions from graph description data.
    
    Args:
        descriptions: List of graph description data
        
    Returns:
        List of question data for testing
    """
    questions = []
    
    for desc_data in descriptions:
        graph_data = desc_data.get('graph_data', {})
        task = desc_data.get('task', 'unknown')
        
        # Extract questions based on task type
        is_weighted = desc_data.get('is_weighted', False)
        is_directed = desc_data.get('is_directed', False)

        if task == "connectivity":
            # Multiple questions per graph
            if 'questions' in graph_data:
                for i, question_pair in enumerate(graph_data['questions']):
                    u, v = question_pair
                    question_text = f"Is there a path between node {u} and node {v}?"

                    questions.append({
                        'graph_id': desc_data['graph_id'],
                        'filename': desc_data['filename'],
                        'task': task,
                        'question_id': f"{desc_data['graph_id']}_q{i}",
                        'question': question_text,
                        'query_params': {'source': u, 'target': v},
                        'graph_data': graph_data,
                        'is_weighted': is_weighted,
                        'is_directed': is_directed,
                        'descriptions': desc_data['descriptions']
                    })

        elif task in ["cycle", "hamilton", "topology"]:
            # Single question per graph
            if task == "cycle":
                question_text = "Is there a cycle in this graph?"
            elif task == "hamilton":
                question_text = "Is there a path in this graph that visits every node exactly once? If yes, give the path."
            elif task == "topology":
                question_text = "Give any topological sorting of the graph."

            questions.append({
                'graph_id': desc_data['graph_id'],
                'filename': desc_data['filename'],
                'task': task,
                'question_id': desc_data['graph_id'],
                'question': question_text,
                'query_params': {},
                'graph_data': graph_data,
                'is_weighted': is_weighted,
                'is_directed': is_directed,
                'descriptions': desc_data['descriptions']
            })

        elif task == "shortest_path":
            # Single question per graph
            if 'query' in graph_data:
                source, target = graph_data['query']
                question_text = f"Give the shortest path from node {source} to node {target}."

                questions.append({
                    'graph_id': desc_data['graph_id'],
                    'filename': desc_data['filename'],
                    'task': task,
                    'question_id': desc_data['graph_id'],
                    'question': question_text,
                    'query_params': {'source': source, 'target': target},
                    'graph_data': graph_data,
                    'is_weighted': is_weighted,
                    'is_directed': is_directed,
                    'descriptions': desc_data['descriptions']
                })

        elif task == "node_classification":
            if 'test_node' in graph_data:
                test_node = graph_data['test_node']
                question_text = f"What is the label of node {test_node}?"

                questions.append({
                    'graph_id': desc_data['graph_id'],
                    'filename': desc_data['filename'],
                    'task': task,
                    'question_id': desc_data['graph_id'],
                    'question': question_text,
                    'query_params': {'test_node': test_node},
                    'graph_data': graph_data,
                    'is_weighted': is_weighted,
                    'is_directed': is_directed,
                    'descriptions': desc_data['descriptions']
                })
    
    return questions


def run_model_test(model_name: str, description_files: List[str],
                  orders: List[str] = None, prompt_styles: List[str] = None,
                  output_dir: str = "model_test_results", max_questions: int = None,
                  gpu_id: int = 0) -> Dict[str, Any]:
    """
    Run model tests on graph descriptions.

    Args:
        model_name: Name of the model to test
        description_files: List of JSON files containing descriptions
        orders: List of orders to test
        prompt_styles: List of prompt styles to test
        output_dir: Output directory for results
        max_questions: Maximum questions to test
        gpu_id: GPU device ID

    Returns:
        Test results dictionary
    """
    if orders is None:
        orders = ["random", "bfs", "dfs", "pagerank", "ppr"]
    
    if prompt_styles is None:
        prompt_styles = ["zero_shot", "zero_shot_cot"]
    
    # Initialize GraphDO with model
    print(f"Initializing GraphDO with model: {model_name}")
    graphdo = GraphDO(
        model_name=model_name,
        gpu_id=gpu_id,
    )
    
    if not graphdo.model_loader:
        raise ValueError(f"Failed to initialize model: {model_name}")
    
    # Load all descriptions
    all_questions = []
    for desc_file in description_files:
        print(f"Loading descriptions from {desc_file}")
        descriptions = graphdo.load_descriptions_from_file(desc_file)
        questions = extract_questions_from_descriptions(descriptions)
        all_questions.extend(questions)
    
    print(f"Total questions loaded: {len(all_questions)}")
    
    if max_questions:
        all_questions = all_questions[:max_questions]
        print(f"Limited to {len(all_questions)} questions")
    
    # Run tests
    results = {
        "metadata": {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "description_files": description_files,
            "orders_tested": orders,
            "prompt_styles_tested": prompt_styles,
            "total_questions": len(all_questions)
        },
        "results": []
    }

    # Accuracy counters: {task: {order: {prompt_style: {'correct': int, 'total': int}}}}
    tasks_in_data = list(dict.fromkeys(q['task'] for q in all_questions))
    accuracy_counters: Dict[str, Any] = {
        task: {
            order: {ps: {'correct': 0, 'total': 0} for ps in prompt_styles}
            for order in orders
        }
        for task in tasks_in_data
    }

    total_tests = len(all_questions) * len(orders) * len(prompt_styles)
    current_test = 0

    for question_data in all_questions:
        question_result = {
            "question_id": question_data['question_id'],
            "graph_id": question_data['graph_id'],
            "task": question_data['task'],
            "question": question_data['question'],
            "order_results": {}
        }

        for order in orders:
            if order not in question_data['descriptions']:
                print(f"Order {order} not available for question {question_data['question_id']}")
                continue

            desc_data = question_data['descriptions'][order]
            if 'description' not in desc_data:
                print(f"Description not available for {order} order in question {question_data['question_id']}")
                continue

            graph_description = desc_data['description']
            order_result = {"prompt_results": {}}

            for prompt_style in prompt_styles:
                current_test += 1
                print(f"Test {current_test}/{total_tests}: {question_data['question_id']} - {order} - {prompt_style}")

                try:
                    response = graphdo.solve_graph_problem(
                        graph_description=graph_description,
                        question=question_data['question'],
                        prompt_style=prompt_style,
                        examples=get_examples(question_data['task']),
                        max_tokens=400,
                        temperature=0.0
                    )

                    # Evaluate answer
                    correct = evaluate_answer(
                        task=question_data['task'],
                        response=response,
                        graph_data=question_data.get('graph_data', {}),
                        is_weighted=question_data.get('is_weighted', False),
                        is_directed=question_data.get('is_directed', False),
                        query_params=question_data.get('query_params')
                    )

                    task = question_data['task']
                    if correct != -1 and task in accuracy_counters:
                        accuracy_counters[task][order][prompt_style]['total'] += 1
                        accuracy_counters[task][order][prompt_style]['correct'] += correct

                    order_result["prompt_results"][prompt_style] = {
                        "response": response,
                        "correct": correct,
                        "status": "success"
                    }

                except Exception as e:
                    print(f"Error: {e}")
                    order_result["prompt_results"][prompt_style] = {
                        "error": str(e),
                        "status": "error"
                    }

            question_result["order_results"][order] = order_result

        results["results"].append(question_result)

    results["accuracy"] = accuracy_counters
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"test_results_{model_name}_{timestamp}.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\\nTest results saved to {results_file}")
    
    # Generate summary
    generate_test_summary(results, output_dir, timestamp)
    
    return results


def generate_test_summary(results: Dict[str, Any], output_dir: str, timestamp: str):
    """Generate a summary of test results including per-task accuracy."""
    orders = results["metadata"]["orders_tested"]
    prompt_styles = results["metadata"]["prompt_styles_tested"]
    accuracy_counters = results.get("accuracy", {})
    tasks = sorted(accuracy_counters.keys())

    summary = {
        "model_name": results["metadata"]["model_name"],
        "timestamp": results["metadata"]["timestamp"],
        "total_questions": results["metadata"]["total_questions"],
        "accuracy": {}
    }

    for task in tasks:
        summary["accuracy"][task] = {}
        for order in orders:
            summary["accuracy"][task][order] = {}
            for prompt_style in prompt_styles:
                stats = accuracy_counters.get(task, {}).get(order, {}).get(
                    prompt_style, {'correct': 0, 'total': 0}
                )
                correct = stats['correct']
                total = stats['total']
                summary["accuracy"][task][order][prompt_style] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total if total > 0 else 0.0
                }

    # Save summary
    summary_file = os.path.join(output_dir, f"test_summary_{results['metadata']['model_name']}_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Test summary saved to {summary_file}")

    # Print accuracy table (tasks × orders × prompt_styles)
    print("\n=== Accuracy Summary ===")
    for task in tasks:
        print(f"\nTask: {task.upper()}")
        for order in orders:
            print(f"  {order}:")
            for prompt_style in prompt_styles:
                stats = summary["accuracy"][task][order][prompt_style]
                acc = stats['accuracy']
                print(f"    {prompt_style}: {stats['correct']}/{stats['total']} ({acc:.1%})")


def main():
    """Main function for running model tests."""
    parser = argparse.ArgumentParser(
        description="GraphDO Model Testing - Test LLMs on graph reasoning tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --description_dir ./descriptions
  python main.py --model Llama-2-7b-chat-hf --gpu_id 0 --description_dir ./descriptions
  python main.py --list_models
        """
    )
    
    # Model arguments
    parser.add_argument("--model", type=str, 
                       help="Model name to test")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU device ID (default: 0)")
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--description_files", nargs="+",
                           help="Specific JSON files containing graph descriptions")
    input_group.add_argument("--description_dir", type=str,
                           help="Directory containing description files")
    
    # Test configuration arguments
    parser.add_argument("--orders", nargs="+", 
                       default=["random", "bfs", "dfs", "pagerank", "ppr"],
                       help="Orders to test (default: all)")
    parser.add_argument("--prompt_styles", nargs="+", 
                       default=["zero_shot", "zero_shot_cot"],
                       help="Prompt styles to test (default: zero_shot zero_shot_cot)")
    parser.add_argument("--max_questions", type=int, default=None,
                       help="Maximum questions to test (default: all)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="model_test_results",
                       help="Output directory for test results (default: model_test_results)")
    
    # Utility arguments
    parser.add_argument("--list_models", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    # Handle list models
    if args.list_models:
        models = list_available_models()
        print("Available models:")
        print("  Local models:", models["local_models"])
        return
    
    # Validate required arguments
    if not args.model:
        parser.error("--model is required (unless using --list_models)")
    
    if not args.description_files and not args.description_dir:
        parser.error("Either --description_files or --description_dir is required")
    
    # Determine description files
    description_files = []
    
    if args.description_files:
        description_files = args.description_files
    elif args.description_dir:
        if not os.path.exists(args.description_dir):
            print(f"Error: Description directory not found: {args.description_dir}")
            return
        
        desc_dir_files = [os.path.join(args.description_dir, f) 
                         for f in os.listdir(args.description_dir) 
                         if f.endswith('_descriptions.json')]
        description_files = desc_dir_files
    
    # Validate files exist
    if not description_files:
        print("Error: No description files found!")
        if args.description_dir:
            print(f"No *_descriptions.json files found in {args.description_dir}")
            print("Please run generate.py first to create graph descriptions.")
        return
    
    for file_path in description_files:
        if not os.path.exists(file_path):
            print(f"Error: Description file not found: {file_path}")
            return
    
    print(f"Testing model {args.model} on {len(description_files)} description files:")
    for f in description_files:
        print(f"  {os.path.basename(f)}")
    print()
    
    try:
        run_model_test(
            model_name=args.model,
            description_files=description_files,
            orders=args.orders,
            prompt_styles=args.prompt_styles,
            output_dir=args.output_dir,
            max_questions=args.max_questions,
            gpu_id=args.gpu_id,
        )
        
        print("\\n=== Model testing completed successfully ===")
        
    except Exception as e:
        print(f"Error running model tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()