#!/usr/bin/env python3
"""
Unified description generation script for GraphDO.
Reads graph data from ./graph/{task}/ (100 pre-sampled graphs per task).

Usage:
  python generate.py                              # Generate all tasks, all orders
  python generate.py --tasks connectivity cycle   # Specific tasks only
  python generate.py --max_graphs 20 --save_stats # Limit + save stats
  python generate.py --stats                      # Show dataset statistics, then exit
  python generate.py --verify                     # Verify paths, then exit
  python generate.py --test connectivity          # Quick test for one task, then exit
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

from description_generator import GraphDescriptionGenerator
from data_loader import GraphDataLoader


ALL_TASKS = ["connectivity", "cycle", "shortest_path", "hamilton", "topology",
             "node_classification"]
ALL_ORDERS = ["random", "bfs", "dfs", "pagerank", "ppr"]


def show_stats(graph_root: str):
    """Show dataset statistics."""
    print("GraphDO Dataset Statistics")
    print("=" * 50)
    try:
        loader = GraphDataLoader(graph_root)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    summary = loader.get_task_summary()
    total = 0
    for task, info in summary.items():
        count = info["graph_count"]
        total += count
        print(f"  {task}: {count} graphs  [{info['graph_type']}, weighted={info['weighted']}]")
    print(f"\nTotal: {total} graphs -> {total * len(ALL_ORDERS)} descriptions ({len(ALL_ORDERS)} orders)")
    est_mb = (total * len(ALL_ORDERS) * 1.5) / 1024
    print(f"Estimated disk: ~{est_mb:.1f} MB")


def verify_paths(graph_root: str, output_dir: str):
    """Verify project paths and file status."""
    print("GraphDO Path Verification")
    print("=" * 50)
    print(f"Working directory: {os.getcwd()}")

    core_files = [
        "info.py", "model_loader.py", "graphdo.py", "main.py",
        "evaluator.py", "generate.py", "data_loader.py",
        "graph_ordering.py", "text_encoder.py", "description_generator.py",
        "requirements.txt",
    ]
    print("\nCore files:")
    for f in core_files:
        status = "OK" if os.path.exists(f) else "MISSING"
        print(f"  [{status}]  {f}")

    print(f"\nGraph data ({graph_root}):")
    if os.path.exists(graph_root):
        for task in ALL_TASKS:
            task_dir = os.path.join(graph_root, task)
            if os.path.exists(task_dir):
                n = len([f for f in os.listdir(task_dir) if f.endswith(".txt")])
                print(f"  [OK]  {task}: {n} graphs")
            else:
                print(f"  [MISSING]  {task}")
    else:
        print(f"  [MISSING] {graph_root}")

    print(f"\nOutput directory ({output_dir}):")
    if os.path.exists(output_dir):
        jsons = [f for f in os.listdir(output_dir) if f.endswith("_descriptions.json")]
        print(f"  [OK] {output_dir}  ({len(jsons)} description files)")
        for f in sorted(jsons):
            print(f"    - {f}")
    else:
        print(f"  [NOT EXISTS] {output_dir}  (will be created on generation)")


def run_test(task: str, graph_root: str, output_dir: str, orders: list):
    """Quick test: generate 2 graphs for one task."""
    print(f"Quick test: task={task}, 2 graphs, orders={orders}")
    generator = GraphDescriptionGenerator(graph_root)
    try:
        descriptions = generator.generate_task_descriptions(task=task, max_graphs=2, orders=orders)
        print(f"Generated {len(descriptions)} graph descriptions")
        if descriptions:
            sample = descriptions[0]
            print(f"\nSample from {sample['filename']}:")
            for order in orders:
                if order in sample["descriptions"]:
                    d = sample["descriptions"][order]
                    if "description" in d:
                        print(f"  {order}: {d['description'][:100]}...")
                    else:
                        print(f"  {order}: ERROR - {d.get('error', '?')}")
        out_file = f"test_{task}_descriptions.json"
        generator.save_descriptions_to_json(descriptions, out_file)
        print(f"\nSaved to {out_file}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_generation(args):
    """Core generation logic."""
    print("=" * 60)
    print("GraphDO Description Generator")
    print("=" * 60)

    generator = GraphDescriptionGenerator(args.graph_root)
    os.makedirs(args.output_dir, exist_ok=True)

    summary = generator.data_loader.get_task_summary()
    tasks_to_run = [t for t in args.tasks if t in summary]

    print(f"Tasks:   {tasks_to_run}")
    print(f"Orders:  {args.orders}")
    print(f"Output:  {args.output_dir}")
    if args.max_graphs:
        print(f"Max graphs per task: {args.max_graphs}")
    print()

    stats = {
        "generation_start_time": datetime.now().isoformat(),
        "orders": args.orders,
        "task_stats": {},
        "order_stats": defaultdict(int),
        "total_graphs": 0,
        "total_descriptions": 0,
        "generated_files": [],
        "errors": [],
    }

    for task in tasks_to_run:
        print(f"Task: {task}")
        try:
            start = time.time()
            descriptions = generator.generate_task_descriptions(
                task=task, max_graphs=args.max_graphs, orders=args.orders
            )
            elapsed = time.time() - start

            output_file = os.path.join(args.output_dir, f"{task}_descriptions.json")
            generator.save_descriptions_to_json(descriptions, output_file)

            valid = sum(
                1 for d in descriptions
                for o in args.orders
                if o in d["descriptions"] and "description" in d["descriptions"][o]
            )
            for o in args.orders:
                stats["order_stats"][o] += sum(
                    1 for d in descriptions
                    if o in d["descriptions"] and "description" in d["descriptions"][o]
                )

            stats["task_stats"][task] = {"graphs": len(descriptions), "descriptions": valid}
            stats["total_graphs"] += len(descriptions)
            stats["total_descriptions"] += valid
            stats["generated_files"].append({
                "file": output_file, "task": task,
                "graphs": len(descriptions), "descriptions": valid,
                "generation_time": round(elapsed, 2),
            })
            print(f"  -> {len(descriptions)} graphs, {valid} descriptions ({elapsed:.1f}s)\n")

        except Exception as e:
            msg = f"{task}: {e}"
            print(f"  ERROR: {e}\n")
            stats["errors"].append(msg)

    stats["generation_end_time"] = datetime.now().isoformat()
    print("=" * 60)
    print(f"Done: {stats['total_graphs']} graphs, {stats['total_descriptions']} descriptions")
    order_summary = ", ".join(f"{o}={stats['order_stats'][o]}" for o in args.orders)
    print(f"By order: {order_summary}")
    print(f"Files saved to: {args.output_dir}/")
    if stats["errors"]:
        print(f"Errors:")
        for err in stats["errors"]:
            print(f"  - {err}")

    if args.save_stats:
        stats_file = os.path.join(args.output_dir, "generation_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        print(f"Stats saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="GraphDO description generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py                              # Generate all tasks
  python generate.py --tasks connectivity cycle   # Specific tasks
  python generate.py --max_graphs 20 --save_stats # Limited + save stats
  python generate.py --orders random bfs          # Specific orders
  python generate.py --stats                      # Dataset statistics only
  python generate.py --verify                     # Path verification only
  python generate.py --test connectivity          # Quick test for one task
        """,
    )

    parser.add_argument("--graph_root", default="./graph",
                        help="Graph data directory (default: ./graph)")
    parser.add_argument("--output_dir", default="./descriptions",
                        help="Output directory (default: ./descriptions)")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                        help="Tasks to generate (default: all 5)")
    parser.add_argument("--orders", nargs="+", default=ALL_ORDERS,
                        help="Ordering methods (default: random bfs dfs pagerank ppr)")
    parser.add_argument("--max_graphs", type=int, default=None,
                        help="Max graphs per task (default: all 100)")
    parser.add_argument("--save_stats", action="store_true",
                        help="Save generation statistics to JSON")

    # Special modes (exit after running)
    parser.add_argument("--stats", action="store_true",
                        help="Show dataset statistics and exit")
    parser.add_argument("--verify", action="store_true",
                        help="Verify paths and file status, then exit")
    parser.add_argument("--test", metavar="TASK",
                        help="Quick test: generate 2 graphs for TASK and exit")

    args = parser.parse_args()

    try:
        if args.stats:
            show_stats(args.graph_root)
        elif args.verify:
            verify_paths(args.graph_root, args.output_dir)
        elif args.test:
            run_test(args.test, args.graph_root, args.output_dir, args.orders)
        else:
            run_generation(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
