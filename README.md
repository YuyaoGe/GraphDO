# GraphDO

Repository for ["Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?"](https://arxiv.org/pdf/2402.07140)

## Setup

```bash
uv sync                      # core deps (networkx, numpy, ...)
uv sync --extra local        # + torch/transformers for local models
```

## Usage

```bash
# 0. Generate graphs (T1-T5: ER random graphs)
uv run python graph_generator.py --num_graphs 280   # paper-scale (280 graphs/task)

# 0b. Generate graphs (T6: node classification, requires torch-geometric)
uv run python graph_generator.py --tasks node_classification --num_graphs 50 --datasets cora citeseer pubmed

# 1. Generate descriptions
uv run python generate.py                           # all tasks, all orders
uv run python generate.py --tasks connectivity      # specific task

# 2. Run model tests
uv run python main.py --model Llama-2-7b-chat-hf --gpu_id 0 --description_dir ./descriptions
uv run python main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --description_dir ./descriptions
```

## Graph Data

Use `graph_generator.py` to generate graph data for experiments:

```bash
# T1-T5: ER random graphs (n ∈ [5, 15], p = 0.3)
uv run python graph_generator.py --num_graphs 280

# T6: Node classification (Ego + Forest Fire sampling from Planetoid datasets)
# Cora/Citeseer/Pubmed will be automatically downloaded to ./data on first run
uv run python graph_generator.py --tasks node_classification --num_graphs 50 --datasets cora citeseer pubmed
```

## Structure

| Path | Description |
|------|-------------|
| `graph/*.jsonl` | Graph data in JSONL format |
| `graph_generator.py` | Generates graphs for all 6 tasks (ER for T1-T5, Planetoid sampling for T6) |
| `generate.py` | Description generation entry point |
| `main.py` | LLM testing entry point |
| `data_loader.py` | Reads JSONL graph data |
| `graph_ordering.py` | 5 edge ordering algorithms (random, BFS, DFS, PR, PPR) |
| `description_generator.py` | Generates NL descriptions with different orderings |
| `text_encoder.py` | Graph-to-text encoding (adjacency format) |
| `graphdo.py` | LLM call wrapper and prompt builder |
| `model_loader.py` | HuggingFace model loading |
| `evaluator.py` | Scores model answers (accuracy) |
| `examples.py` | Few-shot examples for all tasks |
| `info.py` | Model path configuration |

## Tasks & Orderings

**Tasks**: connectivity · cycle · shortest\_path · hamilton · topology · node\_classification

**Orderings**: `random` (baseline) · `bfs` · `dfs` · `pagerank` · `ppr`

**Prompt styles**: `zero_shot` · `zero_shot_cot` · `few_shot` · `cot` · `cot_bag`

## Model Config

Add local model paths in `info.py`.

## Acknowledgments

This paper references code from [GraphLLM](https://github.com/minnesotanlp/GraphLLM) and [NLGraph](https://github.com/Arthur-Heng/NLGraph).
