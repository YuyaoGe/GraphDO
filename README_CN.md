# GraphDO

["Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?"](https://arxiv.org/pdf/2402.07140) 的代码仓库。

## 环境配置

```bash
uv sync                      # 核心依赖（networkx, numpy, ...）
uv sync --extra local        # + torch/transformers（本地模型）
```

## 使用方法

```bash
# 0. 生成图数据（T1-T5：ER 随机图）
uv run python graph_generator.py --num_graphs 280   # 论文规模（每任务 280 个图）

# 0b. 生成图数据（T6：节点分类，需要 torch-geometric）
uv run python graph_generator.py --tasks node_classification --num_graphs 50 --datasets cora citeseer pubmed

# 1. 生成描述文件
uv run python generate.py                           # 全部任务、全部排序
uv run python generate.py --tasks connectivity      # 指定任务

# 2. 运行模型测试
uv run python main.py --model Llama-2-7b-chat-hf --gpu_id 0 --description_dir ./descriptions
uv run python main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --description_dir ./descriptions
```

## 图数据说明

使用 `graph_generator.py` 生成实验所需的图数据：

```bash
# T1-T5：ER 随机图（n ∈ [5, 15], p = 0.3）
uv run python graph_generator.py --num_graphs 280

# T6：节点分类（从 Planetoid 数据集进行 Ego + Forest Fire 采样）
# Cora/Citeseer/Pubmed 数据集会在首次运行时自动下载到 ./data 目录
uv run python graph_generator.py --tasks node_classification --num_graphs 50 --datasets cora citeseer pubmed
```

## 目录说明

| 路径 | 说明 |
|------|------|
| `graph/*.jsonl` | JSONL 格式图数据 |
| `graph_generator.py` | 生成全部 6 个任务的图数据（T1-T5 使用 ER 模型，T6 使用 Planetoid 采样） |
| `generate.py` | 描述生成入口 |
| `main.py` | LLM 测试入口 |
| `data_loader.py` | 读取 JSONL 图数据 |
| `graph_ordering.py` | 5 种边排序算法（random, BFS, DFS, PR, PPR） |
| `description_generator.py` | 按不同排序生成自然语言描述 |
| `text_encoder.py` | 图→文本编码（邻接表格式） |
| `graphdo.py` | LLM 调用封装与 Prompt 构建 |
| `model_loader.py` | HuggingFace 模型加载 |
| `evaluator.py` | 模型答案评估（正确率） |
| `examples.py` | 六个任务的 few-shot 示例 |
| `info.py` | 模型路径配置 |

## 任务与排序

**任务**：connectivity · cycle · shortest\_path · hamilton · topology · node\_classification

**排序方法**：`random`（基线）· `bfs` · `dfs` · `pagerank` · `ppr`

**Prompt 风格**：`zero_shot` · `zero_shot_cot` · `few_shot` · `cot` · `cot_bag`

## 模型配置

在 `info.py` 中添加本地模型路径。

## 致谢

本论文参考了 [GraphLLM](https://github.com/minnesotanlp/GraphLLM) 和 [NLGraph](https://github.com/Arthur-Heng/NLGraph) 的代码。
