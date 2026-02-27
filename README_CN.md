# GraphDO

["Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?"](https://arxiv.org/pdf/2402.07140) 的代码仓库。

## 环境配置

```bash
uv sync                      # 核心依赖（networkx, numpy, ...）
uv sync --extra local        # + torch/transformers（本地模型）
```

## 使用方法

```bash
# 0. 生成图数据
uv run python graph_generator.py --num_graphs 280   # 论文规模（每任务 280 个图）

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
uv run python graph_generator.py --num_graphs 280
```

## 目录说明

| 路径 | 说明 |
|------|------|
| `graph/*.jsonl` | JSONL 格式图数据 |
| `graph_generator.py` | 基于 ER 模型生成各任务随机图数据 |
| `generate.py` | 描述生成入口 |
| `main.py` | LLM 测试入口 |
| `data_loader.py` | 读取 JSONL 图数据 |
| `graph_ordering.py` | 5 种边排序算法（random, BFS, DFS, PR, PPR） |
| `description_generator.py` | 按不同排序生成自然语言描述 |
| `text_encoder.py` | 图→文本编码（邻接表格式） |
| `graphdo.py` | LLM 调用封装与 Prompt 构建 |
| `model_loader.py` | HuggingFace 模型加载 |
| `evaluator.py` | 模型答案评估（正确率） |
| `examples.py` | 五个任务的 few-shot 示例 |
| `info.py` | 模型路径配置 |

## 任务与排序

**任务**：connectivity · cycle · shortest\_path · hamilton · topology

**排序方法**：`random`（基线）· `bfs` · `dfs` · `pagerank` · `ppr`

**Prompt 风格**：`zero_shot` · `zero_shot_cot` · `few_shot` · `cot` · `cot_bag`

## 模型配置

在 `info.py` 中添加本地模型路径。
