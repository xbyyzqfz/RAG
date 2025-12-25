# 南大软件学院SE流程智能体 - 技术文档

## 1. 项目概述

本项目开发了一个基于RAG（Retrieval-Augmented Generation）技术的软件工程流程智能体，旨在为软件学院师生提供专业的软件工程流程指导和技术支持。

## 2. RAG技术选型与实现

### 2.1 嵌入模型

**选型：** SentenceTransformer（paraphrase-multilingual-MiniLM-L12-v2）

**理由：**
- 支持多语言，适合处理中文文档
- 模型体积小，推理速度快
- 在多种NLP任务上表现优异
- 支持批量嵌入生成，提高效率

**实现细节：**
```python
class MultilingualEmbedding:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query], show_progress_bar=False)[0].tolist()
```

### 2.2 索引模式

**选型：** 混合索引（向量索引 + 关键词索引）

**理由：**
- 向量索引擅长语义匹配，能够理解查询的深层含义
- 关键词索引擅长精确匹配，能够快速定位包含特定术语的文档
- 混合索引结合了两者的优势，提高检索准确性和效率

**实现细节：**
```python
class HybridIndex:
    def __init__(self, config: Dict[str, Any]):
        # 初始化向量存储
        self.vector_store = Chroma(
            persist_directory=config.get("vector_store_path", "./chroma_db"),
            embedding_function=config.get("embedding_function")
        )
        
        # 初始化关键词索引（内存实现）
        self.keyword_index = {}
```

### 2.3 检索策略

**选型：** 混合检索（向量检索 + 关键词检索 + 重排）

**理由：**
- 向量检索获取语义相关的文档
- 关键词检索获取包含特定术语的文档
- 通过权重融合两种检索结果
- 使用重排模型进一步优化结果排序

**实现细节：**
```python
class HybridRetriever:
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        # 向量检索
        vector_results = self._vector_retrieve(query, top_k)
        
        # 关键词检索
        keyword_results = self._keyword_retrieve(query, top_k)
        
        # 融合结果
        merged_results = self._merge_results(vector_results, keyword_results)
        
        # 重排
        if self.reranker:
            merged_results = self.reranker.rerank(query, merged_results)
        
        return merged_results[:top_k]
```

### 2.4 重排策略

**选型：** CrossEncoder（cross-encoder/ms-marco-MiniLM-L-6-v2）

**理由：**
- 专为文本对匹配任务设计，重排效果好
- 模型体积小，推理速度快
- 在MS MARCO等基准数据集上表现优异

**实现细节：**
```python
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        
        # 根据分数排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]
```

### 2.5 多跳检索（MCP）

**选型：** Multi-hop Chain of Thought

**理由：**
- 能够处理复杂的多步骤查询
- 通过生成子查询扩展检索范围
- 综合多个查询的结果提高回答质量

**实现细节：**
```python
class MCPProcessor:
    def process_mcp(self, initial_query: str, retrieve_func, num_queries: int = 3) -> str:
        # 1. 生成多跳查询
        queries = self.generate_queries(initial_query, num_queries)
        
        # 2. 对每个查询进行检索
        all_docs = []
        for q in queries:
            docs = retrieve_func(q)
            all_docs.extend(docs)
        
        # 3. 去重和综合
        unique_docs = self._deduplicate_docs(all_docs)
        return self._synthesize_results(initial_query, unique_docs)
```

## 3. 智能体核心功能

### 3.1 SE流程管理

实现了完整的软件工程流程指导，包括：
- 需求分析
- 系统设计
- 编码实现
- 测试
- 部署与维护

每个阶段包含详细的任务、交付物和推荐工具。

### 3.2 工具插件系统

支持多种工具集成：
- 代码审查工具
- 文档生成工具
- 流程可视化工具
- 测试运行工具

### 3.3 记忆系统

- 短期记忆：存储对话上下文，支持上下文理解
- 长期记忆：存储历史交互，支持知识积累和个性化回复

## 4. 系统架构

```
├── src/
│   ├── rag/              # RAG核心模块
│   │   ├── embedding.py  # 嵌入模型
│   │   ├── index.py      # 索引管理
│   │   ├── retrieval.py  # 检索策略
│   │   ├── reranker.py   # 重排模型
│   │   └── rag_system.py # RAG系统整合
│   ├── agent/            # 智能体模块
│   │   ├── agent.py      # 智能体核心
│   │   ├── se_flow.py    # SE流程管理
│   │   └── prompt_templates.py # 提示模板
│   └── utils/            # 工具模块
│       ├── tools.py      # 工具插件
│       ├── memory.py     # 记忆系统
│       └── mcp.py        # 多跳检索
├── config/               # 配置文件
├── requirements.txt      # 依赖列表
└── example_app.py        # 示例应用
```

## 5. 技术栈

| 类别 | 技术/框架 | 版本 | 用途 |
|------|-----------|------|------|
| 核心框架 | LangChain | 0.0.344 | 大语言模型应用开发 |
| 嵌入模型 | SentenceTransformer | 2.2.2 | 文本嵌入生成 |
| 向量数据库 | Chroma | 0.4.14 | 向量存储和检索 |
| 大语言模型 | OpenAI GPT | - | 文本生成 |
| 工具集成 | - | - | 代码审查、文档生成等 |
| 记忆系统 | 自定义实现 | - | 短期和长期记忆管理 |

## 6. 部署与运行

### 6.1 环境要求

- Python 3.8+
- pip
- 适当的硬件资源（推荐8GB RAM以上）

### 6.2 安装步骤

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
```bash
cp .env.example .env
# 编辑.env文件，配置所需的API密钥
```

3. 运行示例应用：
```bash
python example_app.py
```

## 7. 未来改进方向

1. **更丰富的工具集成**：整合更多软件工程相关工具，如代码生成、自动化测试等
2. **多模态支持**：支持文档、图表等多模态内容的处理
3. **个性化推荐**：根据用户历史交互提供个性化的流程指导
4. **性能优化**：优化检索和生成速度，提高系统响应效率
5. **云平台部署**：在华为云、阿里云等平台部署，提供更稳定的服务

## 8. 总结

本项目成功实现了一个基于RAG技术的软件工程流程智能体，通过合理的技术选型和创新的功能设计，为用户提供了专业、高效的软件工程流程指导。系统具有良好的扩展性和可维护性，为未来的功能扩展和性能优化奠定了基础。