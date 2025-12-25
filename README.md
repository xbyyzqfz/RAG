# 南大软件学院SE流程智能体

基于RAG技术的软件工程流程智能助手，用于支持软件工程课程的学习和实践。

## 项目结构

```
├── config/              # 配置文件目录
│   └── config.yaml      # 主配置文件
├── src/                 # 源代码目录
│   ├── agent/           # 智能体核心代码
│   ├── rag/             # RAG相关实现
│   ├── tools/           # 工具插件
│   └── memory/          # 记忆系统
├── data/                # 数据目录
├── docs/                # 文档目录
├── scripts/             # 脚本文件
├── requirements.txt     # 依赖包列表
└── .env.example         # 环境变量模板
```

## 核心功能

1. **RAG技术集成**：混合检索、重排与过滤
2. **SE流程支持**：软件工程全流程指导
3. **工具插件**：搜索、代码解释、文档生成等
4. **长期记忆**：存储和检索历史交互
5. **MCP提升**：多跳推理提升检索效果

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件，填写相关配置
```

3. 运行智能体
```bash
python scripts/run_agent.py
```

## 技术选型

- **嵌入模型**：multilingual-MiniLM-L12-v2
- **向量数据库**：ChromaDB
- **检索策略**：混合检索（向量+关键词）
- **重排模型**：ms-marco-MiniLM-L-6-v2
- **智能体框架**：LangChain + CrewAI

## 开发说明

本项目基于Python开发，使用LangChain作为主要框架，实现了一个支持软件工程流程的智能助手。项目采用模块化设计，便于扩展和维护。
