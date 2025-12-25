from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from src.rag.embedding import create_embedding
from src.rag.index import IndexManager, process_documents
from src.rag.retrieval import RetrievalManager
from src.rag.reranker import create_reranker
from src.utils.mcp import MCPProcessor

class RAGSystem:
    """RAG系统主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化RAG系统
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化嵌入模型
        embedding_config = config.get("embedding", {})
        self.embedding = create_embedding(
            model_name=embedding_config.get("model_name"),
            device=embedding_config.get("device", "cpu")
        )
        
        # 初始化索引
        index_config = config.get("index", {})
        index_manager = IndexManager(index_config)
        self.index = index_manager.create_index(self.embedding)
        
        # 初始化检索器
        retrieval_config = config.get("retrieval", {})
        retrieval_manager = RetrievalManager(retrieval_config)
        self.retriever = retrieval_manager.create_retriever(self.index)
        
        # 初始化重排器
        rerank_config = retrieval_config.get("rerank", {})
        self.reranker = None
        if rerank_config.get("enabled", False):
            self.reranker = create_reranker(
                model_name=rerank_config.get("model"),
                device=embedding_config.get("device", "cpu")
            )
        
        # 过滤配置
        self.filter_config = retrieval_config.get("filter", {})
        
        # 初始化MCP处理器
        self.mcp_enabled = config.get("mcp", {}).get("enabled", False)
        if self.mcp_enabled:
            self.llm = ChatOpenAI(
                model_name=config.get("mcp", {}).get("model", "gpt-3.5-turbo"),
                temperature=0.3
            )
            self.mcp_processor = MCPProcessor(self.llm)
            self.mcp_num_queries = config.get("mcp", {}).get("num_queries", 3)
    
    def add_documents(self, texts: List[str]) -> List[str]:
        """
        添加文档到RAG系统
        
        Args:
            texts: 文本列表
            
        Returns:
            文档ID列表
        """
        # 处理文档
        documents = process_documents(
            texts,
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 100)
        )
        
        # 添加到索引
        return self.index.add_documents(documents)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        # 使用配置的top_k或传入的top_k
        actual_top_k = top_k or self.config.get("retrieval", {}).get("top_k", 10)
        
        # 如果启用了MCP，使用多跳检索
        if self.mcp_enabled:
            return self._retrieve_with_mcp(query, actual_top_k)
        else:
            # 普通检索流程
            results = self.retriever.retrieve(query)
        
        # 重排
        if self.reranker:
            rerank_top_k = self.config.get("retrieval", {}).get("rerank", {}).get("top_k", 5)
            results = self.reranker.rerank(query, results, top_k=rerank_top_k)
        
        # 过滤
        if self.filter_config.get("enabled", False):
            threshold = self.filter_config.get("threshold", 0.5)
            results = self.retriever.filter_results(results, threshold=threshold)
        
        return results[:actual_top_k]
    
    def _retrieve_with_mcp(self, query: str, top_k: int) -> List[Document]:
        """
        使用MCP（Multi-hop Chain of Thought）进行检索
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        # 生成多跳查询
        queries = self.mcp_processor.generate_queries(query, self.mcp_num_queries)
        
        # 对每个查询进行检索
        all_docs = []
        for q in queries:
            docs = self.retriever.retrieve(q)
            all_docs.extend(docs)
        
        # 去重
        unique_docs = []
        seen_contents = set()
        for doc in all_docs:
            content = doc.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)
        
        # 重排合并后的文档
        if self.reranker:
            rerank_top_k = self.config.get("retrieval", {}).get("rerank", {}).get("top_k", 5)
            results = self.reranker.rerank(query, unique_docs, top_k=rerank_top_k)
            return results[:top_k]
        
        return unique_docs[:top_k]
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        获取索引信息
        
        Returns:
            索引信息字典
        """
        return {
            "vector_store": str(self.index.vector_store),
            "keyword_index_size": len(self.index.keyword_index),
            "embedding_dimension": self.embedding.get_embedding_dimension()
        }

# 工厂函数
def create_rag_system(config: Dict[str, Any]) -> RAGSystem:
    """
    创建RAG系统实例
    
    Args:
        config: 配置字典
        
    Returns:
        RAG系统实例
    """
    return RAGSystem(config)
