from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import numpy as np

class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, index, top_k: int = 10, hybrid_alpha: float = 0.7):
        """
        初始化混合检索器
        
        Args:
            index: 索引实例
            top_k: 返回结果数量
            hybrid_alpha: 混合权重 (0=纯关键词, 1=纯向量)
        """
        self.index = index
        self.top_k = top_k
        self.hybrid_alpha = hybrid_alpha
    
    def retrieve(self, query: str) -> List[Document]:
        """
        混合检索
        
        Args:
            query: 查询文本
            
        Returns:
            检索到的文档列表
        """
        # 向量检索
        vector_results = self.index.vector_store.similarity_search_with_score(query, k=self.top_k * 2)
        vector_doc_map = {doc.metadata.get("source", doc.page_content): (doc, score) for doc, score in vector_results}
        
        # 关键词检索
        keywords = self._extract_query_keywords(query)
        keyword_results = self._keyword_search(keywords)
        
        # 混合结果
        hybrid_results = self._hybrid_rerank(vector_doc_map, keyword_results, query)
        
        # 返回top_k结果
        return [doc for doc, score in hybrid_results[:self.top_k]]
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """
        从查询中提取关键词
        
        Args:
            query: 查询文本
            
        Returns:
            关键词列表
        """
        # 使用索引的关键词提取方法
        from src.rag.index import HybridIndex
        if hasattr(self.index, '_extract_keywords'):
            return self.index._extract_keywords(query)
        return []
    
    def _keyword_search(self, keywords: List[str]) -> Dict[str, float]:
        """
        关键词检索
        
        Args:
            keywords: 关键词列表
            
        Returns:
            文档ID到相关性分数的映射
        """
        doc_scores = {}
        
        for keyword in keywords:
            if keyword in self.index.keyword_index:
                doc_ids = self.index.keyword_index[keyword]
                for doc_id in doc_ids:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += 1
        
        # 归一化分数
        if doc_scores:
            max_score = max(doc_scores.values())
            for doc_id in doc_scores:
                doc_scores[doc_id] = doc_scores[doc_id] / max_score
        
        return doc_scores
    
    def _hybrid_rerank(self, vector_results: Dict[str, tuple], 
                       keyword_results: Dict[str, float], 
                       query: str) -> List[tuple]:
        """
        混合重排结果
        
        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            query: 查询文本
            
        Returns:
            重排后的结果列表
        """
        hybrid_scores = {}
        
        # 处理向量结果
        for doc_source, (doc, vector_score) in vector_results.items():
            # 转换向量分数（相似度越高，分数越高）
            vector_score_normalized = 1.0 - vector_score
            
            # 获取关键词分数
            keyword_score = 0.0
            for doc_id, score in keyword_results.items():
                # 简化：假设doc_id包含在文档内容中
                if doc_id in doc.page_content:
                    keyword_score = score
                    break
            
            # 混合分数
            hybrid_score = (self.hybrid_alpha * vector_score_normalized) + \
                          ((1 - self.hybrid_alpha) * keyword_score)
            
            hybrid_scores[doc] = hybrid_score
        
        # 按分数排序
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def filter_results(self, results: List[Document], threshold: float = 0.5) -> List[Document]:
        """
        过滤结果
        
        Args:
            results: 检索结果列表
            threshold: 过滤阈值
            
        Returns:
            过滤后的结果列表
        """
        # 这里可以实现更复杂的过滤逻辑
        # 简化：返回所有结果
        return results

class RetrievalManager:
    """检索管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检索管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
    
    def create_retriever(self, index) -> HybridRetriever:
        """
        创建检索器实例
        
        Args:
            index: 索引实例
            
        Returns:
            检索器实例
        """
        return HybridRetriever(
            index=index,
            top_k=self.config.get("top_k", 10),
            hybrid_alpha=self.config.get("hybrid_alpha", 0.7)
        )
