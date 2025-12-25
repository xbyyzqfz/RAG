from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List, Tuple

class CrossEncoderReranker:
    """基于CrossEncoder的重排器"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 device: str = "cpu"):
        """
        初始化重排器
        
        Args:
            model_name: 重排模型名称
            device: 运行设备 (cpu/gpu)
        """
        self.model = CrossEncoder(model_name, device=device)
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        对检索结果进行重排
        
        Args:
            query: 查询文本
            documents: 检索到的文档列表
            top_k: 返回的结果数量
            
        Returns:
            重排后的文档列表
        """
        if not documents:
            return []
        
        # 准备模型输入
        pairs = [[query, doc.page_content] for doc in documents]
        
        # 获取重排分数
        scores = self.model.predict(pairs)
        
        # 按分数排序
        reranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        # 返回top_k结果
        return [doc for doc, score in reranked_results[:top_k]]
    
    def rerank_with_scores(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        对检索结果进行重排并返回分数
        
        Args:
            query: 查询文本
            documents: 检索到的文档列表
            
        Returns:
            重排后的文档和分数列表
        """
        if not documents:
            return []
        
        # 准备模型输入
        pairs = [[query, doc.page_content] for doc in documents]
        
        # 获取重排分数
        scores = self.model.predict(pairs)
        
        # 按分数排序
        return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# 工厂函数
def create_reranker(model_name: str, device: str = "cpu"):
    """
    创建重排器实例
    
    Args:
        model_name: 模型名称
        device: 运行设备
        
    Returns:
        重排器实例
    """
    if model_name.startswith("cross-encoder/"):
        return CrossEncoderReranker(model_name, device)
    else:
        raise ValueError(f"不支持的重排模型: {model_name}")
