from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List, Optional
import numpy as np

class MultilingualEmbedding(Embeddings):
    """多语言文本嵌入模型"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                 device: str = "cpu"):
        """
        初始化多语言嵌入模型
        
        Args:
            model_name: 模型名称
            device: 运行设备 (cpu/gpu)
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成嵌入
        
        Args:
            texts: 文档文本列表
            
        Returns:
            嵌入向量列表
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        为查询文本生成嵌入
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入维度
        
        Returns:
            嵌入维度
        """
        return self.dimension

# 工厂函数
def create_embedding(model_name: str, device: str = "cpu") -> Embeddings:
    """
    创建嵌入模型实例
    
    Args:
        model_name: 模型名称
        device: 运行设备
        
    Returns:
        嵌入模型实例
    """
    if model_name.startswith("sentence-transformers/"):
        return MultilingualEmbedding(model_name, device)
    else:
        # 可以扩展支持其他嵌入模型
        raise ValueError(f"不支持的嵌入模型: {model_name}")
