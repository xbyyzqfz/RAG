from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import chromadb
import re
import jieba

class HybridIndex:
    """混合索引（向量+关键词）"""
    
    def __init__(self, embedding, vector_store_path: str = "./chromadb", 
                 collection_name: str = "se_knowledge_base"):
        """
        初始化混合索引
        
        Args:
            embedding: 嵌入模型
            vector_store_path: 向量存储路径
            collection_name: 集合名称
        """
        self.embedding = embedding
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        
        # 初始化向量存储
        self.vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embedding,
            collection_name=collection_name
        )
        
        # 初始化关键词索引（内存中）
        self.keyword_index = {}
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档到索引
        
        Args:
            documents: 文档列表
            
        Returns:
            文档ID列表
        """
        # 添加到向量索引
        doc_ids = self.vector_store.add_documents(documents)
        
        # 添加到关键词索引
        for doc_id, doc in zip(doc_ids, documents):
            keywords = self._extract_keywords(doc.page_content)
            for keyword in keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(doc_id)
        
        return doc_ids
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取关键词
        
        Args:
            text: 文本内容
            
        Returns:
            关键词列表
        """
        # 移除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        
        # 使用jieba分词
        words = jieba.cut(text)
        
        # 过滤停用词（简单实现）
        stopwords = set(["的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"])
        keywords = [word for word in words if word not in stopwords and len(word) > 1]
        
        return list(set(keywords))
    
    def get_vector_store(self) -> Chroma:
        """
        获取向量存储
        
        Returns:
            向量存储实例
        """
        return self.vector_store
    
    def get_keyword_index(self) -> Dict[str, List[str]]:
        """
        获取关键词索引
        
        Returns:
            关键词索引字典
        """
        return self.keyword_index

class IndexManager:
    """索引管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化索引管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
    
    def create_index(self, embedding) -> HybridIndex:
        """
        创建索引实例
        
        Args:
            embedding: 嵌入模型
            
        Returns:
            索引实例
        """
        index_type = self.config.get("type", "hybrid")
        
        if index_type == "hybrid":
            return HybridIndex(
                embedding=embedding,
                vector_store_path=self.config.get("vector_store_path", "./chromadb"),
                collection_name=self.config.get("collection_name", "se_knowledge_base")
            )
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")

# 文档加载和处理
def process_documents(texts: List[str], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    处理文本为文档列表
    
    Args:
        texts: 文本列表
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
        
    Returns:
        文档列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "", " "]
    )
    
    documents = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={"source": f"chunk_{i}"}
            )
            documents.append(doc)
    
    return documents
