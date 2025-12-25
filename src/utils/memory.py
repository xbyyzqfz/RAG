from typing import List, Dict, Any, Optional
import time
import json
import os

class LongTermMemory:
    """长期记忆系统，用于存储和检索智能体的记忆"""
    
    def __init__(self, memory_file: str = "memory.json"):
        """
        初始化长期记忆系统
        
        Args:
            memory_file: 记忆存储文件路径
        """
        self.memory_file = memory_file
        self.memories = self._load_memories()
    
    def _load_memories(self) -> List[Dict[str, Any]]:
        """
        从文件加载记忆
        
        Returns:
            记忆列表
        """
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save_memories(self) -> None:
        """
        保存记忆到文件
        """
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)
    
    def add_memory(self, content: str, memory_type: str = "interaction", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        添加记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            metadata: 记忆元数据
            
        Returns:
            添加的记忆
        """
        memory = {
            "id": f"mem_{int(time.time() * 1000)}",
            "content": content,
            "type": memory_type,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.memories.append(memory)
        self._save_memories()
        
        return memory
    
    def get_memories(self, memory_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取记忆
        
        Args:
            memory_type: 记忆类型，None表示获取所有类型
            limit: 返回的记忆数量限制
            
        Returns:
            记忆列表
        """
        filtered = self.memories
        
        if memory_type:
            filtered = [mem for mem in filtered if mem["type"] == memory_type]
        
        # 按时间戳降序排序，返回最新的记忆
        sorted_memories = sorted(filtered, key=lambda x: x["timestamp"], reverse=True)
        
        return sorted_memories[:limit]
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        搜索记忆
        
        Args:
            query: 搜索关键词
            limit: 返回的记忆数量限制
            
        Returns:
            匹配的记忆列表
        """
        # 简单的文本匹配，实际可以使用更复杂的检索算法
        matches = [
            mem for mem in self.memories 
            if query.lower() in mem["content"].lower()
        ]
        
        # 按时间戳降序排序
        sorted_matches = sorted(matches, key=lambda x: x["timestamp"], reverse=True)
        
        return sorted_matches[:limit]
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否删除成功
        """
        initial_length = len(self.memories)
        self.memories = [mem for mem in self.memories if mem["id"] != memory_id]
        
        if len(self.memories) < initial_length:
            self._save_memories()
            return True
        
        return False
    
    def clear_memories(self, memory_type: Optional[str] = None) -> None:
        """
        清除记忆
        
        Args:
            memory_type: 记忆类型，None表示清除所有类型
        """
        if memory_type:
            self.memories = [mem for mem in self.memories if mem["type"] != memory_type]
        else:
            self.memories = []
        
        self._save_memories()

class ShortTermMemory:
    """短期记忆系统，用于临时存储对话上下文"""
    
    def __init__(self, capacity: int = 10):
        """
        初始化短期记忆系统
        
        Args:
            capacity: 短期记忆容量
        """
        self.capacity = capacity
        self.memories: List[Dict[str, Any]] = []
    
    def add(self, role: str, content: str) -> None:
        """
        添加短期记忆
        
        Args:
            role: 角色（user, assistant, system）
            content: 内容
        """
        memory = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        self.memories.append(memory)
        
        # 保持记忆容量
        if len(self.memories) > self.capacity:
            self.memories.pop(0)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        获取所有短期记忆
        
        Returns:
            短期记忆列表
        """
        return self.memories.copy()
    
    def clear(self) -> None:
        """
        清除所有短期记忆
        """
        self.memories.clear()
    
    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        获取上下文文本
        
        Args:
            max_tokens: 最大token数量限制
            
        Returns:
            上下文文本
        """
        context = ""
        for memory in self.memories:
            if max_tokens and len(context.split()) > max_tokens:
                break
            context += f"{memory['role']}: {memory['content']}\n"
        
        return context.strip()
