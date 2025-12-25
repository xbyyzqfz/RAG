from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from src.rag.rag_system import RAGSystem
from src.agent.se_flow import SEFlow
from src.agent.prompt_templates import agent_system_prompt, rag_answer_prompt, se_flow_prompt, code_review_prompt
from src.utils.tools import ToolManager
from src.utils.memory import LongTermMemory, ShortTermMemory

class SEAgent:
    """南大软件学院SE流程智能体"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化智能体
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化RAG系统
        self.rag_system = RAGSystem(config.get("rag", {}))
        
        # 初始化SE流程管理
        self.se_flow = SEFlow()
        
        # 初始化大语言模型
        self.llm = ChatOpenAI(
            model_name=config.get("agent", {}).get("model"),
            temperature=config.get("agent", {}).get("temperature", 0.3)
        )
        
        # 初始化工具管理器
        self.tool_manager = ToolManager()
        
        # 初始化记忆系统
        self.long_term_memory = LongTermMemory(config.get("memory", {}).get("file_path", "memory.json"))
        self.short_term_memory = ShortTermMemory(config.get("memory", {}).get("capacity", 10))
        
        # 初始化对话历史
        self.chat_history = [
            ("system", SYSTEM_PROMPT)
        ]
    
    def add_knowledge(self, texts: List[str]) -> List[str]:
        """
        向智能体添加知识
        
        Args:
            texts: 文本列表
            
        Returns:
            文档ID列表
        """
        return self.rag_system.add_documents(texts)
    
    def process_query(self, query: str) -> str:
        """
        处理用户查询
        
        Args:
            query: 用户查询文本
            
        Returns:
            智能体的回答
        """
        # 保存用户查询到对话历史和短期记忆
        self.chat_history.append(("user", query))
        self.short_term_memory.add("user", query)
        
        # 从长期记忆中检索相关信息
        long_term_memories = self.long_term_memory.search_memories(query, limit=3)
        memory_context = "\n".join([mem["content"] for mem in long_term_memories])
        
        # 检索相关知识
        retrieved_docs = self.rag_system.retrieve(query)
        
        # 构建上下文（结合RAG检索结果和长期记忆）
        rag_context = "\n".join([doc.page_content for doc in retrieved_docs])
        context = f"RAG检索结果：\n{rag_context}\n\n长期记忆：\n{memory_context}"
        
        # 生成回答
        answer = self._generate_answer(query, context)
        
        # 保存智能体回答到对话历史、短期记忆和长期记忆
        self.chat_history.append(("assistant", answer))
        self.short_term_memory.add("assistant", answer)
        self.long_term_memory.add_memory(
            f"用户：{query}\n智能体：{answer}",
            memory_type="interaction",
            metadata={"query": query, "answer": answer}
        )
        
        return answer
    
    def _generate_answer(self, query: str, context: str) -> str:
        """
        生成回答
        
        Args:
            query: 用户查询
            context: 检索到的上下文
            
        Returns:
            生成的回答
        """
        # 构建提示
        prompt = rag_prompt.format(context=context, question=query)
        
        # 生成回答
        response = self.llm.predict(prompt)
        
        return response.strip()
    
    def get_se_flow_guide(self, user_need: str) -> str:
        """
        获取软件工程流程指导
        
        Args:
            user_need: 用户需求
            
        Returns:
            流程指导
        """
        return self.se_flow.generate_flow_guide(user_need)
    
    def get_chat_history(self) -> List[tuple]:
        """
        获取对话历史
        
        Returns:
            对话历史列表
        """
        return self.chat_history
    
    def clear_chat_history(self) -> None:
        """
        清除对话历史
        """
        self.chat_history = [
            ("system", SYSTEM_PROMPT)
        ]
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        获取智能体信息
        
        Returns:
            智能体信息字典
        """
        return {
            "name": self.config.get("agent", {}).get("name"),
            "description": self.config.get("agent", {}).get("description"),
            "model": self.config.get("agent", {}).get("model"),
            "rag_info": self.rag_system.get_index_info(),
            "available_tools": self.tool_manager.get_available_tools(),
            "memory_info": {
                "long_term_memory_count": len(self.long_term_memory.get_memories()),
                "short_term_memory_capacity": self.short_term_memory.capacity
            }
        }
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        return self.tool_manager.call_tool(tool_name, **kwargs)
    
    def get_available_tools(self) -> List[str]:
        """
        获取可用工具列表
        
        Returns:
            工具名称列表
        """
        return self.tool_manager.get_available_tools()
    
    def clear_memory(self, memory_type: Optional[str] = None) -> None:
        """
        清除记忆
        
        Args:
            memory_type: 记忆类型（short, long, 或 None 表示清除所有）
        """
        if memory_type == "short" or memory_type is None:
            self.short_term_memory.clear()
        if memory_type == "long" or memory_type is None:
            self.long_term_memory.clear_memories()
