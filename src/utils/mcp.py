from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI

class MCPProcessor:
    """Multi-hop Chain of Thought处理器，用于提升检索效果"""
    
    def __init__(self, llm: ChatOpenAI):
        """
        初始化MCP处理器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
        
    def generate_queries(self, initial_query: str, num_queries: int = 3) -> List[str]:
        """
        根据初始查询生成多跳查询
        
        Args:
            initial_query: 初始查询
            num_queries: 生成的查询数量
            
        Returns:
            多跳查询列表
        """
        prompt = f"""
你是一个帮助用户进行多跳检索的助手。请根据用户的初始查询，生成{num_queries}个相关的子查询，每个子查询应该关注初始查询的不同方面或相关概念。

初始查询：{initial_query}

要求：
1. 每个子查询应该是具体的、可独立检索的
2. 子查询之间应该有逻辑关联，但关注不同的方面
3. 生成的查询应该覆盖初始查询的主要内容
4. 只返回生成的查询，每行一个，不要添加任何解释或编号

生成的查询：
        """
        
        response = self.llm.predict(prompt)
        queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
        
        # 确保生成了足够数量的查询
        if len(queries) < num_queries:
            # 可以根据需要添加一些默认查询
            queries.extend([f"{initial_query} {i}" for i in range(num_queries - len(queries))])
        
        return queries[:num_queries]
    
    def synthesize_answers(self, query: str, answers: List[str]) -> str:
        """
        综合多个回答生成最终结果
        
        Args:
            query: 初始查询
            answers: 多个回答列表
            
        Returns:
            综合后的回答
        """
        prompt = f"""
你是一个信息综合助手。请根据以下查询和多个相关回答，综合生成一个全面、连贯的最终回答。

查询：{query}

相关回答：
{"\n".join([f"回答 {i+1}: {ans}" for i, ans in enumerate(answers)])}

要求：
1. 综合所有回答中的关键信息，不要遗漏重要内容
2. 按照逻辑顺序组织信息，确保回答连贯流畅
3. 避免重复内容，保持回答简洁明了
4. 只返回综合后的回答，不要添加任何解释

最终回答：
        """
        
        response = self.llm.predict(prompt)
        return response.strip()
    
    def process_mcp(self, initial_query: str, retrieve_func, num_queries: int = 3) -> str:
        """
        完整的MCP处理流程
        
        Args:
            initial_query: 初始查询
            retrieve_func: 检索函数，接受查询并返回相关文档
            num_queries: 生成的查询数量
            
        Returns:
            最终回答
        """
        # 1. 生成多跳查询
        queries = self.generate_queries(initial_query, num_queries)
        
        # 2. 对每个查询进行检索
        all_docs = []
        for q in queries:
            docs = retrieve_func(q)
            all_docs.extend(docs)
        
        # 3. 去重
        unique_docs = []
        seen_contents = set()
        for doc in all_docs:
            content = doc.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)
        
        # 4. 构建上下文
        context = "\n".join([doc.page_content for doc in unique_docs])
        
        # 5. 生成最终回答
        answer_prompt = f"""
请根据以下上下文回答用户的问题。

上下文：{context}

问题：{initial_query}

要求：
1. 基于提供的上下文回答问题
2. 回答要准确、全面
3. 使用简洁明了的语言
4. 只返回回答内容，不要添加任何解释

回答：
        """
        
        final_answer = self.llm.predict(answer_prompt)
        return final_answer.strip()