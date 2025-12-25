from typing import List, Dict, Any, Optional
import subprocess
import requests
import json

class ToolManager:
    """工具管理器，用于管理和调用各种工具插件"""
    
    def __init__(self):
        self.tools = {
            "code_reviewer": self.code_reviewer,
            "document_generator": self.document_generator,
            "flow_visualizer": self.flow_visualizer,
            "test_runner": self.test_runner
        }
    
    def get_available_tools(self) -> List[str]:
        """
        获取可用工具列表
        
        Returns:
            工具名称列表
        """
        return list(self.tools.keys())
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        调用指定工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        return self.tools[tool_name](**kwargs)
    
    def code_reviewer(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        代码审查工具
        
        Args:
            code: 代码文本
            language: 编程语言
            
        Returns:
            审查结果
        """
        # 这里可以集成更复杂的代码审查工具，如flake8、pylint等
        lines = code.splitlines()
        return {
            "tool": "code_reviewer",
            "language": language,
            "line_count": len(lines),
            "comment_count": len([line for line in lines if line.strip().startswith("#")]),
            "suggestions": [
                "考虑添加更多注释",
                "检查异常处理",
                "考虑代码重构以提高可读性"
            ]
        }
    
    def document_generator(self, content: str, doc_type: str = "markdown") -> str:
        """
        文档生成工具
        
        Args:
            content: 内容
            doc_type: 文档类型
            
        Returns:
            生成的文档
        """
        if doc_type == "markdown":
            return f"# 生成的文档\n\n{content}\n\n---\n\n*自动生成的文档*"
        elif doc_type == "html":
            return f"<html><body><h1>生成的文档</h1><p>{content}</p></body></html>"
        else:
            return content
    
    def flow_visualizer(self, flow_data: Dict[str, Any]) -> str:
        """
        流程可视化工具
        
        Args:
            flow_data: 流程数据
            
        Returns:
            可视化结果（如Mermaid代码）
        """
        stages = flow_data.get("stages", [])
        mermaid_code = "graph TD\n"
        
        for i, stage in enumerate(stages):
            stage_id = f"stage_{i}"
            mermaid_code += f"{stage_id}[\"{stage['name']}\"]\n"
            
            if i > 0:
                prev_stage_id = f"stage_{i-1}"
                mermaid_code += f"{prev_stage_id} --> {stage_id}\n"
        
        return mermaid_code
    
    def test_runner(self, test_code: str, language: str = "python") -> Dict[str, Any]:
        """
        测试运行工具
        
        Args:
            test_code: 测试代码
            language: 编程语言
            
        Returns:
            测试结果
        """
        if language == "python":
            try:
                # 执行测试代码
                result = subprocess.run(
                    ["python", "-c", test_code],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                return {
                    "tool": "test_runner",
                    "language": language,
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            except Exception as e:
                return {
                    "tool": "test_runner",
                    "language": language,
                    "success": False,
                    "error": str(e)
                }
        else:
            return {
                "tool": "test_runner",
                "language": language,
                "success": False,
                "error": f"Language {language} not supported"
            }
    
    def web_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        网络搜索工具
        
        Args:
            query: 搜索查询
            num_results: 结果数量
            
        Returns:
            搜索结果列表
        """
        # 这里可以集成真实的搜索引擎API
        return [
            {
                "title": f"搜索结果 {i+1}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"这是关于 '{query}' 的搜索结果 {i+1} 的摘要。"
            }
            for i in range(num_results)
        ]
