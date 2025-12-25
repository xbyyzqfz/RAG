from typing import List, Dict, Any

class SEFlow:
    """软件工程流程管理"""
    
    def __init__(self):
        """
        初始化软件工程流程
        """
        self.phases = [
            {
                "name": "需求分析",
                "key": "requirements_analysis",
                "description": "确定软件系统的功能、性能和约束",
                "tasks": [
                    "收集用户需求",
                    "分析需求的可行性",
                    "确定需求优先级",
                    "编写需求规格说明书"
                ],
                "deliverables": [
                    "需求规格说明书 (SRS)",
                    "用户故事地图",
                    "用例图"
                ],
                "tools": [
                    "Jira",
                    "Confluence",
                    "Draw.io"
                ]
            },
            {
                "name": "系统设计",
                "key": "system_design",
                "description": "设计软件系统的架构和组件",
                "tasks": [
                    "架构设计",
                    "详细设计",
                    "数据库设计",
                    "API设计"
                ],
                "deliverables": [
                    "架构设计文档",
                    "类图",
                    "序列图",
                    "数据库 schema"
                ],
                "tools": [
                    "UMLet",
                    "ERwin",
                    "Swagger"
                ]
            },
            {
                "name": "编码实现",
                "key": "implementation",
                "description": "根据设计文档编写代码",
                "tasks": [
                    "搭建开发环境",
                    "编写代码",
                    "代码审查",
                    "单元测试"
                ],
                "deliverables": [
                    "源代码",
                    "单元测试报告",
                    "代码审查记录"
                ],
                "tools": [
                    "Git",
                    "VS Code",
                    "Jenkins",
                    "SonarQube"
                ]
            },
            {
                "name": "测试",
                "key": "testing",
                "description": "验证软件系统的功能和质量",
                "tasks": [
                    "集成测试",
                    "系统测试",
                    "验收测试",
                    "性能测试"
                ],
                "deliverables": [
                    "测试计划",
                    "测试用例",
                    "测试报告",
                    "缺陷报告"
                ],
                "tools": [
                    "Selenium",
                    "JMeter",
                    "TestLink"
                ]
            },
            {
                "name": "部署与维护",
                "key": "deployment_maintenance",
                "description": "部署软件系统并提供长期维护",
                "tasks": [
                    "系统部署",
                    "监控设置",
                    "问题修复",
                    "功能迭代"
                ],
                "deliverables": [
                    "部署文档",
                    "监控报告",
                    "维护记录",
                    "版本更新日志"
                ],
                "tools": [
                    "Docker",
                    "Kubernetes",
                    "Prometheus",
                    "ELK Stack"
                ]
            }
        ]
    
    def get_all_phases(self) -> List[Dict[str, Any]]:
        """
        获取所有软件工程阶段
        
        Returns:
            阶段列表
        """
        return self.phases
    
    def get_phase_by_key(self, phase_key: str) -> Dict[str, Any]:
        """
        根据阶段key获取阶段信息
        
        Args:
            phase_key: 阶段key
            
        Returns:
            阶段信息
        """
        for phase in self.phases:
            if phase["key"] == phase_key:
                return phase
        return None
    
    def generate_flow_guide(self, user_need: str) -> str:
        """
        根据用户需求生成流程指导
        
        Args:
            user_need: 用户需求
            
        Returns:
            流程指导文本
        """
        guide = f"# 软件工程流程指导\n\n针对需求：{user_need}\n\n"
        
        for i, phase in enumerate(self.phases, 1):
            guide += f"## {i}. {phase['name']}\n"
            guide += f"**描述**：{phase['description']}\n\n"
            
            guide += f"### 主要任务\n"
            for task in phase['tasks']:
                guide += f"- {task}\n"
            
            guide += f"\n### 输出文档\n"
            for deliverable in phase['deliverables']:
                guide += f"- {deliverable}\n"
            
            guide += f"\n### 推荐工具\n"
            for tool in phase['tools']:
                guide += f"- {tool}\n"
            
            guide += "\n"
        
        return guide
