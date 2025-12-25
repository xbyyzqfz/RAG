import yaml
import os
from src.agent.agent import SEAgent
from src.utils.tools import ToolManager

# 加载配置
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 创建智能体实例
agent = SEAgent(config)

print("南大软件学院SE流程智能体")
print("=" * 50)
print("可用命令：")
print("1. 输入问题直接与智能体对话")
print("2. 输入 'flow' 获取软件工程流程指导")
print("3. 输入 'tools' 查看可用工具")
print("4. 输入 'info' 查看智能体信息")
print("5. 输入 'quit' 退出程序")
print("=" * 50)

while True:
    user_input = input("\n用户：").strip()
    
    if user_input.lower() == "quit":
        print("再见！")
        break
    
    elif user_input.lower() == "flow":
        need = input("请描述您的软件需求：")
        guide = agent.get_se_flow_guide(need)
        print(f"\n智能体：\n{guide}")
    
    elif user_input.lower() == "tools":
        tools = agent.get_available_tools()
        print("\n可用工具：")
        for tool in tools:
            print(f"- {tool}")
    
    elif user_input.lower() == "info":
        info = agent.get_agent_info()
        print("\n智能体信息：")
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    
    else:
        response = agent.process_query(user_input)
        print(f"\n智能体：\n{response}")