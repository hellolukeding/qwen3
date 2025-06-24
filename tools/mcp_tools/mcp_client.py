#!/usr/bin/env python3
"""
MCP 客户端 - 用于与 MCP 服务器交互
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from mcp_server import MCPServer

class MCPClient:
    """MCP 客户端"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server = MCPServer(server_url)
    
    async def chat(self, message: str, tools: List[str] = None) -> Dict[str, Any]:
        """发送聊天消息"""
        return await self.server.process_request(message, tools)
    
    def get_available_tools(self) -> Dict[str, Any]:
        """获取可用工具列表"""
        return self.server.get_tools_info()
    
    async def interactive_chat(self):
        """交互式聊天模式"""
        print("🤖 Qwen3-0.6B MCP 工具调用助手")
        print("输入 'help' 查看可用工具，输入 'quit' 退出")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if user_input.lower() == 'help':
                    tools_info = self.get_available_tools()
                    print("\n🛠️ 可用工具:")
                    for tool in tools_info["tools"]:
                        print(f"  • {tool['name']}: {tool['description']}")
                    continue
                
                if not user_input:
                    continue
                
                print("🤔 思考中...")
                result = await self.chat(user_input)
                
                if result["success"]:
                    print(f"\n🤖 助手: {result['final_response']}")
                    
                    if result["has_tool_calls"]:
                        print(f"\n🔧 使用了工具: {[call['tool'] for call in result['tool_calls']]}")
                else:
                    print(f"\n❌ 错误: {result['error']}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")

async def main():
    """主函数"""
    client = MCPClient()
    
    # 检查工具
    print("🔍 检查可用工具...")
    tools_info = client.get_available_tools()
    print(f"✅ 已加载 {tools_info['total_count']} 个工具")
    
    # 启动交互模式
    await client.interactive_chat()

if __name__ == "__main__":
    asyncio.run(main())
