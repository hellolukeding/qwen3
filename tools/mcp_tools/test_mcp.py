#!/usr/bin/env python3
"""
快速测试脚本 - 测试 MCP 工具调用功能
"""

import asyncio
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp_server import MCPServer

async def quick_test():
    """快速测试工具调用功能"""
    print("🚀 启动 Qwen3-0.6B MCP 工具调用测试")
    print("=" * 60)
    
    # 创建服务器实例
    server = MCPServer()
    
    # 测试用例
    test_cases = [
        {
            "query": "帮我计算 15 * 8 + 25",
            "description": "数学计算测试"
        },
        {
            "query": "现在是什么时间？",
            "description": "时间查询测试"
        },
        {
            "query": "分析文本'人工智能很有趣'有多少个字符",
            "description": "文本分析测试"
        },
        {
            "query": "你好，请介绍一下你自己",
            "description": "普通对话测试"
        }
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试 {i}/{total_count}: {test_case['description']}")
        print(f"问题: {test_case['query']}")
        print("-" * 40)
        
        try:
            result = await server.process_request(test_case['query'])
            
            if result["success"]:
                success_count += 1
                print(f"✅ 成功")
                print(f"回答: {result['final_response']}")
                
                if result["has_tool_calls"]:
                    tools_used = [call['tool'] for call in result['tool_calls']]
                    print(f"🔧 使用工具: {', '.join(tools_used)}")
                else:
                    print("💬 直接回答（未使用工具）")
            else:
                print(f"❌ 失败: {result['error']}")
                
        except Exception as e:
            print(f"❌ 异常: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试完成: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("🎉 所有测试通过！MCP 工具调用系统运行正常")
    else:
        print("⚠️  部分测试失败，请检查 vLLM 服务是否正常运行")
        print("   运行命令: ./start_vllm.sh")

def main():
    """主函数"""
    try:
        asyncio.run(quick_test())
    except KeyboardInterrupt:
        print("\n\n⏹️  测试中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("\n💡 请确保:")
        print("  1. vLLM 服务正在运行 (./start_vllm.sh)")
        print("  2. 端口 8000 可访问")
        print("  3. 网络连接正常")

if __name__ == "__main__":
    main()
