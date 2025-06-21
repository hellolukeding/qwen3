#!/usr/bin/env python3
"""
Qwen3-0.6B MCP 工具调用服务器
简洁的 MCP 协议实现，为 Qwen3 提供工具调用能力
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
import re
from datetime import datetime
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: callable

class ToolRegistry:
    """工具注册器"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def register(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]
    
    def _register_default_tools(self):
        """注册默认工具"""
        
        # 计算器工具
        def calculate(expression: str) -> Dict[str, Any]:
            """安全的数学计算"""
            try:
                # 只允许安全的数学表达式
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return {"error": "不支持的计算符号"}
                
                result = eval(expression)
                return {"result": result, "expression": expression}
            except Exception as e:
                return {"error": f"计算错误: {str(e)}"}
        
        self.register(Tool(
            name="calculate",
            description="数学计算器，支持基本运算",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如：2+3*4"
                    }
                },
                "required": ["expression"]
            },
            function=calculate
        ))
        
        # 时间工具
        def get_time(format_type: str = "datetime") -> Dict[str, Any]:
            """获取当前时间"""
            now = datetime.now()
            
            formats = {
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "timestamp": int(now.timestamp())
            }
            
            return {
                "current_time": formats.get(format_type, formats["datetime"]),
                "format": format_type
            }
        
        self.register(Tool(
            name="get_time",
            description="获取当前时间",
            parameters={
                "type": "object",
                "properties": {
                    "format_type": {
                        "type": "string",
                        "enum": ["datetime", "date", "time", "timestamp"],
                        "description": "时间格式类型"
                    }
                }
            },
            function=get_time
        ))
        
        # 文本处理工具
        def text_analysis(text: str, operation: str) -> Dict[str, Any]:
            """文本分析工具"""
            operations = {
                "length": len(text),
                "words": len(text.split()),
                "lines": len(text.split('\n')),
                "uppercase": text.upper(),
                "lowercase": text.lower(),
                "reverse": text[::-1]
            }
            
            if operation not in operations:
                return {"error": f"不支持的操作: {operation}"}
            
            return {
                "operation": operation,
                "result": operations[operation],
                "original_text": text[:100] + "..." if len(text) > 100 else text
            }
        
        self.register(Tool(
            name="text_analysis",
            description="文本分析和处理",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要分析的文本"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["length", "words", "lines", "uppercase", "lowercase", "reverse"],
                        "description": "分析操作类型"
                    }
                },
                "required": ["text", "operation"]
            },
            function=text_analysis
        ))

class ToolCallParser:
    """工具调用解析器"""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry
    
    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        """解析模型响应中的工具调用"""
        tool_calls = []
        
        # 匹配工具调用模式: TOOL:function_name(params)
        pattern = r'TOOL:(\w+)\((.*?)\)'
        matches = re.findall(pattern, response)
        
        for tool_name, params_str in matches:
            try:
                # 尝试解析参数
                if params_str.strip():
                    # 简单参数解析
                    if '=' in params_str:
                        # 键值对格式: key=value
                        params = {}
                        for param in params_str.split(','):
                            if '=' in param:
                                key, value = param.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"\'')
                                params[key] = value
                    else:
                        # 单个值
                        params = {"expression": params_str.strip().strip('"\'') if tool_name == "calculate" else params_str.strip()}
                else:
                    params = {}
                
                tool_calls.append({
                    "tool": tool_name,
                    "parameters": params
                })
            except Exception as e:
                logger.error(f"解析工具调用失败: {tool_name}, {params_str}, {e}")
        
        return tool_calls
    
    def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行工具调用"""
        results = []
        
        for call in tool_calls:
            tool_name = call["tool"]
            parameters = call["parameters"]
            
            tool = self.registry.get_tool(tool_name)
            if not tool:
                results.append({
                    "tool": tool_name,
                    "error": f"未知工具: {tool_name}"
                })
                continue
            
            try:
                result = tool.function(**parameters)
                results.append({
                    "tool": tool_name,
                    "parameters": parameters,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "parameters": parameters,
                    "error": f"执行错误: {str(e)}"
                })
        
        return results

class MCPServer:
    """MCP 服务器"""
    
    def __init__(self, qwen_url: str = "http://localhost:8000"):
        self.qwen_url = qwen_url
        self.tool_registry = ToolRegistry()
        self.parser = ToolCallParser(self.tool_registry)
        
    def create_tool_prompt(self, user_input: str, available_tools: List[str] = None) -> str:
        """创建包含工具信息的提示"""
        
        if available_tools is None:
            available_tools = list(self.tool_registry.tools.keys())
        
        tool_descriptions = []
        for tool_name in available_tools:
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                tool_descriptions.append(f"- {tool_name}: {tool.description}")
        
        prompt = f"""你是一个智能助手，可以使用以下工具来帮助回答问题：

可用工具：
{chr(10).join(tool_descriptions)}

使用工具的格式：TOOL:工具名(参数)
例如：
- 计算：TOOL:calculate(2+3*4)
- 获取时间：TOOL:get_time(format_type=datetime)
- 文本分析：TOOL:text_analysis(text=你好世界, operation=length)

用户问题：{user_input}

请分析用户问题，如果需要使用工具，请先使用TOOL:格式调用工具，然后基于工具结果回答用户问题。如果不需要工具，直接回答即可。

回答："""
        
        return prompt
    
    async def process_request(self, user_input: str, available_tools: List[str] = None) -> Dict[str, Any]:
        """处理用户请求"""
        
        # 1. 创建包含工具的提示
        prompt = self.create_tool_prompt(user_input, available_tools)
        
        # 2. 调用 Qwen3 模型
        try:
            response = requests.post(
                f"{self.qwen_url}/v1/completions",
                json={
                    "model": "./Qwen3-0.6B",
                    "prompt": prompt,
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "stop": ["用户问题：", "用户："]
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"模型调用失败: {response.status_code}"
                }
            
            model_response = response.json()["choices"][0]["text"].strip()
            
        except Exception as e:
            return {
                "success": False,
                "error": f"模型调用异常: {str(e)}"
            }
        
        # 3. 解析工具调用
        tool_calls = self.parser.parse_response(model_response)
        
        # 4. 执行工具
        tool_results = []
        if tool_calls:
            tool_results = self.parser.execute_tools(tool_calls)
        
        # 5. 如果有工具调用，生成最终回答
        final_response = model_response
        if tool_results:
            # 创建包含工具结果的二次提示
            tool_results_text = "\n".join([
                f"工具 {result['tool']}: {result.get('result', result.get('error'))}"
                for result in tool_results
            ])
            
            final_prompt = f"""基于以下工具执行结果，回答用户问题：

用户问题：{user_input}

工具执行结果：
{tool_results_text}

请基于工具结果给出完整、准确的回答："""
            
            try:
                final_response_req = requests.post(
                    f"{self.qwen_url}/v1/completions",
                    json={
                        "model": "./Qwen3-0.6B",
                        "prompt": final_prompt,
                        "max_tokens": 200,
                        "temperature": 0.6
                    },
                    timeout=30
                )
                
                if final_response_req.status_code == 200:
                    final_response = final_response_req.json()["choices"][0]["text"].strip()
            
            except Exception as e:
                logger.error(f"二次调用失败: {e}")
        
        return {
            "success": True,
            "user_input": user_input,
            "model_response": model_response,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "final_response": final_response,
            "has_tool_calls": len(tool_calls) > 0
        }
    
    def get_tools_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            "tools": self.tool_registry.list_tools(),
            "total_count": len(self.tool_registry.tools)
        }

if __name__ == "__main__":
    # 测试服务器
    server = MCPServer()
    
    # 测试工具调用
    test_queries = [
        "帮我计算一下 25 * 4 + 10 等于多少",
        "现在几点了？",
        "分析一下这个文本'Hello World'的字符长度",
        "你好，介绍一下自己"  # 不需要工具的问题
    ]
    
    async def test():
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"测试问题: {query}")
            print(f"{'='*60}")
            
            result = await server.process_request(query)
            
            if result["success"]:
                print(f"原始回答: {result['model_response']}")
                if result["has_tool_calls"]:
                    print(f"工具调用: {result['tool_calls']}")
                    print(f"工具结果: {result['tool_results']}")
                print(f"最终回答: {result['final_response']}")
            else:
                print(f"错误: {result['error']}")
    
    # 运行测试
    asyncio.run(test())
