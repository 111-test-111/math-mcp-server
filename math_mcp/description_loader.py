# -*- coding: utf-8 -*-
"""
工具描述加载器
用于动态加载工具描述配置文件并生成标准文档字符串
"""

import os
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional


class DescriptionLoader:
    """工具描述加载器类"""

    def __init__(self, config_module: Optional[str] = None):
        """
        初始化描述加载器

        Args:
            config_module: 配置模块名，如果为None则使用默认模块
        """
        if config_module is None:
            # 默认使用同目录下的 tool_descriptions.py
            self.config_module = "tool_descriptions"
        else:
            self.config_module = config_module

        self.descriptions = {}
        self.load_descriptions()

    def load_descriptions(self):
        """加载描述配置模块"""
        try:
            # 尝试直接导入模块
            try:
                import math_mcp.tool_descriptions as td

                self.descriptions = td.TOOL_DESCRIPTIONS
            except ImportError:
                # 如果直接导入失败，尝试从当前目录加载
                current_dir = Path(__file__).parent
                module_path = current_dir / f"{self.config_module}.py"

                if not module_path.exists():
                    raise FileNotFoundError(f"配置模块文件不存在: {module_path}")

                spec = importlib.util.spec_from_file_location(
                    self.config_module, module_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.descriptions = module.TOOL_DESCRIPTIONS

            print(f"已加载 {len(self.descriptions)} 个工具的描述信息")

        except Exception as e:
            print(f"加载描述配置模块失败: {e}")
            self.descriptions = {}

    def get_tool_description(self, tool_name: str) -> Dict[str, Any]:
        """
        获取指定工具的描述信息

        Args:
            tool_name: 工具名称

        Returns:
            包含工具描述信息的字典
        """
        return self.descriptions.get(tool_name, {})

    def generate_docstring(self, tool_name: str) -> str:
        """
        为指定工具生成标准格式的文档字符串

        Args:
            tool_name: 工具名称

        Returns:
            格式化的文档字符串
        """
        desc = self.get_tool_description(tool_name)

        if not desc:
            return f"工具 {tool_name} 的描述信息未找到"

        # 生成文档字符串
        docstring_parts = []

        # 添加主要描述
        if "description" in desc:
            docstring_parts.append(desc["description"])
            docstring_parts.append("")  # 空行

        # 添加参数说明
        if "args" in desc and desc["args"]:
            docstring_parts.append("Args:")
            for arg_name, arg_desc in desc["args"].items():
                docstring_parts.append(f"    {arg_name}: {arg_desc}")
            docstring_parts.append("")  # 空行

        # 添加返回值说明
        if "returns" in desc:
            docstring_parts.append("Returns:")
            docstring_parts.append(f"    {desc['returns']}")
            docstring_parts.append("")  # 空行

        # 添加使用示例
        if "examples" in desc and desc["examples"]:
            docstring_parts.append("Examples:")
            for example in desc["examples"]:
                docstring_parts.append(f"    {example}")

        return "\n".join(docstring_parts)

    def apply_docstring(self, func, tool_name: str):
        """
        将生成的文档字符串应用到函数上

        Args:
            func: 目标函数
            tool_name: 工具名称
        """
        docstring = self.generate_docstring(tool_name)
        func.__doc__ = docstring
        return func

    def get_all_tool_names(self) -> list:
        """
        获取所有工具名称列表

        Returns:
            工具名称列表
        """
        return list(self.descriptions.keys())

    def reload_descriptions(self):
        """重新加载描述配置模块"""
        self.load_descriptions()

    def export_descriptions_to_file(self, output_file: str, format: str = "json"):
        """
        将描述信息导出到文件

        Args:
            output_file: 输出文件路径
            format: 输出格式 ('json', 'markdown', 'text')
        """
        if format == "json":
            import json

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"tools": self.descriptions}, f, ensure_ascii=False, indent=2)

        elif format == "markdown":
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("# Math MCP Server 工具文档\n\n")

                for tool_name, desc in self.descriptions.items():
                    f.write(f"## {tool_name}\n\n")
                    f.write(f"**描述**: {desc.get('description', '无描述')}\n\n")

                    if "args" in desc and desc["args"]:
                        f.write("**参数**:\n\n")
                        for arg_name, arg_desc in desc["args"].items():
                            f.write(f"- `{arg_name}`: {arg_desc}\n")
                        f.write("\n")

                    if "returns" in desc:
                        f.write(f"**返回值**: {desc['returns']}\n\n")

                    if "examples" in desc and desc["examples"]:
                        f.write("**示例**:\n\n")
                        for example in desc["examples"]:
                            f.write(f"```python\n{example}\n```\n\n")

                    f.write("---\n\n")

        elif format == "text":
            with open(output_file, "w", encoding="utf-8") as f:
                for tool_name in self.descriptions:
                    f.write(f"{tool_name}:\n")
                    f.write(self.generate_docstring(tool_name))
                    f.write("\n" + "=" * 50 + "\n\n")

        print(f"描述信息已导出到: {output_file}")


def create_description_decorator(loader: DescriptionLoader):
    """
    创建描述装饰器工厂函数

    Args:
        loader: 描述加载器实例

    Returns:
        装饰器函数
    """

    def description_decorator(tool_name: str):
        """装饰器，用于自动应用工具描述"""

        def decorator(func):
            return loader.apply_docstring(func, tool_name)

        return decorator

    return description_decorator


# 创建全局描述加载器实例
description_loader = DescriptionLoader()

# 创建装饰器
apply_description = create_description_decorator(description_loader)
