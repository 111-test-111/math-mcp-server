#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件工具模块：提供文件名处理、路径构建和文件存在性检查等功能
"""

import os
import uuid
import datetime
from typing import Optional, Tuple


def get_output_path() -> str:
    """获取输出文件保存的路径，优先使用环境变量，否则使用系统临时目录"""
    # 优先使用OUTPUT_PATH环境变量
    output_path = os.environ.get("OUTPUT_PATH")
    if output_path and os.path.isdir(output_path):
        return output_path

    # 然后尝试使用TMPFILE_PATH环境变量（已有的信号处理用）
    tmp_path = os.environ.get("TMPFILE_PATH")
    if tmp_path and os.path.isdir(tmp_path):
        return tmp_path

    # 最后使用系统临时目录
    import tempfile

    return tempfile.gettempdir()


def ensure_file_directory(filepath: str) -> None:
    """确保文件所在目录存在，如果不存在则创建"""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def generate_unique_filename(
    prefix: str, extension: str, custom_name: Optional[str] = None
) -> Tuple[str, str]:
    """
    生成唯一的文件名，确保不会覆盖已存在的文件

    Args:
        prefix: 文件名前缀
        extension: 文件扩展名（不带点）
        custom_name: 用户提供的自定义文件名（可选）

    Returns:
        Tuple[str, str]: (完整文件路径, 仅文件名)
    """
    output_path = get_output_path()

    # 如果提供了自定义名称，使用它作为基础
    if custom_name:
        # 确保文件名有正确的扩展名
        if not custom_name.lower().endswith(f".{extension.lower()}"):
            filename = f"{custom_name}.{extension}"
        else:
            filename = custom_name

        # 构建完整路径
        filepath = os.path.join(output_path, filename)

        # 检查文件是否已存在，如果存在则添加时间戳
        if os.path.exists(filepath):
            name_part, ext_part = os.path.splitext(filename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name_part}_{timestamp}{ext_part}"
            filepath = os.path.join(output_path, filename)

            # 如果还存在（极少数情况），添加随机UUID
            if os.path.exists(filepath):
                unique_id = str(uuid.uuid4())[:8]
                filename = f"{name_part}_{timestamp}_{unique_id}{ext_part}"
                filepath = os.path.join(output_path, filename)
    else:
        # 使用默认命名规则
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{random_id}.{extension}"
        filepath = os.path.join(output_path, filename)

        # 以防万一还有冲突
        while os.path.exists(filepath):
            random_id = str(uuid.uuid4())[:8]
            filename = f"{prefix}_{timestamp}_{random_id}.{extension}"
            filepath = os.path.join(output_path, filename)

    # 确保目录存在
    ensure_file_directory(filepath)

    return filepath, filename


def resolve_signal_file_path(signal_file: str) -> str:
    """
    解析信号文件路径，支持相对路径和绝对路径

    Args:
        signal_file: 信号文件名或路径

    Returns:
        str: 完整的信号文件路径
    """
    # 如果已经是绝对路径，直接返回
    if os.path.isabs(signal_file) or signal_file.startswith(get_output_path()):
        return signal_file

    # 如果不是.json结尾，添加扩展名
    if not signal_file.lower().endswith(".json"):
        signal_file = f"{signal_file}.json"

    # 拼接到输出目录
    return os.path.join(get_output_path(), signal_file)
