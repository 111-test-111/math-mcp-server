# -*- coding: utf-8 -*-
"""
信号处理计算模块
提供数字信号处理功能
"""

import math
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

try:
    from .file_utils import generate_unique_filename, resolve_signal_file_path
except ImportError:
    from math_mcp.file_utils import generate_unique_filename, resolve_signal_file_path


class SignalProcessingCalculator:
    """信号处理计算器类"""

    def __init__(self):
        pass

    def signal_processing_tool(
        self,
        operation: str,
        signal: Optional[List[float]] = None,
        signal_file: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        frequency: Optional[float] = None,
        filter_type: Optional[str] = None,
        cutoff_freq: Optional[float] = None,
        window_size: Optional[int] = None,
        overlap: Optional[float] = None,
        order: Optional[int] = None,
        duration: Optional[float] = None,
        noise_level: Optional[float] = None,
        signal_type: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        信号处理工具

        Args:
            operation: 操作类型
            signal: 输入信号数据（可选，与signal_file二选一）
            signal_file: 信号文件路径（可选，与signal二选一）
            sampling_rate: 采样率
            frequency: 频率
            filter_type: 滤波器类型
            cutoff_freq: 截止频率
            window_size: 窗口大小
            overlap: 重叠度
            order: 滤波器阶数
            duration: 信号持续时间
            noise_level: 噪声水平
            signal_type: 信号类型（用于信号生成）
            output_filename: 输出文件名（不含路径，可选）
        """
        # 处理信号输入（文件或直接数据）
        actual_signal = signal
        actual_sampling_rate = sampling_rate

        # 检查是否提供了有效的信号文件路径
        if signal_file is not None and signal_file.strip():
            # 解析文件路径（处理相对路径）
            full_signal_file_path = resolve_signal_file_path(signal_file)

            # 从文件加载信号
            load_result = self._load_signal_from_file(full_signal_file_path)
            if "error" in load_result:
                return load_result

            actual_signal = load_result["signal"]
            # 如果文件中有采样率且用户未指定，使用文件中的采样率
            if sampling_rate is None and load_result["sampling_rate"] is not None:
                actual_sampling_rate = load_result["sampling_rate"]

        # 参数验证（使用原始参数，不是处理后的）
        validation_result = self._validate_parameters(
            operation,
            signal,  # 使用原始signal参数，不是actual_signal
            signal_file,
            sampling_rate,  # 使用原始sampling_rate参数
            frequency,
            filter_type,
            cutoff_freq,
            window_size,
            order,
            duration,
            signal_type,
        )
        if validation_result is not None:
            return validation_result

        try:
            if operation == "fft":
                return self._fft_analysis(
                    actual_signal, actual_sampling_rate or 1.0, output_filename
                )
            elif operation == "generate_signal":
                return self._generate_signal(
                    frequency,
                    actual_sampling_rate,
                    duration or 1.0,
                    signal_type or "sine",
                    noise_level,
                    output_filename,
                )
            elif operation == "filter":
                return self._advanced_filter(
                    actual_signal,
                    filter_type,
                    cutoff_freq,
                    actual_sampling_rate,
                    order,
                    output_filename,
                )
            elif operation == "windowing":
                return self._apply_window(
                    actual_signal, window_size, "hanning", output_filename
                )
            elif operation == "autocorrelation":
                return self._autocorrelation(actual_signal, output_filename)
            elif operation == "crosscorrelation":
                return self._crosscorrelation(
                    actual_signal[: len(actual_signal) // 2],
                    actual_signal[len(actual_signal) // 2 :],
                    output_filename,
                )
            elif operation == "spectral_analysis":
                return self._spectral_analysis(
                    actual_signal, actual_sampling_rate, window_size, output_filename
                )
            elif operation == "signal_metrics":
                return self._signal_metrics(actual_signal, actual_sampling_rate or 1.0)
            elif operation == "convolution":
                return self._convolution(
                    actual_signal[: len(actual_signal) // 2],
                    actual_signal[len(actual_signal) // 2 :],
                    output_filename,
                )
            elif operation == "deconvolution":
                return self._deconvolution(
                    actual_signal[: len(actual_signal) // 2],
                    actual_signal[len(actual_signal) // 2 :],
                    output_filename,
                )
            elif operation == "envelope_detection":
                return self._envelope_detection(actual_signal, output_filename)
            elif operation == "phase_analysis":
                return self._phase_analysis(
                    actual_signal, actual_sampling_rate, output_filename
                )
            elif operation == "noise_reduction":
                return self._noise_reduction(
                    actual_signal,
                    filter_type or "moving_average",
                    window_size or 5,
                    output_filename,
                )
            elif operation == "resampling":
                new_rate = cutoff_freq or (
                    actual_sampling_rate * 2
                )  # 使用cutoff_freq作为新采样率
                return self._resampling(
                    actual_signal, actual_sampling_rate, new_rate, output_filename
                )
            elif operation == "modulation":
                return self._amplitude_modulation(
                    actual_signal,
                    frequency or 10.0,
                    actual_sampling_rate,
                    output_filename,
                )
            elif operation == "demodulation":
                return self._amplitude_demodulation(
                    actual_signal,
                    frequency or 10.0,
                    actual_sampling_rate,
                    output_filename,
                )
            else:
                return {"error": f"不支持的操作: {operation}"}

        except Exception as e:
            return {"error": f"信号处理错误: {str(e)}"}

    def _validate_parameters(
        self,
        operation: str,
        signal: Optional[List[float]],
        signal_file: Optional[str],
        sampling_rate: Optional[float],
        frequency: Optional[float],
        filter_type: Optional[str],
        cutoff_freq: Optional[float],
        window_size: Optional[int],
        order: Optional[int],
        duration: Optional[float],
        signal_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """参数验证"""
        valid_operations = [
            "fft",
            "generate_signal",
            "filter",
            "windowing",
            "autocorrelation",
            "crosscorrelation",
            "spectral_analysis",
            "signal_metrics",
            "convolution",
            "deconvolution",
            "envelope_detection",
            "phase_analysis",
            "noise_reduction",
            "resampling",
            "modulation",
            "demodulation",
        ]

        if operation not in valid_operations:
            return {
                "error": f"无效的操作类型: {operation}，支持的操作: {valid_operations}"
            }

        # 检查需要信号的操作
        signal_required_ops = [
            "fft",
            "filter",
            "windowing",
            "autocorrelation",
            "crosscorrelation",
            "spectral_analysis",
            "signal_metrics",
            "convolution",
            "deconvolution",
            "envelope_detection",
            "phase_analysis",
            "noise_reduction",
            "resampling",
            "modulation",
            "demodulation",
        ]

        if operation in signal_required_ops:
            # 检查是否同时提供了两者（但要排除默认的None值）
            signal_provided = signal is not None and (
                len(signal) > 0 if isinstance(signal, list) else True
            )
            signal_file_provided = signal_file is not None and (
                signal_file.strip() != "" if isinstance(signal_file, str) else True
            )

            # 检查是否提供了信号数据（直接数据或文件路径）
            if not signal_provided and not signal_file_provided:
                return {
                    "error": f"操作 {operation} 需要提供 signal 或 signal_file 参数"
                }

            if signal_provided and signal_file_provided:
                return {
                    "error": "不能同时提供 signal 和 signal_file 参数，请选择其中一个"
                }

            # 如果提供的是直接信号数据，进行验证
            if signal is not None:
                if len(signal) == 0:
                    return {"error": "signal 不能为空"}
                if not all(isinstance(x, (int, float)) for x in signal):
                    return {"error": "signal 中的所有元素必须是数值类型"}

            # 如果提供的是文件路径，验证文件格式
            if signal_file is not None:
                if not isinstance(signal_file, str) or not signal_file.strip():
                    return {"error": "signal_file 必须是非空字符串"}

                # 检查文件扩展名
                if not signal_file.lower().endswith(".json"):
                    return {"error": "signal_file 必须是 .json 格式的文件"}

        # 检查采样率
        if sampling_rate is not None and sampling_rate <= 0:
            return {"error": "sampling_rate 必须是正数"}

        # 检查频率参数
        if frequency is not None and frequency <= 0:
            return {"error": "frequency 必须是正数"}

        # 检查截止频率
        if cutoff_freq is not None and cutoff_freq <= 0:
            return {"error": "cutoff_freq 必须是正数"}

        # 检查窗口大小
        if window_size is not None and (window_size <= 0 or window_size > 10000):
            return {"error": "window_size 必须是1到10000之间的正整数"}

        # 检查滤波器阶数
        if order is not None and (order <= 0 or order > 20):
            return {"error": "order 必须是1到20之间的正整数"}

        # 检查持续时间
        if duration is not None and duration <= 0:
            return {"error": "duration 必须是正数"}

        # 检查特定操作的参数
        if operation == "generate_signal":
            if sampling_rate is None:
                return {"error": "信号生成需要提供 sampling_rate 参数"}

            # 验证信号类型
            if signal_type is not None:
                valid_signal_types = [
                    "sine",
                    "cosine",
                    "square",
                    "sawtooth",
                    "triangle",
                    "chirp",
                    "white_noise",
                    "pink_noise",
                    "brown_noise",
                    "gaussian_noise",
                    "uniform_noise",
                    "impulse",
                    "step",
                    "exponential",
                    "dc",
                ]
                if signal_type not in valid_signal_types:
                    return {
                        "error": f"无效的信号类型: {signal_type}，支持的类型: {valid_signal_types}"
                    }

        if operation == "filter":
            if filter_type is None or cutoff_freq is None or sampling_rate is None:
                return {
                    "error": "滤波操作需要提供 filter_type, cutoff_freq, sampling_rate 参数"
                }
            valid_filters = [
                "lowpass",
                "highpass",
                "bandpass",
                "bandstop",
                "moving_average",
                "median",
            ]
            if filter_type not in valid_filters:
                return {
                    "error": f"无效的滤波器类型: {filter_type}，支持的类型: {valid_filters}"
                }

        return None

    def _fft_analysis(
        self,
        signal: List[float],
        sampling_rate: float = 1.0,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """快速傅里叶变换分析"""
        if not signal:
            return {"error": "信号不能为空"}

        signal_array = np.array(signal)

        # 计算FFT
        fft_result = np.fft.fft(signal_array)
        fft_magnitude = np.abs(fft_result)
        fft_phase = np.angle(fft_result)

        # 频率轴
        n = len(signal)
        frequencies = np.fft.fftfreq(n, 1 / sampling_rate)

        # 只取正频率部分（单边频谱）
        positive_freq_indices = frequencies >= 0
        positive_frequencies = frequencies[positive_freq_indices]
        positive_magnitude = fft_magnitude[positive_freq_indices]
        positive_phase = fft_phase[positive_freq_indices]

        # 找到主频率
        dominant_freq_index = np.argmax(positive_magnitude[1:]) + 1  # 跳过DC成分
        dominant_frequency = positive_frequencies[dominant_freq_index]

        # 计算功率谱密度
        power_spectrum = (positive_magnitude**2) / (n * sampling_rate)

        # 准备完整的FFT数据
        full_data = {
            "operation": "fft",
            "sampling_rate": sampling_rate,
            "frequencies": positive_frequencies.tolist(),
            "magnitude": positive_magnitude.tolist(),
            "phase": positive_phase.tolist(),
            "power_spectrum": power_spectrum.tolist(),
            "dominant_frequency": float(dominant_frequency),
            "dc_component": float(fft_magnitude[0]),
            "nyquist_frequency": sampling_rate / 2,
            "frequency_resolution": sampling_rate / n,
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "fft", output_filename)

        # 返回摘要信息
        return {
            "operation": "fft",
            "sampling_rate": sampling_rate,
            "dominant_frequency": float(dominant_frequency),
            "dc_component": float(fft_magnitude[0]),
            "nyquist_frequency": sampling_rate / 2,
            "frequency_resolution": sampling_rate / n,
            "data_file": filepath,
            "message": f"FFT分析数据已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
            "summary": {
                "frequency_points": len(positive_frequencies),
                "max_magnitude": float(np.max(positive_magnitude)),
                "spectral_centroid": float(
                    np.sum(positive_frequencies * positive_magnitude)
                    / np.sum(positive_magnitude)
                ),
            },
        }

    def _load_signal_from_file(self, filepath: str) -> Dict[str, Any]:
        """从文件加载信号数据"""
        try:
            if not os.path.exists(filepath):
                return {"error": f"信号文件不存在: {filepath}"}

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 验证文件格式
            if not isinstance(data, dict):
                return {"error": "信号文件格式错误：应为JSON对象"}

            # 尝试提取信号数据
            signal = None
            sampling_rate = None
            operation_type = data.get("operation")

            # 根据不同分析结果文件的典型字段优先提取
            if operation_type == "filter":
                # 滤波结果
                if "filtered_signal" in data:
                    signal = data["filtered_signal"]
                elif "original_signal" in data:
                    signal = data["original_signal"]
            elif operation_type == "spectral_analysis":
                # 频谱分析结果
                if "power_spectrum" in data:
                    signal = data["power_spectrum"]
            elif operation_type == "envelope_detection":
                if "amplitude_envelope" in data:
                    signal = data["amplitude_envelope"]

            # 检查是否为标准信号文件格式
            if signal is None and "signal" in data and "sampling_rate" in data:
                signal = data["signal"]
                sampling_rate = data["sampling_rate"]
            elif signal is None and "signal" in data:
                signal = data["signal"]
                sampling_rate = data.get("sampling_rate", 1.0)
            # 检查是否为FFT分析结果文件（新增）
            elif signal is None and "magnitude" in data and "frequencies" in data:
                # 对于FFT结果，我们需要重构原始信号（使用逆FFT）
                frequencies = np.array(data["frequencies"])
                magnitude = np.array(data["magnitude"])
                # 这里简化处理，直接使用magnitude作为信号数据
                signal = magnitude.tolist()
                sampling_rate = data.get("sampling_rate", 1.0)
            else:
                # 尝试其他可能的键名
                possible_signal_keys = [
                    "filtered_signal",
                    "original_signal",
                    "data",
                    "values",
                    "amplitude",
                    "y",
                    "magnitude",
                    "power_spectrum",
                ]
                for key in possible_signal_keys:
                    if key in data:
                        signal = data[key]
                        break

            if signal is None:
                return {"error": "无法在文件中找到信号数据"}

            # 验证信号数据格式
            if not isinstance(signal, list) or not signal:
                return {"error": "信号数据格式错误：应为非空列表"}

            if not all(isinstance(x, (int, float)) for x in signal):
                return {"error": "信号数据包含非数值元素"}

            return {"signal": signal, "sampling_rate": sampling_rate, "metadata": data}

        except json.JSONDecodeError:
            return {"error": "信号文件JSON格式错误"}
        except Exception as e:
            return {"error": f"读取信号文件失败: {str(e)}"}

    def _save_signal_to_file(
        self,
        signal_data: Dict[str, Any],
        operation: str,
        custom_filename: Optional[str] = None,
    ) -> str:
        """将信号数据保存到文件并返回文件路径"""
        try:
            # 使用工具函数生成唯一文件名
            filepath, filename = generate_unique_filename(
                "signal_" + operation, "json", custom_filename
            )

            # 保存数据到文件
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(signal_data, f, indent=2, ensure_ascii=False)

            return filepath
        except Exception as e:
            logging.error(f"保存信号文件失败: {str(e)}")
            return f"文件保存失败: {str(e)}"

    def _generate_signal(
        self,
        frequency: Optional[float],
        sampling_rate: float,
        duration: float,
        signal_type: str = "sine",
        noise_level: Optional[float] = None,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成各种类型的信号"""
        n_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, n_samples)

        # 基本信号生成
        if signal_type == "sine":
            if frequency is None:
                return {"error": "sine信号需要提供frequency参数"}
            signal = np.sin(2 * np.pi * frequency * t)

        elif signal_type == "cosine":
            if frequency is None:
                return {"error": "cosine信号需要提供frequency参数"}
            signal = np.cos(2 * np.pi * frequency * t)

        elif signal_type == "square":
            if frequency is None:
                return {"error": "square信号需要提供frequency参数"}
            signal = np.sign(np.sin(2 * np.pi * frequency * t))

        elif signal_type == "sawtooth":
            if frequency is None:
                return {"error": "sawtooth信号需要提供frequency参数"}
            signal = 2 * (t * frequency - np.floor(t * frequency + 0.5))

        elif signal_type == "triangle":
            if frequency is None:
                return {"error": "triangle信号需要提供frequency参数"}
            signal = 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1

        elif signal_type == "chirp":
            if frequency is None:
                return {"error": "chirp信号需要提供frequency参数"}
            # 线性调频信号，从frequency到2*frequency
            f_end = frequency * 2
            signal = np.sin(
                2
                * np.pi
                * (frequency * t + (f_end - frequency) * t**2 / (2 * duration))
            )

        # 噪声信号
        elif signal_type == "white_noise":
            # 白噪声 - 所有频率成分功率相等
            amplitude = noise_level if noise_level is not None else 1.0
            signal = amplitude * np.random.normal(0, 1, n_samples)

        elif signal_type == "pink_noise":
            # 粉红噪声 - 功率谱密度与频率成反比
            amplitude = noise_level if noise_level is not None else 1.0
            # 简化的粉红噪声生成
            white = np.random.normal(0, 1, n_samples)
            # 使用简单的滤波器近似粉红噪声
            signal = np.zeros_like(white)
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            # 简化实现
            signal = amplitude * np.convolve(white, b, mode="same")

        elif signal_type == "brown_noise":
            # 棕色噪声 - 功率谱密度与频率平方成反比
            amplitude = noise_level if noise_level is not None else 1.0
            white = np.random.normal(0, 1, n_samples)
            # 通过积分白噪声生成棕色噪声
            signal = amplitude * np.cumsum(white) / np.sqrt(n_samples)

        elif signal_type == "gaussian_noise":
            # 高斯噪声
            amplitude = noise_level if noise_level is not None else 1.0
            signal = amplitude * np.random.normal(0, 1, n_samples)

        elif signal_type == "uniform_noise":
            # 均匀分布噪声
            amplitude = noise_level if noise_level is not None else 1.0
            signal = amplitude * np.random.uniform(-1, 1, n_samples)

        # 特殊信号
        elif signal_type == "impulse":
            # 冲激信号
            signal = np.zeros(n_samples)
            if n_samples > 0:
                signal[0] = 1.0

        elif signal_type == "step":
            # 阶跃信号
            signal = np.ones(n_samples)

        elif signal_type == "exponential":
            # 指数衰减信号
            if frequency is None:
                frequency = 1.0  # 默认衰减常数
            signal = np.exp(-frequency * t)

        elif signal_type == "dc":
            # 直流信号
            amplitude = noise_level if noise_level is not None else 1.0
            signal = amplitude * np.ones(n_samples)

        else:
            # 默认生成正弦信号
            if frequency is None:
                frequency = 1.0
            signal = np.sin(2 * np.pi * frequency * t)

        # 准备完整的信号数据
        full_data = {
            "operation": "generate_signal",
            "signal_type": signal_type,
            "frequency": frequency,
            "sampling_rate": sampling_rate,
            "duration": duration,
            "noise_level": noise_level,
            "time": t.tolist(),
            "signal": signal.tolist(),
            "samples": n_samples,
            "amplitude_stats": {
                "min": float(np.min(signal)),
                "max": float(np.max(signal)),
                "mean": float(np.mean(signal)),
                "std": float(np.std(signal)),
                "rms": float(np.sqrt(np.mean(signal**2))),
            },
        }

        # 保存到文件，支持自定义文件名
        filepath = self._save_signal_to_file(full_data, "generate", output_filename)

        # 返回摘要信息，不包含大量数据
        return {
            "operation": "generate_signal",
            "signal_type": signal_type,
            "frequency": frequency,
            "sampling_rate": sampling_rate,
            "duration": duration,
            "noise_level": noise_level,
            "samples": n_samples,
            "amplitude_stats": full_data["amplitude_stats"],
            "data_file": filepath,
            "message": f"{signal_type}信号数据已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _advanced_filter(
        self,
        signal: List[float],
        filter_type: str,
        cutoff_freq: float,
        sampling_rate: float,
        order: Optional[int] = None,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """高级滤波器"""
        signal_array = np.array(signal)
        filtered_signal = signal_array.copy()

        # 归一化截止频率
        nyquist = sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        if normalized_cutoff >= 1:
            return {"error": "截止频率必须小于奈奎斯特频率"}

        if filter_type == "lowpass":
            # 简单的低通滤波器（巴特沃斯近似）
            order = order or 2
            # 使用递归滤波器
            alpha = 1 / (1 + 2 * np.pi * cutoff_freq / sampling_rate)
            filtered_signal = np.zeros_like(signal_array)
            filtered_signal[0] = signal_array[0]
            for i in range(1, len(signal_array)):
                filtered_signal[i] = (
                    alpha * filtered_signal[i - 1] + (1 - alpha) * signal_array[i]
                )

        elif filter_type == "highpass":
            # 高通滤波器
            alpha = (
                2
                * np.pi
                * cutoff_freq
                / sampling_rate
                / (1 + 2 * np.pi * cutoff_freq / sampling_rate)
            )
            filtered_signal = np.zeros_like(signal_array)
            filtered_signal[0] = signal_array[0]
            for i in range(1, len(signal_array)):
                filtered_signal[i] = alpha * (
                    filtered_signal[i - 1] + signal_array[i] - signal_array[i - 1]
                )

        elif filter_type == "bandpass":
            # 带通滤波器（需要两个截止频率）
            # 这里使用简化版本
            low_cutoff = cutoff_freq * 0.8
            high_cutoff = cutoff_freq * 1.2
            # 先低通再高通
            temp_signal = self._simple_lowpass(signal_array, high_cutoff, sampling_rate)
            filtered_signal = self._simple_highpass(
                temp_signal, low_cutoff, sampling_rate
            )

        elif filter_type == "moving_average":
            window_size = order or 5
            kernel = np.ones(window_size) / window_size
            filtered_signal = np.convolve(signal_array, kernel, mode="same")

        elif filter_type == "median":
            try:
                from scipy.signal import medfilt

                window_size = order or 5
                filtered_signal = medfilt(signal_array, kernel_size=window_size)
            except ImportError:
                # 简单的中值滤波器实现
                window_size = order or 5
                filtered_signal = np.array(signal_array)
                for i in range(len(signal_array)):
                    start = max(0, i - window_size // 2)
                    end = min(len(signal_array), i + window_size // 2 + 1)
                    filtered_signal[i] = np.median(signal_array[start:end])

        else:
            return {"error": f"不支持的滤波器类型: {filter_type}"}

        # 计算信噪比改善
        snr_improvement = self._calculate_snr_improvement(signal_array, filtered_signal)

        # 准备完整数据
        full_data = {
            "operation": "filter",
            "filter_type": filter_type,
            "cutoff_freq": cutoff_freq,
            "sampling_rate": sampling_rate,
            "order": order,
            "original_signal": signal,
            "filtered_signal": filtered_signal.tolist(),
            "snr_improvement_db": float(snr_improvement),
            "filter_stats": {
                "original_std": float(np.std(signal_array)),
                "filtered_std": float(np.std(filtered_signal)),
                "noise_reduction": float(
                    (np.std(signal_array) - np.std(filtered_signal))
                    / np.std(signal_array)
                    * 100
                ),
            },
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "filter", output_filename)

        # 返回摘要信息
        return {
            "operation": "filter",
            "filter_type": filter_type,
            "cutoff_freq": cutoff_freq,
            "sampling_rate": sampling_rate,
            "order": order,
            "snr_improvement_db": float(snr_improvement),
            "filter_stats": full_data["filter_stats"],
            "data_file": filepath,
            "message": f"滤波结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _simple_lowpass(
        self, signal: np.ndarray, cutoff_freq: float, sampling_rate: float
    ) -> np.ndarray:
        """简单低通滤波器"""
        alpha = 1 / (1 + 2 * np.pi * cutoff_freq / sampling_rate)
        filtered = np.zeros_like(signal)
        filtered[0] = signal[0]
        for i in range(1, len(signal)):
            filtered[i] = alpha * filtered[i - 1] + (1 - alpha) * signal[i]
        return filtered

    def _simple_highpass(
        self, signal: np.ndarray, cutoff_freq: float, sampling_rate: float
    ) -> np.ndarray:
        """简单高通滤波器"""
        alpha = (
            2
            * np.pi
            * cutoff_freq
            / sampling_rate
            / (1 + 2 * np.pi * cutoff_freq / sampling_rate)
        )
        filtered = np.zeros_like(signal)
        filtered[0] = signal[0]
        for i in range(1, len(signal)):
            filtered[i] = alpha * (filtered[i - 1] + signal[i] - signal[i - 1])
        return filtered

    def _calculate_snr_improvement(
        self, original: np.ndarray, filtered: np.ndarray
    ) -> float:
        """计算信噪比改善"""
        signal_power = np.var(filtered)
        noise_power = np.var(original - filtered)
        if noise_power == 0:
            return float("inf")
        snr = signal_power / noise_power
        return 10 * np.log10(snr) if snr > 0 else 0

    def _apply_window(
        self,
        signal: List[float],
        window_size: int,
        window_type: str = "hanning",
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """应用窗函数"""
        signal_array = np.array(signal)

        if window_size > len(signal_array):
            return {"error": "窗口大小不能大于信号长度"}

        # 生成窗函数
        if window_type == "hanning":
            window = np.hanning(window_size)
        elif window_type == "hamming":
            window = np.hamming(window_size)
        elif window_type == "blackman":
            window = np.blackman(window_size)
        elif window_type == "bartlett":
            window = np.bartlett(window_size)
        elif window_type == "kaiser":
            # 使用beta=5作为默认值
            window = np.kaiser(window_size, 5)
        else:
            window = np.hanning(window_size)

        # 分段应用窗函数
        hop_length = window_size // 2
        n_frames = (len(signal_array) - window_size) // hop_length + 1

        windowed_frames = []
        for i in range(n_frames):
            start = i * hop_length
            end = start + window_size
            frame = signal_array[start:end]
            windowed_frame = frame * window
            windowed_frames.append(windowed_frame.tolist())

        # 重建信号（简化版本）
        windowed_signal = np.zeros_like(signal_array)
        for i, frame in enumerate(windowed_frames):
            start = i * hop_length
            end = start + window_size
            if end <= len(windowed_signal):
                windowed_signal[start:end] += frame

        # 准备完整数据
        full_data = {
            "operation": "windowing",
            "window_type": window_type,
            "window_size": window_size,
            "window_function": window.tolist(),
            "windowed_frames": windowed_frames,
            "windowed_signal": windowed_signal.tolist(),
            "frame_count": n_frames,
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "window", output_filename)

        # 返回摘要信息
        return {
            "operation": "windowing",
            "window_type": window_type,
            "window_size": window_size,
            "frame_count": n_frames,
            "data_file": filepath,
            "message": f"窗函数处理结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
            "window_stats": {
                "window_energy": float(np.sum(window**2)),
                "window_gain": float(np.mean(window)),
                "signal_reduction": float(
                    np.sum(windowed_signal**2) / np.sum(signal_array**2)
                ),
            },
        }

    def _autocorrelation(
        self, signal: List[float], output_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """自相关分析"""
        signal_array = np.array(signal)
        n = len(signal_array)

        # 计算自相关
        autocorr = np.correlate(signal_array, signal_array, mode="full")
        autocorr = autocorr[n - 1 :]  # 只取正延迟部分

        # 归一化
        autocorr_normalized = autocorr / autocorr[0]

        # 找到第一个过零点
        zero_crossing = None
        for i in range(1, len(autocorr_normalized)):
            if autocorr_normalized[i - 1] > 0 and autocorr_normalized[i] <= 0:
                zero_crossing = i
                break

        # 计算自相关特性
        max_correlation = float(autocorr[0])
        correlation_length = np.sum(
            np.abs(autocorr_normalized) > 0.1
        )  # 10%阈值的相关长度

        # 准备完整数据
        full_data = {
            "operation": "autocorrelation",
            "signal_length": n,
            "autocorrelation": autocorr.tolist(),
            "normalized_autocorrelation": autocorr_normalized.tolist(),
            "first_zero_crossing": zero_crossing,
            "max_correlation": max_correlation,
            "correlation_length": int(correlation_length),
            "autocorr_properties": {
                "peak_value": float(np.max(autocorr)),
                "decay_rate": (
                    float(autocorr_normalized[min(10, len(autocorr_normalized) - 1)])
                    if len(autocorr_normalized) > 10
                    else None
                ),
            },
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "autocorr", output_filename)

        # 返回摘要信息
        return {
            "operation": "autocorrelation",
            "signal_length": n,
            "first_zero_crossing": zero_crossing,
            "max_correlation": max_correlation,
            "correlation_length": int(correlation_length),
            "autocorr_properties": full_data["autocorr_properties"],
            "data_file": filepath,
            "message": f"自相关分析结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _crosscorrelation(
        self,
        signal1: List[float],
        signal2: List[float],
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """互相关分析"""
        if not signal1 or not signal2:
            return {"error": "两个信号都不能为空"}

        signal1_array = np.array(signal1)
        signal2_array = np.array(signal2)

        # 计算互相关
        crosscorr = np.correlate(signal1_array, signal2_array, mode="full")

        # 找到最大相关值和对应的延迟
        max_corr_index = np.argmax(np.abs(crosscorr))
        max_correlation = crosscorr[max_corr_index]
        delay = max_corr_index - len(signal2_array) + 1

        # 归一化互相关
        norm_factor = np.sqrt(np.sum(signal1_array**2) * np.sum(signal2_array**2))
        normalized_crosscorr = crosscorr / norm_factor if norm_factor > 0 else crosscorr

        # 准备完整数据
        full_data = {
            "operation": "crosscorrelation",
            "signal1_length": len(signal1),
            "signal2_length": len(signal2),
            "crosscorrelation": crosscorr.tolist(),
            "normalized_crosscorrelation": normalized_crosscorr.tolist(),
            "max_correlation": float(max_correlation),
            "normalized_max_correlation": float(normalized_crosscorr[max_corr_index]),
            "optimal_delay": int(delay),
            "correlation_coefficient": (
                float(np.corrcoef(signal1, signal2)[0, 1])
                if len(signal1) == len(signal2)
                else None
            ),
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "crosscorr", output_filename)

        # 返回摘要信息
        return {
            "operation": "crosscorrelation",
            "signal1_length": len(signal1),
            "signal2_length": len(signal2),
            "max_correlation": float(max_correlation),
            "normalized_max_correlation": float(normalized_crosscorr[max_corr_index]),
            "optimal_delay": int(delay),
            "correlation_coefficient": full_data["correlation_coefficient"],
            "data_file": filepath,
            "message": f"互相关分析结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _spectral_analysis(
        self,
        signal: List[float],
        sampling_rate: float,
        window_size: Optional[int] = None,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """频谱分析"""
        signal_array = np.array(signal)

        if window_size and window_size < len(signal_array):
            # 分段进行频谱分析
            hop_length = window_size // 2
            n_frames = (len(signal_array) - window_size) // hop_length + 1

            spectrograms = []
            for i in range(n_frames):
                start = i * hop_length
                end = start + window_size
                frame = signal_array[start:end]

                # 应用窗函数
                windowed_frame = frame * np.hanning(len(frame))

                # 计算FFT
                fft_result = np.fft.fft(windowed_frame)
                magnitude = np.abs(fft_result[: len(fft_result) // 2])
                spectrograms.append(magnitude.tolist())

            frequencies = np.fft.fftfreq(window_size, 1 / sampling_rate)[
                : window_size // 2
            ]

            # 计算平均功率谱
            avg_power_spectrum = np.mean(spectrograms, axis=0)

        else:
            # 整个信号的频谱分析
            fft_result = np.fft.fft(signal_array)
            power_spectrum = np.abs(fft_result) ** 2
            frequencies = np.fft.fftfreq(len(signal_array), 1 / sampling_rate)

            # 只取正频率部分
            positive_freq_indices = frequencies >= 0
            positive_frequencies = frequencies[positive_freq_indices]
            positive_power = power_spectrum[positive_freq_indices]

            frequencies = positive_frequencies
            avg_power_spectrum = positive_power
            spectrograms = [positive_power.tolist()]

        # 谱重心
        spectral_centroid = (
            np.sum(frequencies * avg_power_spectrum) / np.sum(avg_power_spectrum)
            if np.sum(avg_power_spectrum) > 0
            else 0
        )

        # 谱带宽
        spectral_bandwidth = (
            np.sqrt(
                np.sum(((frequencies - spectral_centroid) ** 2) * avg_power_spectrum)
                / np.sum(avg_power_spectrum)
            )
            if np.sum(avg_power_spectrum) > 0
            else 0
        )

        # 谱滚降
        cumulative_power = np.cumsum(avg_power_spectrum)
        total_power = cumulative_power[-1]
        rolloff_threshold = 0.85 * total_power
        rolloff_index = np.where(cumulative_power >= rolloff_threshold)[0]
        spectral_rolloff = (
            frequencies[rolloff_index[0]] if len(rolloff_index) > 0 else frequencies[-1]
        )

        # 准备完整数据
        full_data = {
            "operation": "spectral_analysis",
            "sampling_rate": sampling_rate,
            "window_size": window_size,
            "frequencies": frequencies.tolist(),
            "power_spectrum": avg_power_spectrum.tolist(),
            "spectrogram": spectrograms,
            "spectral_features": {
                "spectral_centroid": float(spectral_centroid),
                "spectral_bandwidth": float(spectral_bandwidth),
                "spectral_rolloff": float(spectral_rolloff),
                "total_power": float(np.sum(avg_power_spectrum)),
                "peak_frequency": float(frequencies[np.argmax(avg_power_spectrum)]),
            },
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "spectral", output_filename)

        # 返回摘要信息
        return {
            "operation": "spectral_analysis",
            "sampling_rate": sampling_rate,
            "window_size": window_size,
            "spectral_features": full_data["spectral_features"],
            "data_file": filepath,
            "message": f"频谱分析数据已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
            "summary": {
                "frequency_points": len(frequencies),
                "frequency_range": [
                    float(np.min(frequencies)),
                    float(np.max(frequencies)),
                ],
                "power_range": [
                    float(np.min(avg_power_spectrum)),
                    float(np.max(avg_power_spectrum)),
                ],
            },
        }

    def _signal_metrics(
        self, signal: List[float], sampling_rate: float
    ) -> Dict[str, Any]:
        """信号度量"""
        if not signal:
            return {"error": "信号不能为空"}

        signal_array = np.array(signal)

        # 基本统计量
        mean_value = np.mean(signal_array)
        std_value = np.std(signal_array)
        rms_value = np.sqrt(np.mean(signal_array**2))
        peak_value = np.max(np.abs(signal_array))

        # 峰值因子和波峰因子
        crest_factor = peak_value / rms_value if rms_value > 0 else 0

        # 偏度和峰度
        skewness = (
            np.mean(((signal_array - mean_value) / std_value) ** 3)
            if std_value > 0
            else 0
        )
        kurtosis = (
            np.mean(((signal_array - mean_value) / std_value) ** 4)
            if std_value > 0
            else 0
        )

        # 过零率
        zero_crossings = np.sum(np.diff(np.sign(signal_array)) != 0)
        zero_crossing_rate = zero_crossings / (len(signal_array) - 1)

        # 信号能量和功率
        signal_energy = np.sum(signal_array**2)
        signal_power = signal_energy / len(signal_array)

        # 动态范围
        min_val = np.min(signal_array)
        max_val = np.max(signal_array)
        dynamic_range = max_val - min_val

        # THD (总谐波失真) - 简化计算
        fft_result = np.fft.fft(signal_array)
        magnitude = np.abs(fft_result[: len(fft_result) // 2])
        fundamental_index = np.argmax(magnitude[1:]) + 1  # 跳过DC
        fundamental_power = magnitude[fundamental_index] ** 2

        harmonic_power = 0
        for i in range(2, 6):  # 计算前5个谐波
            harmonic_index = fundamental_index * i
            if harmonic_index < len(magnitude):
                harmonic_power += magnitude[harmonic_index] ** 2

        thd = (
            np.sqrt(harmonic_power / fundamental_power) if fundamental_power > 0 else 0
        )

        return {
            "operation": "signal_metrics",
            "sampling_rate": sampling_rate,
            "signal_length": len(signal),
            "duration": len(signal) / sampling_rate,
            "amplitude_metrics": {
                "mean": float(mean_value),
                "std": float(std_value),
                "rms": float(rms_value),
                "peak": float(peak_value),
                "min": float(min_val),
                "max": float(max_val),
                "dynamic_range": float(dynamic_range),
                "crest_factor": float(crest_factor),
            },
            "statistical_metrics": {
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "zero_crossing_rate": float(zero_crossing_rate),
            },
            "energy_metrics": {
                "signal_energy": float(signal_energy),
                "signal_power": float(signal_power),
                "average_power_db": (
                    float(10 * np.log10(signal_power))
                    if signal_power > 0
                    else float("-inf")
                ),
            },
            "distortion_metrics": {
                "thd": float(thd),
                "thd_db": float(20 * np.log10(thd)) if thd > 0 else float("-inf"),
            },
        }

    def _convolution(
        self,
        signal1: List[float],
        signal2: List[float],
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """卷积运算"""
        if not signal1 or not signal2:
            return {"error": "两个信号都不能为空"}

        signal1_array = np.array(signal1)
        signal2_array = np.array(signal2)

        # 计算卷积
        conv_result = np.convolve(signal1_array, signal2_array, mode="full")
        conv_valid = np.convolve(signal1_array, signal2_array, mode="valid")
        conv_same = np.convolve(signal1_array, signal2_array, mode="same")

        # 准备完整数据
        full_data = {
            "operation": "convolution",
            "signal1_length": len(signal1),
            "signal2_length": len(signal2),
            "signal1": signal1,
            "signal2": signal2,
            "convolution_full": conv_result.tolist(),
            "convolution_valid": conv_valid.tolist(),
            "convolution_same": conv_same.tolist(),
            "output_lengths": {
                "full": len(conv_result),
                "valid": len(conv_valid),
                "same": len(conv_same),
            },
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "convolution", output_filename)

        # 返回摘要信息
        return {
            "operation": "convolution",
            "signal1_length": len(signal1),
            "signal2_length": len(signal2),
            "output_lengths": full_data["output_lengths"],
            "data_file": filepath,
            "message": f"卷积运算结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
            "summary": {
                "full_max": float(np.max(np.abs(conv_result))),
                "full_energy": float(np.sum(conv_result**2)),
                "valid_points": len(conv_valid),
            },
        }

    def _deconvolution(
        self,
        signal: List[float],
        kernel: List[float],
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """反卷积运算"""
        if not signal or not kernel:
            return {"error": "信号和核都不能为空"}

        signal_array = np.array(signal)
        kernel_array = np.array(kernel)

        # 使用频域反卷积
        signal_fft = np.fft.fft(signal_array, len(signal_array) + len(kernel_array) - 1)
        kernel_fft = np.fft.fft(kernel_array, len(signal_array) + len(kernel_array) - 1)

        # 添加正则化防止除零
        regularization = 1e-10
        deconv_fft = signal_fft / (kernel_fft + regularization)
        deconv_result = np.real(np.fft.ifft(deconv_fft))

        # 截取有效部分
        deconv_result = deconv_result[: len(signal_array)]

        # 准备完整数据
        full_data = {
            "operation": "deconvolution",
            "signal_length": len(signal),
            "kernel_length": len(kernel),
            "original_signal": signal,
            "kernel": kernel,
            "deconvolution_result": deconv_result.tolist(),
            "regularization_used": regularization,
        }

        # 保存到文件
        filepath = self._save_signal_to_file(
            full_data, "deconvolution", output_filename
        )

        # 返回摘要信息
        return {
            "operation": "deconvolution",
            "signal_length": len(signal),
            "kernel_length": len(kernel),
            "regularization_used": regularization,
            "data_file": filepath,
            "message": f"反卷积运算结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
            "result_stats": {
                "max_value": float(np.max(np.abs(deconv_result))),
                "mean_value": float(np.mean(deconv_result)),
                "std_value": float(np.std(deconv_result)),
            },
        }

    def _envelope_detection(
        self, signal: List[float], output_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """包络检测"""
        signal_array = np.array(signal)

        # 计算解析信号（希尔伯特变换）
        try:
            from scipy.signal import hilbert

            analytic_signal = hilbert(signal_array)
        except ImportError:
            # 简化版本：使用绝对值作为包络
            analytic_signal = signal_array + 1j * np.zeros_like(signal_array)

        # 提取幅度包络和相位
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        instantaneous_frequency = np.diff(np.unwrap(instantaneous_phase)) / (2 * np.pi)

        envelope_stats = {
            "max_amplitude": float(np.max(amplitude_envelope)),
            "min_amplitude": float(np.min(amplitude_envelope)),
            "mean_amplitude": float(np.mean(amplitude_envelope)),
            "amplitude_variation": float(np.std(amplitude_envelope)),
        }

        # 准备完整数据
        full_data = {
            "operation": "envelope_detection",
            "original_signal": signal,
            "amplitude_envelope": amplitude_envelope.tolist(),
            "instantaneous_phase": instantaneous_phase.tolist(),
            "instantaneous_frequency": instantaneous_frequency.tolist(),
            "envelope_stats": envelope_stats,
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "envelope", output_filename)

        # 返回摘要信息
        return {
            "operation": "envelope_detection",
            "envelope_stats": envelope_stats,
            "data_file": filepath,
            "message": f"包络检测结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
            "summary": {
                "envelope_points": len(amplitude_envelope),
                "frequency_points": len(instantaneous_frequency),
                "amplitude_range": [
                    envelope_stats["min_amplitude"],
                    envelope_stats["max_amplitude"],
                ],
            },
        }

    def _phase_analysis(
        self,
        signal: List[float],
        sampling_rate: float,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """相位分析"""
        if not signal:
            return {"error": "信号不能为空"}

        signal_array = np.array(signal)

        # 计算解析信号（希尔伯特变换）
        try:
            from scipy.signal import hilbert

            analytic_signal = hilbert(signal_array)
        except ImportError:
            # 简化版本：使用FFT计算希尔伯特变换
            fft_result = np.fft.fft(signal_array)
            n = len(signal_array)
            h = np.zeros(n)
            h[0] = 1
            h[1 : n // 2] = 2
            h[n // 2] = 1
            analytic_signal = np.fft.ifft(fft_result * h)

        # 提取瞬时相位
        instantaneous_phase = np.angle(analytic_signal)
        instantaneous_frequency = (
            np.diff(np.unwrap(instantaneous_phase)) * sampling_rate / (2 * np.pi)
        )

        # 计算相位统计
        phase_unwrapped = np.unwrap(instantaneous_phase)
        phase_variance = np.var(instantaneous_phase)
        phase_linearity = np.corrcoef(np.arange(len(phase_unwrapped)), phase_unwrapped)[
            0, 1
        ]

        # FFT相位分析
        fft_result = np.fft.fft(signal_array)
        fft_phase = np.angle(fft_result)
        frequencies = np.fft.fftfreq(len(signal_array), 1 / sampling_rate)

        # 只取正频率部分
        positive_freq_indices = frequencies >= 0
        positive_frequencies = frequencies[positive_freq_indices]
        positive_phase = fft_phase[positive_freq_indices]
        positive_magnitude = np.abs(fft_result[positive_freq_indices])

        # 准备完整数据
        full_data = {
            "operation": "phase_analysis",
            "sampling_rate": sampling_rate,
            "instantaneous_phase": instantaneous_phase.tolist(),
            "instantaneous_frequency": instantaneous_frequency.tolist(),
            "frequencies": positive_frequencies.tolist(),
            "fft_phase": positive_phase.tolist(),
            "magnitude": positive_magnitude.tolist(),
            "phase_statistics": {
                "phase_variance": float(phase_variance),
                "phase_linearity": float(phase_linearity),
                "mean_frequency": float(np.mean(instantaneous_frequency)),
                "frequency_std": float(np.std(instantaneous_frequency)),
            },
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "phase", output_filename)

        # 返回摘要信息
        return {
            "operation": "phase_analysis",
            "sampling_rate": sampling_rate,
            "phase_statistics": full_data["phase_statistics"],
            "data_file": filepath,
            "message": f"相位分析结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
            "summary": {
                "phase_points": len(instantaneous_phase),
                "frequency_range": [
                    float(np.min(instantaneous_frequency)),
                    float(np.max(instantaneous_frequency)),
                ],
                "phase_range": [
                    float(np.min(instantaneous_phase)),
                    float(np.max(instantaneous_phase)),
                ],
            },
        }

    def _noise_reduction(
        self,
        signal: List[float],
        method: str,
        window_size: int,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """噪声减少"""
        if not signal:
            return {"error": "信号不能为空"}

        signal_array = np.array(signal)
        denoised = signal_array.copy()

        if method == "moving_average":
            # 移动平均滤波
            kernel = np.ones(window_size) / window_size
            denoised = np.convolve(signal_array, kernel, mode="same")

        elif method == "median":
            # 中值滤波
            for i in range(len(signal_array)):
                start = max(0, i - window_size // 2)
                end = min(len(signal_array), i + window_size // 2 + 1)
                denoised[i] = np.median(signal_array[start:end])

        elif method == "gaussian":
            # 高斯滤波
            sigma = window_size / 6  # 标准差
            kernel_size = 2 * window_size + 1
            kernel = np.exp(-np.arange(kernel_size) ** 2 / (2 * sigma**2))
            kernel = kernel / np.sum(kernel)
            denoised = np.convolve(signal_array, kernel, mode="same")

        elif method == "savitzky_golay":
            # Savitzky-Golay滤波（简化版本）
            try:
                from scipy.signal import savgol_filter

                denoised = savgol_filter(signal_array, window_size, 3)
            except ImportError:
                # 退回到移动平均
                kernel = np.ones(window_size) / window_size
                denoised = np.convolve(signal_array, kernel, mode="same")

        elif method == "wiener":
            # 维纳滤波（简化版本）
            # 估计噪声方差
            noise_var = np.var(signal_array - np.mean(signal_array))
            signal_power = np.var(signal_array)
            wiener_gain = signal_power / (signal_power + noise_var)
            denoised = signal_array * wiener_gain

        else:
            return {"error": f"不支持的噪声减少方法: {method}"}

        # 计算性能指标
        mse = np.mean((signal_array - denoised) ** 2)
        snr_original = np.var(signal_array) / np.var(
            signal_array - np.mean(signal_array)
        )
        snr_denoised = np.var(denoised) / np.var(denoised - np.mean(denoised))
        snr_improvement = (
            10 * np.log10(snr_denoised / snr_original) if snr_original > 0 else 0
        )

        # 准备完整数据
        full_data = {
            "operation": "noise_reduction",
            "method": method,
            "window_size": window_size,
            "original_signal": signal,
            "denoised_signal": denoised.tolist(),
            "performance_metrics": {
                "mse": float(mse),
                "snr_improvement_db": float(snr_improvement),
                "noise_reduction_ratio": float(
                    (np.std(signal_array) - np.std(denoised)) / np.std(signal_array)
                ),
            },
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "denoise", output_filename)

        # 返回摘要信息
        return {
            "operation": "noise_reduction",
            "method": method,
            "window_size": window_size,
            "performance_metrics": full_data["performance_metrics"],
            "data_file": filepath,
            "message": f"噪声减少结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _resampling(
        self,
        signal: List[float],
        original_rate: float,
        target_rate: float,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """重采样"""
        if not signal:
            return {"error": "信号不能为空"}

        if original_rate <= 0 or target_rate <= 0:
            return {"error": "采样率必须为正数"}

        signal_array = np.array(signal)
        ratio = target_rate / original_rate

        if ratio == 1.0:
            resampled = signal_array
        elif ratio > 1.0:
            # 上采样：插值
            new_length = int(len(signal_array) * ratio)
            old_indices = np.arange(len(signal_array))
            new_indices = np.linspace(0, len(signal_array) - 1, new_length)
            resampled = np.interp(new_indices, old_indices, signal_array)
        else:
            # 下采样：抽取
            decimation_factor = int(1 / ratio)
            resampled = signal_array[::decimation_factor]

        # 准备完整数据
        full_data = {
            "operation": "resampling",
            "original_rate": original_rate,
            "target_rate": target_rate,
            "ratio": float(ratio),
            "original_signal": signal,
            "resampled_signal": resampled.tolist(),
            "length_change": {
                "original_length": len(signal),
                "resampled_length": len(resampled),
                "length_ratio": len(resampled) / len(signal),
            },
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "resample", output_filename)

        # 返回摘要信息
        return {
            "operation": "resampling",
            "original_rate": original_rate,
            "target_rate": target_rate,
            "ratio": float(ratio),
            "length_change": full_data["length_change"],
            "data_file": filepath,
            "message": f"重采样结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _amplitude_modulation(
        self,
        signal: List[float],
        carrier_freq: float,
        sampling_rate: float,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """幅度调制"""
        if not signal:
            return {"error": "信号不能为空"}

        signal_array = np.array(signal)
        t = np.arange(len(signal_array)) / sampling_rate

        # 生成载波信号
        carrier = np.cos(2 * np.pi * carrier_freq * t)

        # AM调制
        modulated = (1 + 0.5 * signal_array) * carrier

        # 准备完整数据
        full_data = {
            "operation": "amplitude_modulation",
            "carrier_freq": carrier_freq,
            "sampling_rate": sampling_rate,
            "time": t.tolist(),
            "original_signal": signal,
            "carrier_signal": carrier.tolist(),
            "modulated_signal": modulated.tolist(),
            "modulation_stats": {
                "modulation_index": 0.5,
                "carrier_power": float(np.mean(carrier**2)),
                "modulated_power": float(np.mean(modulated**2)),
            },
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "modulation", output_filename)

        # 返回摘要信息
        return {
            "operation": "amplitude_modulation",
            "carrier_freq": carrier_freq,
            "sampling_rate": sampling_rate,
            "modulation_stats": full_data["modulation_stats"],
            "data_file": filepath,
            "message": f"幅度调制结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _amplitude_demodulation(
        self,
        signal: List[float],
        carrier_freq: float,
        sampling_rate: float,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """幅度解调"""
        if not signal:
            return {"error": "信号不能为空"}

        signal_array = np.array(signal)
        t = np.arange(len(signal_array)) / sampling_rate

        # 包络检测
        envelope = np.abs(signal_array)

        # 低通滤波去除载波
        cutoff_freq = carrier_freq / 10  # 设置截止频率为载波频率的1/10
        filtered_demod = self._simple_lowpass(envelope, cutoff_freq, sampling_rate)

        # 去除直流分量
        filtered_demod = filtered_demod - np.mean(filtered_demod)

        # 准备完整数据
        full_data = {
            "operation": "amplitude_demodulation",
            "carrier_freq": carrier_freq,
            "sampling_rate": sampling_rate,
            "cutoff_freq": cutoff_freq,
            "modulated_signal": signal,
            "envelope": envelope.tolist(),
            "demodulated_signal": filtered_demod.tolist(),
            "demodulation_stats": {
                "envelope_mean": float(np.mean(envelope)),
                "envelope_std": float(np.std(envelope)),
                "demod_power": float(np.mean(filtered_demod**2)),
            },
        }

        # 保存到文件
        filepath = self._save_signal_to_file(full_data, "demodulation", output_filename)

        # 返回摘要信息
        return {
            "operation": "amplitude_demodulation",
            "carrier_freq": carrier_freq,
            "sampling_rate": sampling_rate,
            "cutoff_freq": cutoff_freq,
            "demodulation_stats": full_data["demodulation_stats"],
            "data_file": filepath,
            "message": f"幅度解调结果已保存到文件: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }
