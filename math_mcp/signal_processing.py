# -*- coding: utf-8 -*-
"""
Signal Processing Calculation Module
Provides digital signal processing functions
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
    """Signal Processing Calculator Class"""

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
        Signal Processing Tool

        Args:
            operation: The type of operation to perform
            signal: Input signal data (mutually exclusive with signal_file)
            signal_file: Path to the signal file (mutually exclusive with signal)
            sampling_rate: The sampling rate of the signal
            frequency: The frequency for signal generation or analysis
            filter_type: The type of filter to apply
            cutoff_freq: The cutoff frequency for the filter
            window_size: The size of the window for windowing operations
            overlap: The overlap percentage for windowing
            order: The order of the filter
            duration: The duration of the generated signal
            noise_level: The level of noise to add or generate
            signal_type: The type of signal to generate
            output_filename: Name for the output file (without path, optional)
        """
        # Handle signal input (from file or direct data)
        actual_signal = signal
        actual_sampling_rate = sampling_rate

        # Check if a valid signal file path is provided
        if signal_file is not None and signal_file.strip():
            # Resolve file path (handles relative paths)
            full_signal_file_path = resolve_signal_file_path(signal_file)

            # Load signal from file
            load_result = self._load_signal_from_file(full_signal_file_path)
            if "error" in load_result:
                return load_result

            actual_signal = load_result["signal"]
            # If the file contains a sampling rate and it's not specified by the user, use the one from the file
            if sampling_rate is None and load_result["sampling_rate"] is not None:
                actual_sampling_rate = load_result["sampling_rate"]

        # Parameter validation (using original arguments, not processed ones)
        validation_result = self._validate_parameters(
            operation,
            signal,  # Use the original signal parameter, not actual_signal
            signal_file,
            sampling_rate,  # Use the original sampling_rate parameter
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
                )  # Use cutoff_freq as the new sampling rate
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
                return {"error": f"Unsupported operation: {operation}"}

        except Exception as e:
            return {"error": f"Signal processing error: {str(e)}"}

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
        """Validate parameters"""
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
                "error": f"Invalid operation type: {operation}, supported operations: {valid_operations}"
            }

        # Check for operations that require a signal
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
            # Check if both are provided (excluding default None values)
            signal_provided = signal is not None and (
                len(signal) > 0 if isinstance(signal, list) else True
            )
            signal_file_provided = signal_file is not None and (
                signal_file.strip() != "" if isinstance(signal_file, str) else True
            )

            # Check if signal data is provided (either direct data or a file path)
            if not signal_provided and not signal_file_provided:
                return {
                    "error": f"Operation {operation} requires either a 'signal' or 'signal_file' parameter"
                }

            if signal_provided and signal_file_provided:
                return {
                    "error": "Cannot provide both 'signal' and 'signal_file' parameters. Please choose one."
                }

            # If direct signal data is provided, validate it
            if signal is not None:
                if len(signal) == 0:
                    return {"error": "The 'signal' list cannot be empty"}
                if not all(isinstance(x, (int, float)) for x in signal):
                    return {"error": "All elements in 'signal' must be numeric"}

            # If a file path is provided, validate the file format
            if signal_file is not None:
                if not isinstance(signal_file, str) or not signal_file.strip():
                    return {"error": "'signal_file' must be a non-empty string"}

                # Check the file extension
                if not signal_file.lower().endswith(".json"):
                    return {"error": "'signal_file' must be a .json file"}

        # Check sampling rate
        if sampling_rate is not None and sampling_rate <= 0:
            return {"error": "'sampling_rate' must be a positive number"}

        # Check frequency parameter
        if frequency is not None and frequency <= 0:
            return {"error": "'frequency' must be a positive number"}

        # Check cutoff frequency
        if cutoff_freq is not None and cutoff_freq <= 0:
            return {"error": "'cutoff_freq' must be a positive number"}

        # Check window size
        if window_size is not None and (window_size <= 0 or window_size > 10000):
            return {
                "error": "'window_size' must be a positive integer between 1 and 10000"
            }

        # Check filter order
        if order is not None and (order <= 0 or order > 20):
            return {"error": "'order' must be a positive integer between 1 and 20"}

        # Check duration
        if duration is not None and duration <= 0:
            return {"error": "'duration' must be a positive number"}

        # Check parameters for specific operations
        if operation == "generate_signal":
            if sampling_rate is None:
                return {
                    "error": "Signal generation requires the 'sampling_rate' parameter"
                }

            # Validate signal type
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
                        "error": f"Invalid signal_type: {signal_type}, supported types: {valid_signal_types}"
                    }

        if operation == "filter":
            if filter_type is None or cutoff_freq is None or sampling_rate is None:
                return {
                    "error": "Filter operation requires 'filter_type', 'cutoff_freq', and 'sampling_rate' parameters"
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
                    "error": f"Invalid filter_type: {filter_type}, supported types: {valid_filters}"
                }

        return None

    def _fft_analysis(
        self,
        signal: List[float],
        sampling_rate: float = 1.0,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fast Fourier Transform Analysis"""
        if not signal:
            return {"error": "Input signal cannot be empty"}

        signal_array = np.array(signal)

        # Calculate FFT
        fft_result = np.fft.fft(signal_array)
        fft_magnitude = np.abs(fft_result)
        fft_phase = np.angle(fft_result)

        # Frequency axis
        n = len(signal)
        frequencies = np.fft.fftfreq(n, 1 / sampling_rate)

        # Take only the positive frequency part (single-sided spectrum)
        positive_freq_indices = frequencies >= 0
        positive_frequencies = frequencies[positive_freq_indices]
        positive_magnitude = fft_magnitude[positive_freq_indices]
        positive_phase = fft_phase[positive_freq_indices]

        # Find the dominant frequency
        dominant_freq_index = (
            np.argmax(positive_magnitude[1:]) + 1
        )  # Skip the DC component
        dominant_frequency = positive_frequencies[dominant_freq_index]

        # Calculate Power Spectral Density
        power_spectrum = (positive_magnitude**2) / (n * sampling_rate)

        # Prepare the full FFT data
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

        # Save to file
        filepath = self._save_signal_to_file(full_data, "fft", output_filename)

        # Return summary information
        return {
            "operation": "fft",
            "sampling_rate": sampling_rate,
            "dominant_frequency": float(dominant_frequency),
            "dc_component": float(fft_magnitude[0]),
            "nyquist_frequency": sampling_rate / 2,
            "frequency_resolution": sampling_rate / n,
            "data_file": filepath,
            "message": f"FFT analysis data has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
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
        """Load signal data from a file"""
        try:
            if not os.path.exists(filepath):
                return {"error": f"Signal file not found: {filepath}"}

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate file format
            if not isinstance(data, dict):
                return {"error": "Invalid signal file format: should be a JSON object"}

            # Try to extract signal data
            signal = None
            sampling_rate = None
            operation_type = data.get("operation")

            # Prioritize extraction based on typical fields of different analysis result files
            if operation_type == "filter":
                # Filter result
                if "filtered_signal" in data:
                    signal = data["filtered_signal"]
                elif "original_signal" in data:
                    signal = data["original_signal"]
            elif operation_type == "spectral_analysis":
                # Spectral analysis result
                if "power_spectrum" in data:
                    signal = data["power_spectrum"]
            elif operation_type == "envelope_detection":
                if "amplitude_envelope" in data:
                    signal = data["amplitude_envelope"]

            # Check for standard signal file format
            if signal is None and "signal" in data and "sampling_rate" in data:
                signal = data["signal"]
                sampling_rate = data["sampling_rate"]
            elif signal is None and "signal" in data:
                signal = data["signal"]
                sampling_rate = data.get("sampling_rate", 1.0)
            # Check for FFT analysis result file (new)
            elif signal is None and "magnitude" in data and "frequencies" in data:
                # For FFT results, we would need to reconstruct the original signal (using IFFT)
                # This is a simplified handling, directly using magnitude as signal data
                frequencies = np.array(data["frequencies"])
                magnitude = np.array(data["magnitude"])
                signal = magnitude.tolist()
                sampling_rate = data.get("sampling_rate", 1.0)
            else:
                # Try other possible key names
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
                return {"error": "Could not find signal data in the file"}

            # Validate signal data format
            if not isinstance(signal, list) or not signal:
                return {
                    "error": "Invalid signal data format: should be a non-empty list"
                }

            if not all(isinstance(x, (int, float)) for x in signal):
                return {"error": "Signal data contains non-numeric elements"}

            return {"signal": signal, "sampling_rate": sampling_rate, "metadata": data}

        except json.JSONDecodeError:
            return {"error": "Invalid JSON format in signal file"}
        except Exception as e:
            return {"error": f"Failed to read signal file: {str(e)}"}

    def _save_signal_to_file(
        self,
        signal_data: Dict[str, Any],
        operation: str,
        custom_filename: Optional[str] = None,
    ) -> str:
        """Save signal data to a file and return the file path"""
        try:
            # Use the utility function to generate a unique filename
            filepath, filename = generate_unique_filename(
                "signal_" + operation, "json", custom_filename
            )

            # Save data to the file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(signal_data, f, indent=2, ensure_ascii=False)

            return filepath
        except Exception as e:
            logging.error(f"Failed to save signal file: {str(e)}")
            return f"File save failed: {str(e)}"

    def _generate_signal(
        self,
        frequency: Optional[float],
        sampling_rate: float,
        duration: float,
        signal_type: str = "sine",
        noise_level: Optional[float] = None,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate various types of signals"""
        n_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, n_samples)

        # Basic signal generation
        if signal_type == "sine":
            if frequency is None:
                return {"error": "'sine' signal requires a 'frequency' parameter"}
            signal = np.sin(2 * np.pi * frequency * t)

        elif signal_type == "cosine":
            if frequency is None:
                return {"error": "'cosine' signal requires a 'frequency' parameter"}
            signal = np.cos(2 * np.pi * frequency * t)

        elif signal_type == "square":
            if frequency is None:
                return {"error": "'square' signal requires a 'frequency' parameter"}
            signal = np.sign(np.sin(2 * np.pi * frequency * t))

        elif signal_type == "sawtooth":
            if frequency is None:
                return {"error": "'sawtooth' signal requires a 'frequency' parameter"}
            signal = 2 * (t * frequency - np.floor(t * frequency + 0.5))

        elif signal_type == "triangle":
            if frequency is None:
                return {"error": "'triangle' signal requires a 'frequency' parameter"}
            signal = 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1

        elif signal_type == "chirp":
            if frequency is None:
                return {"error": "'chirp' signal requires a 'frequency' parameter"}
            # Linear chirp signal, from frequency to 2*frequency
            f_end = frequency * 2
            signal = np.sin(
                2
                * np.pi
                * (frequency * t + (f_end - frequency) * t**2 / (2 * duration))
            )

        # Noise signals
        elif signal_type == "white_noise":
            # White noise - equal power at all frequencies
            amplitude = noise_level if noise_level is not None else 1.0
            signal = amplitude * np.random.normal(0, 1, n_samples)

        elif signal_type == "pink_noise":
            # Pink noise - power spectral density is inversely proportional to frequency
            amplitude = noise_level if noise_level is not None else 1.0
            # Simplified pink noise generation
            white = np.random.normal(0, 1, n_samples)
            # Approximate pink noise using a simple filter
            signal = np.zeros_like(white)
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            # Simplified implementation
            signal = amplitude * np.convolve(white, b, mode="same")

        elif signal_type == "brown_noise":
            # Brown(ian) noise - power spectral density is inversely proportional to the square of the frequency
            amplitude = noise_level if noise_level is not None else 1.0
            white = np.random.normal(0, 1, n_samples)
            # Generate brown noise by integrating white noise
            signal = amplitude * np.cumsum(white) / np.sqrt(n_samples)

        elif signal_type == "gaussian_noise":
            # Gaussian noise
            amplitude = noise_level if noise_level is not None else 1.0
            signal = amplitude * np.random.normal(0, 1, n_samples)

        elif signal_type == "uniform_noise":
            # Uniformly distributed noise
            amplitude = noise_level if noise_level is not None else 1.0
            signal = amplitude * np.random.uniform(-1, 1, n_samples)

        # Special signals
        elif signal_type == "impulse":
            # Impulse signal
            signal = np.zeros(n_samples)
            if n_samples > 0:
                signal[0] = 1.0

        elif signal_type == "step":
            # Step signal
            signal = np.ones(n_samples)

        elif signal_type == "exponential":
            # Exponential decay signal
            if frequency is None:
                frequency = 1.0  # Default decay constant
            signal = np.exp(-frequency * t)

        elif signal_type == "dc":
            # DC signal
            amplitude = noise_level if noise_level is not None else 1.0
            signal = amplitude * np.ones(n_samples)

        else:
            # Generate a sine wave by default
            if frequency is None:
                frequency = 1.0
            signal = np.sin(2 * np.pi * frequency * t)

        # Prepare the full signal data
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

        # Save to file, supporting custom filenames
        filepath = self._save_signal_to_file(full_data, "generate", output_filename)

        # Return summary information, without the large data arrays
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
            "message": f"{signal_type} signal data has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
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
        """Advanced filtering operations"""
        signal_array = np.array(signal)
        filtered_signal = signal_array.copy()

        # Normalize the cutoff frequency
        nyquist = sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        if normalized_cutoff >= 1:
            return {"error": "Cutoff frequency must be less than the Nyquist frequency"}

        if filter_type == "lowpass":
            # Simple low-pass filter (Butterworth approximation)
            order = order or 2
            # Using a recursive filter
            alpha = 1 / (1 + 2 * np.pi * cutoff_freq / sampling_rate)
            filtered_signal = np.zeros_like(signal_array)
            filtered_signal[0] = signal_array[0]
            for i in range(1, len(signal_array)):
                filtered_signal[i] = (
                    alpha * filtered_signal[i - 1] + (1 - alpha) * signal_array[i]
                )

        elif filter_type == "highpass":
            # High-pass filter
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
            # Band-pass filter (requires two cutoff frequencies)
            # Simplified version used here
            low_cutoff = cutoff_freq * 0.8
            high_cutoff = cutoff_freq * 1.2
            # Low-pass then high-pass
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
                # Simple median filter implementation
                window_size = order or 5
                filtered_signal = np.array(signal_array)
                for i in range(len(signal_array)):
                    start = max(0, i - window_size // 2)
                    end = min(len(signal_array), i + window_size // 2 + 1)
                    filtered_signal[i] = np.median(signal_array[start:end])

        else:
            return {"error": f"Unsupported filter type: {filter_type}"}

        # Calculate SNR improvement
        snr_improvement = self._calculate_snr_improvement(signal_array, filtered_signal)

        # Prepare full data
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

        # Save to file
        filepath = self._save_signal_to_file(full_data, "filter", output_filename)

        # Return summary information
        return {
            "operation": "filter",
            "filter_type": filter_type,
            "cutoff_freq": cutoff_freq,
            "sampling_rate": sampling_rate,
            "order": order,
            "snr_improvement_db": float(snr_improvement),
            "filter_stats": full_data["filter_stats"],
            "data_file": filepath,
            "message": f"Filter result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _simple_lowpass(
        self, signal: np.ndarray, cutoff_freq: float, sampling_rate: float
    ) -> np.ndarray:
        """Simple low-pass filter"""
        alpha = 1 / (1 + 2 * np.pi * cutoff_freq / sampling_rate)
        filtered = np.zeros_like(signal)
        filtered[0] = signal[0]
        for i in range(1, len(signal)):
            filtered[i] = alpha * filtered[i - 1] + (1 - alpha) * signal[i]
        return filtered

    def _simple_highpass(
        self, signal: np.ndarray, cutoff_freq: float, sampling_rate: float
    ) -> np.ndarray:
        """Simple high-pass filter"""
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
        """Calculate Signal-to-Noise Ratio (SNR) improvement"""
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
        """Apply a window function"""
        signal_array = np.array(signal)

        if window_size > len(signal_array):
            return {"error": "Window size cannot be greater than signal length"}

        # Generate window function
        if window_type == "hanning":
            window = np.hanning(window_size)
        elif window_type == "hamming":
            window = np.hamming(window_size)
        elif window_type == "blackman":
            window = np.blackman(window_size)
        elif window_type == "bartlett":
            window = np.bartlett(window_size)
        elif window_type == "kaiser":
            # Use beta=5 as a default value
            window = np.kaiser(window_size, 5)
        else:
            window = np.hanning(window_size)

        # Apply window function in segments
        hop_length = window_size // 2
        n_frames = (len(signal_array) - window_size) // hop_length + 1

        windowed_frames = []
        for i in range(n_frames):
            start = i * hop_length
            end = start + window_size
            frame = signal_array[start:end]
            windowed_frame = frame * window
            windowed_frames.append(windowed_frame.tolist())

        # Reconstruct signal (simplified version)
        windowed_signal = np.zeros_like(signal_array)
        for i, frame in enumerate(windowed_frames):
            start = i * hop_length
            end = start + window_size
            if end <= len(windowed_signal):
                windowed_signal[start:end] += frame

        # Prepare full data
        full_data = {
            "operation": "windowing",
            "window_type": window_type,
            "window_size": window_size,
            "window_function": window.tolist(),
            "windowed_frames": windowed_frames,
            "windowed_signal": windowed_signal.tolist(),
            "frame_count": n_frames,
        }

        # Save to file
        filepath = self._save_signal_to_file(full_data, "window", output_filename)

        # Return summary information
        return {
            "operation": "windowing",
            "window_type": window_type,
            "window_size": window_size,
            "frame_count": n_frames,
            "data_file": filepath,
            "message": f"Windowing result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
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
        """Autocorrelation Analysis"""
        signal_array = np.array(signal)
        n = len(signal_array)

        # Calculate autocorrelation
        autocorr = np.correlate(signal_array, signal_array, mode="full")
        autocorr = autocorr[n - 1 :]  # Take only the positive lag part

        # Normalize
        autocorr_normalized = autocorr / autocorr[0]

        # Find the first zero-crossing
        zero_crossing = None
        for i in range(1, len(autocorr_normalized)):
            if autocorr_normalized[i - 1] > 0 and autocorr_normalized[i] <= 0:
                zero_crossing = i
                break

        # Calculate autocorrelation features
        max_correlation = float(autocorr[0])
        correlation_length = np.sum(
            np.abs(autocorr_normalized) > 0.1
        )  # Correlation length at 10% threshold

        # Prepare full data
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

        # Save to file
        filepath = self._save_signal_to_file(full_data, "autocorr", output_filename)

        # Return summary information
        return {
            "operation": "autocorrelation",
            "signal_length": n,
            "first_zero_crossing": zero_crossing,
            "max_correlation": max_correlation,
            "correlation_length": int(correlation_length),
            "autocorr_properties": full_data["autocorr_properties"],
            "data_file": filepath,
            "message": f"Autocorrelation analysis result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _crosscorrelation(
        self,
        signal1: List[float],
        signal2: List[float],
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cross-correlation Analysis"""
        if not signal1 or not signal2:
            return {"error": "Both signals cannot be empty"}

        signal1_array = np.array(signal1)
        signal2_array = np.array(signal2)

        # Calculate cross-correlation
        crosscorr = np.correlate(signal1_array, signal2_array, mode="full")

        # Find the maximum correlation value and the corresponding delay
        max_corr_index = np.argmax(np.abs(crosscorr))
        max_correlation = crosscorr[max_corr_index]
        delay = max_corr_index - len(signal2_array) + 1

        # Normalize cross-correlation
        norm_factor = np.sqrt(np.sum(signal1_array**2) * np.sum(signal2_array**2))
        normalized_crosscorr = crosscorr / norm_factor if norm_factor > 0 else crosscorr

        # Prepare full data
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

        # Save to file
        filepath = self._save_signal_to_file(full_data, "crosscorr", output_filename)

        # Return summary information
        return {
            "operation": "crosscorrelation",
            "signal1_length": len(signal1),
            "signal2_length": len(signal2),
            "max_correlation": float(max_correlation),
            "normalized_max_correlation": float(normalized_crosscorr[max_corr_index]),
            "optimal_delay": int(delay),
            "correlation_coefficient": full_data["correlation_coefficient"],
            "data_file": filepath,
            "message": f"Cross-correlation analysis result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _spectral_analysis(
        self,
        signal: List[float],
        sampling_rate: float,
        window_size: Optional[int] = None,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Spectral Analysis"""
        signal_array = np.array(signal)

        if window_size and window_size < len(signal_array):
            # Segmented spectral analysis
            hop_length = window_size // 2
            n_frames = (len(signal_array) - window_size) // hop_length + 1

            spectrograms = []
            for i in range(n_frames):
                start = i * hop_length
                end = start + window_size
                frame = signal_array[start:end]

                # Apply window function
                windowed_frame = frame * np.hanning(len(frame))

                # Calculate FFT
                fft_result = np.fft.fft(windowed_frame)
                magnitude = np.abs(fft_result[: len(fft_result) // 2])
                spectrograms.append(magnitude.tolist())

            frequencies = np.fft.fftfreq(window_size, 1 / sampling_rate)[
                : window_size // 2
            ]

            # Calculate average power spectrum
            avg_power_spectrum = np.mean(spectrograms, axis=0)

        else:
            # Spectral analysis of the entire signal
            fft_result = np.fft.fft(signal_array)
            power_spectrum = np.abs(fft_result) ** 2
            frequencies = np.fft.fftfreq(len(signal_array), 1 / sampling_rate)

            # Take only the positive frequency part
            positive_freq_indices = frequencies >= 0
            positive_frequencies = frequencies[positive_freq_indices]
            positive_power = power_spectrum[positive_freq_indices]

            frequencies = positive_frequencies
            avg_power_spectrum = positive_power
            spectrograms = [positive_power.tolist()]

        # Spectral Centroid
        spectral_centroid = (
            np.sum(frequencies * avg_power_spectrum) / np.sum(avg_power_spectrum)
            if np.sum(avg_power_spectrum) > 0
            else 0
        )

        # Spectral Bandwidth
        spectral_bandwidth = (
            np.sqrt(
                np.sum(((frequencies - spectral_centroid) ** 2) * avg_power_spectrum)
                / np.sum(avg_power_spectrum)
            )
            if np.sum(avg_power_spectrum) > 0
            else 0
        )

        # Spectral Rolloff
        cumulative_power = np.cumsum(avg_power_spectrum)
        total_power = cumulative_power[-1]
        rolloff_threshold = 0.85 * total_power
        rolloff_index = np.where(cumulative_power >= rolloff_threshold)[0]
        spectral_rolloff = (
            frequencies[rolloff_index[0]] if len(rolloff_index) > 0 else frequencies[-1]
        )

        # Prepare full data
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

        # Save to file
        filepath = self._save_signal_to_file(full_data, "spectral", output_filename)

        # Return summary information
        return {
            "operation": "spectral_analysis",
            "sampling_rate": sampling_rate,
            "window_size": window_size,
            "spectral_features": full_data["spectral_features"],
            "data_file": filepath,
            "message": f"Spectral analysis data has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
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
        """Calculate signal metrics"""
        if not signal:
            return {"error": "Input signal cannot be empty"}

        signal_array = np.array(signal)

        # Basic statistics
        mean_value = np.mean(signal_array)
        std_value = np.std(signal_array)
        rms_value = np.sqrt(np.mean(signal_array**2))
        peak_value = np.max(np.abs(signal_array))

        # Peak factor and crest factor
        crest_factor = peak_value / rms_value if rms_value > 0 else 0

        # Skewness and Kurtosis
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

        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal_array)) != 0)
        zero_crossing_rate = zero_crossings / (len(signal_array) - 1)

        # Signal energy and power
        signal_energy = np.sum(signal_array**2)
        signal_power = signal_energy / len(signal_array)

        # Dynamic range
        min_val = np.min(signal_array)
        max_val = np.max(signal_array)
        dynamic_range = max_val - min_val

        # THD (Total Harmonic Distortion) - simplified calculation
        fft_result = np.fft.fft(signal_array)
        magnitude = np.abs(fft_result[: len(fft_result) // 2])
        fundamental_index = np.argmax(magnitude[1:]) + 1  # Skip DC
        fundamental_power = magnitude[fundamental_index] ** 2

        harmonic_power = 0
        for i in range(2, 6):  # Calculate the first 5 harmonics
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
        """Convolution operation"""
        if not signal1 or not signal2:
            return {"error": "Both signals cannot be empty"}

        signal1_array = np.array(signal1)
        signal2_array = np.array(signal2)

        # Calculate convolution
        conv_result = np.convolve(signal1_array, signal2_array, mode="full")
        conv_valid = np.convolve(signal1_array, signal2_array, mode="valid")
        conv_same = np.convolve(signal1_array, signal2_array, mode="same")

        # Prepare full data
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

        # Save to file
        filepath = self._save_signal_to_file(full_data, "convolution", output_filename)

        # Return summary information
        return {
            "operation": "convolution",
            "signal1_length": len(signal1),
            "signal2_length": len(signal2),
            "output_lengths": full_data["output_lengths"],
            "data_file": filepath,
            "message": f"Convolution result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
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
        """Deconvolution operation"""
        if not signal or not kernel:
            return {"error": "Signal and kernel cannot be empty"}

        signal_array = np.array(signal)
        kernel_array = np.array(kernel)

        # Frequency-domain deconvolution
        signal_fft = np.fft.fft(signal_array, len(signal_array) + len(kernel_array) - 1)
        kernel_fft = np.fft.fft(kernel_array, len(signal_array) + len(kernel_array) - 1)

        # Add regularization to prevent division by zero
        regularization = 1e-10
        deconv_fft = signal_fft / (kernel_fft + regularization)
        deconv_result = np.real(np.fft.ifft(deconv_fft))

        # Truncate to the valid part
        deconv_result = deconv_result[: len(signal_array)]

        # Prepare full data
        full_data = {
            "operation": "deconvolution",
            "signal_length": len(signal),
            "kernel_length": len(kernel),
            "original_signal": signal,
            "kernel": kernel,
            "deconvolution_result": deconv_result.tolist(),
            "regularization_used": regularization,
        }

        # Save to file
        filepath = self._save_signal_to_file(
            full_data, "deconvolution", output_filename
        )

        # Return summary information
        return {
            "operation": "deconvolution",
            "signal_length": len(signal),
            "kernel_length": len(kernel),
            "regularization_used": regularization,
            "data_file": filepath,
            "message": f"Deconvolution result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
            "result_stats": {
                "max_value": float(np.max(np.abs(deconv_result))),
                "mean_value": float(np.mean(deconv_result)),
                "std_value": float(np.std(deconv_result)),
            },
        }

    def _envelope_detection(
        self, signal: List[float], output_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Envelope Detection"""
        signal_array = np.array(signal)

        # Calculate the analytic signal (Hilbert transform)
        try:
            from scipy.signal import hilbert

            analytic_signal = hilbert(signal_array)
        except ImportError:
            # Simplified version: use absolute value as the envelope
            analytic_signal = signal_array + 1j * np.zeros_like(signal_array)

        # Extract amplitude envelope and phase
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        instantaneous_frequency = np.diff(np.unwrap(instantaneous_phase)) / (2 * np.pi)

        envelope_stats = {
            "max_amplitude": float(np.max(amplitude_envelope)),
            "min_amplitude": float(np.min(amplitude_envelope)),
            "mean_amplitude": float(np.mean(amplitude_envelope)),
            "amplitude_variation": float(np.std(amplitude_envelope)),
        }

        # Prepare full data
        full_data = {
            "operation": "envelope_detection",
            "original_signal": signal,
            "amplitude_envelope": amplitude_envelope.tolist(),
            "instantaneous_phase": instantaneous_phase.tolist(),
            "instantaneous_frequency": instantaneous_frequency.tolist(),
            "envelope_stats": envelope_stats,
        }

        # Save to file
        filepath = self._save_signal_to_file(full_data, "envelope", output_filename)

        # Return summary information
        return {
            "operation": "envelope_detection",
            "envelope_stats": envelope_stats,
            "data_file": filepath,
            "message": f"Envelope detection result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
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
        """Phase Analysis"""
        if not signal:
            return {"error": "Input signal cannot be empty"}

        signal_array = np.array(signal)

        # Calculate the analytic signal (Hilbert transform)
        try:
            from scipy.signal import hilbert

            analytic_signal = hilbert(signal_array)
        except ImportError:
            # Simplified version: use FFT to calculate Hilbert transform
            fft_result = np.fft.fft(signal_array)
            n = len(signal_array)
            h = np.zeros(n)
            h[0] = 1
            h[1 : n // 2] = 2
            h[n // 2] = 1
            analytic_signal = np.fft.ifft(fft_result * h)

        # Extract instantaneous phase
        instantaneous_phase = np.angle(analytic_signal)
        instantaneous_frequency = (
            np.diff(np.unwrap(instantaneous_phase)) * sampling_rate / (2 * np.pi)
        )

        # Calculate phase statistics
        phase_unwrapped = np.unwrap(instantaneous_phase)
        phase_variance = np.var(instantaneous_phase)
        phase_linearity = np.corrcoef(np.arange(len(phase_unwrapped)), phase_unwrapped)[
            0, 1
        ]

        # FFT phase analysis
        fft_result = np.fft.fft(signal_array)
        fft_phase = np.angle(fft_result)
        frequencies = np.fft.fftfreq(len(signal_array), 1 / sampling_rate)

        # Take only the positive frequency part
        positive_freq_indices = frequencies >= 0
        positive_frequencies = frequencies[positive_freq_indices]
        positive_phase = fft_phase[positive_freq_indices]
        positive_magnitude = np.abs(fft_result[positive_freq_indices])

        # Prepare full data
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

        # Save to file
        filepath = self._save_signal_to_file(full_data, "phase", output_filename)

        # Return summary information
        return {
            "operation": "phase_analysis",
            "sampling_rate": sampling_rate,
            "phase_statistics": full_data["phase_statistics"],
            "data_file": filepath,
            "message": f"Phase analysis result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
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
        """Noise Reduction"""
        if not signal:
            return {"error": "Input signal cannot be empty"}

        signal_array = np.array(signal)
        denoised = signal_array.copy()

        if method == "moving_average":
            # Moving average filter
            kernel = np.ones(window_size) / window_size
            denoised = np.convolve(signal_array, kernel, mode="same")

        elif method == "median":
            # Median filter
            for i in range(len(signal_array)):
                start = max(0, i - window_size // 2)
                end = min(len(signal_array), i + window_size // 2 + 1)
                denoised[i] = np.median(signal_array[start:end])

        elif method == "gaussian":
            # Gaussian filter
            sigma = window_size / 6  # standard deviation
            kernel_size = 2 * window_size + 1
            kernel = np.exp(-np.arange(kernel_size) ** 2 / (2 * sigma**2))
            kernel = kernel / np.sum(kernel)
            denoised = np.convolve(signal_array, kernel, mode="same")

        elif method == "savitzky_golay":
            # Savitzky-Golay filter (simplified version)
            try:
                from scipy.signal import savgol_filter

                denoised = savgol_filter(signal_array, window_size, 3)
            except ImportError:
                # Fallback to moving average
                kernel = np.ones(window_size) / window_size
                denoised = np.convolve(signal_array, kernel, mode="same")

        elif method == "wiener":
            # Wiener filter (simplified version)
            # Estimate noise variance
            noise_var = np.var(signal_array - np.mean(signal_array))
            signal_power = np.var(signal_array)
            wiener_gain = signal_power / (signal_power + noise_var)
            denoised = signal_array * wiener_gain

        else:
            return {"error": f"Unsupported noise reduction method: {method}"}

        # Calculate performance metrics
        mse = np.mean((signal_array - denoised) ** 2)
        snr_original = np.var(signal_array) / np.var(
            signal_array - np.mean(signal_array)
        )
        snr_denoised = np.var(denoised) / np.var(denoised - np.mean(denoised))
        snr_improvement = (
            10 * np.log10(snr_denoised / snr_original) if snr_original > 0 else 0
        )

        # Prepare full data
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

        # Save to file
        filepath = self._save_signal_to_file(full_data, "denoise", output_filename)

        # Return summary information
        return {
            "operation": "noise_reduction",
            "method": method,
            "window_size": window_size,
            "performance_metrics": full_data["performance_metrics"],
            "data_file": filepath,
            "message": f"Noise reduction result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _resampling(
        self,
        signal: List[float],
        original_rate: float,
        target_rate: float,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resampling"""
        if not signal:
            return {"error": "Input signal cannot be empty"}

        if original_rate <= 0 or target_rate <= 0:
            return {"error": "Sampling rates must be positive"}

        signal_array = np.array(signal)
        ratio = target_rate / original_rate

        if ratio == 1.0:
            resampled = signal_array
        elif ratio > 1.0:
            # Upsampling: interpolation
            new_length = int(len(signal_array) * ratio)
            old_indices = np.arange(len(signal_array))
            new_indices = np.linspace(0, len(signal_array) - 1, new_length)
            resampled = np.interp(new_indices, old_indices, signal_array)
        else:
            # Downsampling: decimation
            decimation_factor = int(1 / ratio)
            resampled = signal_array[::decimation_factor]

        # Prepare full data
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

        # Save to file
        filepath = self._save_signal_to_file(full_data, "resample", output_filename)

        # Return summary information
        return {
            "operation": "resampling",
            "original_rate": original_rate,
            "target_rate": target_rate,
            "ratio": float(ratio),
            "length_change": full_data["length_change"],
            "data_file": filepath,
            "message": f"Resampling result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _amplitude_modulation(
        self,
        signal: List[float],
        carrier_freq: float,
        sampling_rate: float,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Amplitude Modulation"""
        if not signal:
            return {"error": "Input signal cannot be empty"}

        signal_array = np.array(signal)
        t = np.arange(len(signal_array)) / sampling_rate

        # Generate carrier signal
        carrier = np.cos(2 * np.pi * carrier_freq * t)

        # AM modulation
        modulated = (1 + 0.5 * signal_array) * carrier

        # Prepare full data
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

        # Save to file
        filepath = self._save_signal_to_file(full_data, "modulation", output_filename)

        # Return summary information
        return {
            "operation": "amplitude_modulation",
            "carrier_freq": carrier_freq,
            "sampling_rate": sampling_rate,
            "modulation_stats": full_data["modulation_stats"],
            "data_file": filepath,
            "message": f"Amplitude modulation result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }

    def _amplitude_demodulation(
        self,
        signal: List[float],
        carrier_freq: float,
        sampling_rate: float,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Amplitude Demodulation"""
        if not signal:
            return {"error": "Input signal cannot be empty"}

        signal_array = np.array(signal)
        t = np.arange(len(signal_array)) / sampling_rate

        # Envelope detection
        envelope = np.abs(signal_array)

        # Low-pass filter to remove carrier
        cutoff_freq = carrier_freq / 10  # Set cutoff to 1/10 of carrier frequency
        filtered_demod = self._simple_lowpass(envelope, cutoff_freq, sampling_rate)

        # Remove DC component
        filtered_demod = filtered_demod - np.mean(filtered_demod)

        # Prepare full data
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

        # Save to file
        filepath = self._save_signal_to_file(full_data, "demodulation", output_filename)

        # Return summary information
        return {
            "operation": "amplitude_demodulation",
            "carrier_freq": carrier_freq,
            "sampling_rate": sampling_rate,
            "cutoff_freq": cutoff_freq,
            "demodulation_stats": full_data["demodulation_stats"],
            "data_file": filepath,
            "message": f"Amplitude demodulation result has been saved to file: {os.path.basename(filepath) if isinstance(filepath, str) else 'unknown'}",
        }
