"""Adapt Comfy IMAGE batches to the independently bundled Velum engine."""
from __future__ import annotations

import json
import math
from numbers import Integral
import threading
import time

import numpy as np
import torch

from .isolated_client import IsolatedVelumClient

_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def _hooks(total):
    import importlib
    management = importlib.import_module("comfy.model_management")
    utilities = importlib.import_module("comfy.utils")
    return management.throw_exception_if_processing_interrupted, utilities.ProgressBar(total)


def _validate_image(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} 必须是 ComfyUI IMAGE 浮点 Tensor。")
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"{name} 必须为 [batch, height, width, 3] RGB；不接受MASK、LATENT或RGBA。")
    if any(n == 0 for n in value.shape):
        raise ValueError(f"{name} 不能为空。")


def _to_rgb8(frame, name):
    # Conversion allocates new storage: never clamp or round an input in-place.
    pixels = frame.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    if not np.all(np.isfinite(pixels)):
        raise ValueError(f"{name} 含NaN或Inf，无法进行色彩迁移。")
    clipped = int(np.count_nonzero((pixels < 0.0) | (pixels > 1.0)))
    rgb = np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb, clipped


class VelumColorTransfer:
    CATEGORY = "image/Velum"
    FUNCTION = "transfer"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    DESCRIPTION = (
        "将参考图的色彩迁移到源图，保持源图尺寸。单张参考图可复用于整批源图；"
        "等长参考批次逐张配对。auto去色带及GPU自动检测为默认。"
        "当前内核按8位RGB计算；输出为0～1的ComfyUI IMAGE。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_image": ("IMAGE", {"tooltip": "需要调色的源图；输出保持其尺寸。"}),
            "reference_image": ("IMAGE", {"tooltip": "色彩参考图；允许不同尺寸，批次必须为1或与源图相同。"}),
            "deband_mode": (["auto", "off", "manual"], {"default": "auto", "tooltip": "off只关闭去色带，仍执行色彩迁移。"}),
            "manual_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "仅manual模式使用；auto由内核选择强度。"}),
            "backend": (["auto", "cpu", "gpu"], {"default": "auto", "tooltip": "auto无可用CUDA或显存不足时回退CPU；gpu为严格GPU模式，不可用时明确报错。"}),
            "cpu_workers": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1, "tooltip": "Numba CPU线程上限；不会修改全局Torch/OpenCV线程设置。"}),
        }, "optional": {
            "structure_field": (["gradient_energy", "dog_guided", "dog_raw"], {
                "default": "gradient_energy", "tooltip": "去色带区域算法：梯度能量（默认）、DoG引导滤波、原始DoG。旧工作流未提供时仍用梯度能量。"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "调色结果与原图的混合比例：1为完整调色，0为原图并跳过计算；与去色带强度无关。"}),
        }}

    def transfer(self, source_image, reference_image, deband_mode="auto", manual_strength=1.0,
                  backend="auto", cpu_workers=8, structure_field="gradient_energy", strength=1.0):
        started = time.perf_counter()
        _validate_image(source_image, "source_image")
        _validate_image(reference_image, "reference_image")
        count, references = source_image.shape[0], reference_image.shape[0]
        if references not in (1, count):
            raise ValueError(f"参考图批次数应为1或{count}，实际为{references}；不会循环、截断或错配。")
        if deband_mode not in {"auto", "off", "manual"}:
            raise ValueError("deband_mode 必须为 auto/off/manual。")
        if backend not in {"auto", "cpu", "gpu"}:
            raise ValueError("backend 必须为 auto/cpu/gpu。")
        if structure_field not in {"gradient_energy", "dog_guided", "dog_raw"}:
            raise ValueError("未知去色带区域算法。")
        if not math.isfinite(float(manual_strength)) or not 0.0 <= float(manual_strength) <= 1.0:
            raise ValueError("manual_strength 必须为0～1的有限数。")
        if isinstance(cpu_workers, bool) or not isinstance(cpu_workers, Integral) or not 1 <= cpu_workers <= 64:
            raise ValueError("cpu_workers 必须为1～64的整数。")
        strength = float(strength)
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("strength 必须为0～1的有限数。")
        interrupt, progress = _hooks(count)
        interrupt()
        if strength == 0.0:
            output = source_image.detach().to(device="cpu", dtype=torch.float32, copy=True)
            if not np.all(np.isfinite(output.numpy())):
                raise ValueError("source_image 含NaN或Inf，无法进行色彩迁移。")
            progress.update_absolute(count)
            return output, json.dumps({"velum_version": "4.10", "node_package_version": "4.10.1",
                "strength": strength, "batch_size": count, "skipped": True, "reason": "strength_zero",
                "node_seconds": time.perf_counter() - started, "images": []}, ensure_ascii=False, indent=2)
        shared_reference = _to_rgb8(reference_image[0], "reference_image[0]") if references == 1 else None
        result_tensor = torch.empty(tuple(source_image.shape), dtype=torch.float32, device="cpu")
        reports = []
        global _CLIENT
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = IsolatedVelumClient()
            client = _CLIENT
        for index in range(count):
            interrupt()
            src, source_clipped = _to_rgb8(source_image[index], f"source_image[{index}]")
            ref_index = 0 if references == 1 else index
            ref, reference_clipped = shared_reference if shared_reference is not None else _to_rgb8(reference_image[index], f"reference_image[{index}]")
            tick = time.perf_counter()
            result, response = client.call(src, ref, deband_mode=deband_mode,
                                           manual_strength=float(manual_strength) if deband_mode == "manual" else 1.0,
                                           backend=backend, cpu_workers=int(cpu_workers), cancel=interrupt, structure_field=structure_field,
                                           cuda_device=torch.cuda.current_device() if backend!='cpu' and torch.cuda.is_initialized() else None)
            rpc_seconds = time.perf_counter() - tick
            interrupt()
            if result.shape != src.shape or result.dtype != np.uint8:
                raise RuntimeError("Velum输出格式异常；拒绝返回尺寸或位深错误的图像。")
            result_tensor[index].copy_(torch.from_numpy(result.astype(np.float32) / 255.0))
            if strength < 1.0:
                original = source_image[index].detach().to(device="cpu", dtype=torch.float32)
                result_tensor[index].mul_(strength).add_(original, alpha=1.0 - strength)
            audit = response.get("stats", {}).get("deband", {})
            reports.append({
                "source_index": index, "reference_index": ref_index,
                "source_size": list(src.shape[:2][::-1]), "reference_size": list(ref.shape[:2][::-1]),
                "algorithm_seconds": response.get("algorithm_seconds", 0.0),
                "stage_seconds": response.get("stats", {}).get("seconds", {}),
                "rpc_seconds": rpc_seconds,
                "worker_pid": response.get("worker_pid"),
                "worker_threading_layer": response.get('threading_layer'),
                "worker_log_dir": response.get('worker_log_dir'),
                "stability_revision": response.get('stability_revision'),
                "isolation": "persistent-child-process",
                "velum_version": response.get("version"),
                "backend": response.get("backend", {}),
                "deband_amount": audit.get("amount", 0.0), "deband_skipped": audit.get("skipped", False),
                "structure_field": audit.get('structure_field',structure_field),
                "input_clipped_channel_values": {"source": source_clipped, "reference": reference_clipped},
            })
            progress.update_absolute(index + 1)
        report = {"velum_version": "4.10", "node_package_version":"4.10.1", "deband_mode": deband_mode, "structure_field":structure_field, "batch_size": count, "strength": strength,
                  "input_conversion": "clamp [0,1], round to RGB uint8; no resize; output float32 RGB / 255",
                  "output_blend": "graded * strength + original_float * (1 - strength)",
                  "node_seconds": time.perf_counter() - started, "images": reports}
        return result_tensor, json.dumps(report, ensure_ascii=False, indent=2)
