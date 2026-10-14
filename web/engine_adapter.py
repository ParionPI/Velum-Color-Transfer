"""Legacy web parameter contract backed exclusively by Velum 4.10."""
from dataclasses import dataclass, field
from pathlib import Path
import math
import os
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isolated_client import IsolatedVelumClient
from web_io import quantise, resize_max_side

VERSION='4.10'
CPU_ONLY = os.environ.get('VELUM_STANDALONE_CPU') == '1'
CLIENT=IsolatedVelumClient(runtime_root=Path(os.environ.get('VELUM_WEB_RUNTIME_ROOT', ROOT/'.runtime'/'web')))


@dataclass
class VelumParams:
    deband: float = 1.0
    deband_auto: bool = True
    deband_structure_field: str = 'gradient_energy'
    compute_backend: str = 'cpu' if CPU_ONLY else 'auto'
    analysis_max_side: int = 0
    dither: bool = False
    debug: dict = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.deband) or not 0 <= self.deband <= 1:
            raise ValueError('deband必须为0～1的有限数')
        if self.compute_backend not in {'auto','cpu','gpu'}:
            raise ValueError('compute_backend必须为auto/cpu/gpu')
        if CPU_ONLY:
            if self.compute_backend == 'gpu':
                raise ValueError('独立CPU整合包不含GPU运行库；GPU加速请将整个插件文件夹放入ComfyUI/custom_nodes。')
            self.compute_backend = 'cpu'
        if self.deband_structure_field not in {'gradient_energy','dog_guided','dog_raw'}:
            raise ValueError('未知纹理强度算法')
        if not isinstance(self.analysis_max_side,int) or self.analysis_max_side < 0:
            raise ValueError('analysis_max_side必须为非负整数')
        if self.dither:
            self.debug['compatibility_notes'] = ['4.10无抖动输出：旧版dither请求不再向结果注入噪声。']
            self.dither = False


def grade(src_rgb01, ref_rgb01, p=None, progress=None, reference_stats=None):
    p = p or VelumParams()
    # This argument is retained only for old callers; no historical CAM16 cache is used.
    if reference_stats is not None:
        p.debug.setdefault('compatibility_notes', []).append('旧reference_stats未用于4.10计算。')
    p.debug['compute_backend'] = {'requested':p.compute_backend, 'effective':'pending'}
    if progress: progress(.03, '准备 sRGB 输入')
    src = np.asarray(src_rgb01, dtype=np.float32)
    ref = np.asarray(ref_rgb01, dtype=np.float32)
    for name, image in (('source',src),('reference',ref)):
        if image.ndim != 3 or image.shape[-1] != 3 or 0 in image.shape:
            raise ValueError(f'{name}必须为非空HxWx3 RGB')
        if not np.all(np.isfinite(image)):
            raise ValueError(f'{name}含NaN或Inf')
    if p.analysis_max_side:
        ref = resize_max_side(ref,p.analysis_max_side)
    source_u8, reference_u8 = quantise(src), quantise(ref)
    mode='auto' if p.deband_auto else ('manual' if p.deband>0 else 'off')
    if progress: progress(.10, 'Velum 4.10 完整计算中（含可选GPU映射）')
    started = time.perf_counter()
    rgb,response = CLIENT.call(source_u8,reference_u8,deband_mode=mode,
        manual_strength=1.0 if p.deband_auto else float(p.deband),backend=p.compute_backend,
        cpu_workers=8,structure_field=p.deband_structure_field)
    elapsed = time.perf_counter()-started
    p.debug['compute_backend'] = dict(response['backend'],engine_version=VERSION,isolation='persistent-child-process',worker_pid=response['worker_pid'])
    p.debug['joint_transport'] = {'engine_version':VERSION,
        'method':'4.10 quantile baseline, followed by optional source-structure mapping',
        'final_J_order_guaranteed':mode=='off'}
    p.debug['continuous_mapping'] = dict(response['stats']['deband'],auto=bool(p.deband_auto))
    p.debug['algorithm_seconds'] = response['algorithm_seconds']
    p.debug['rpc_seconds'] = elapsed
    p.debug['engine_version'] = VERSION
    if progress: progress(.90, '4.10计算完成，准备图层与显示输出')
    return rgb.astype(np.float32)/255.0
