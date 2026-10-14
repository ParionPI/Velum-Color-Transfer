"""4.05 L-calibration fix; accepted fitter, chroma calibration and gate unchanged."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .legacy25.oklab import srgb_to_oklab, oklab_to_srgb


@dataclass(frozen=True)
class DebandOptions:
    enabled: bool = True
    auto: bool = True
    strength: float = 1.0
    structure_field: str = 'gradient_energy'

    def __post_init__(self):
        if not isinstance(self.enabled, (bool, np.bool_)) or not isinstance(self.auto, (bool, np.bool_)):
            raise TypeError('enabled and auto must be booleans.')
        if not np.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError('strength must be finite in [0,1].')
        if self.structure_field not in {'gradient_energy', 'dog_guided', 'dog_raw'}:
            raise ValueError('Unknown legacy structure_field.')

    @property
    def active(self):
        return self.enabled and (self.auto or self.strength > 0)


def apply(source_rgb_float, baseline_rgb_float, options):
    """Per-pixel encoded sRGB float in/out; no quantization inside this module.

    Inherits the legacy difference-anchored Oklab decode and sRGB display clip.
    Does NOT re-run J ranking or reference-support projection afterwards.
    """
    if not options.active:
        return baseline_rgb_float, {'skipped': True, 'amount': 0.0, 'auto': bool(options.auto)}
    started = perf_counter()
    source = np.asarray(source_rgb_float, np.float32)
    baseline = np.asarray(baseline_rgb_float, np.float32)
    if source.shape != baseline.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError('Expected matching HxWx3 source and baseline RGB.')
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(baseline)):
        raise ValueError('Expected finite floating RGB.')
    src_lab = srgb_to_oklab(source)
    baseline_lab = srgb_to_oklab(baseline)
    return _apply_mapping(src_lab, baseline_lab, baseline, options, started)


def _apply_palette(source_rgb, baseline_rgb, indices, shape, options):
    # Oklab conversion is pointwise: transform each unique colour before scatter.
    started = perf_counter()
    source = np.asarray(source_rgb, np.float32)
    baseline = np.asarray(baseline_rgb, np.float32)
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(baseline)):
        raise ValueError('Expected finite floating RGB.')
    src_lab = srgb_to_oklab(source)[indices].reshape(shape)
    baseline_lab = srgb_to_oklab(baseline)[indices].reshape(shape)
    baseline = baseline[indices].reshape(shape)
    return _apply_mapping(src_lab, baseline_lab, baseline, options, started)


def _apply_mapping(src_lab, baseline_lab, baseline, options, started):
    from .legacy25 import continuous_mapping as cm, cuda_kernels
    from . import gpu_mapping
    converted = perf_counter()
    with cuda_kernels.backend('cpu'):
        mapped, old_info = cm.apply_continuous_spectrum(
            src_lab, baseline_lab, amount=None if options.auto else options.strength,
            auto=options.auto, structure_field=options.structure_field)
    mapped_at = perf_counter()
    from .gpu_decode import try_anchored_decode
    decoded = try_anchored_decode(mapped, baseline_lab, baseline)
    if decoded is None:
        decoded = baseline + (oklab_to_srgb(mapped) - oklab_to_srgb(baseline_lab))
    if not np.all(np.isfinite(decoded)):
        raise FloatingPointError('Legacy mapping produced nonfinite RGB.')
    outside = np.any((decoded < 0) | (decoded > 1), axis=-1)
    output = np.clip(decoded, 0, 1).astype(np.float32)
    report = dict(old_info)
    report.update(
        skipped=False, implementation='4.04_mapping_4.05_mean_L_residual',
        backend=gpu_mapping.current_info()['effective'], compute_backend=gpu_mapping.current_info(),
        lightness_calibration='least_squares_constant_residual_shift',
        chroma_calibration='unchanged_legacy_channel_percentiles',
        input_is_unquantized_baseline=True, source_coordinates='Oklab plus xy',
        source_RGB_single_valued=False, J_restored_after_mapping=False,
        reference_support_reapplied=False, final_J_order_guaranteed=False,
        display_clip='inherited legacy 2.5 anchored decode then sRGB [0,1] clip',
        clipped_pixel_fraction=float(outside.mean()),
        clip_mean_abs_RGB=float(np.abs(output - decoded).mean()),
        min_before_display_clip=float(decoded.min()), max_before_display_clip=float(decoded.max()),
        seconds_colour_conversion=converted - started, seconds_legacy_mapping=mapped_at - converted,
        seconds_decode_and_clip=perf_counter() - mapped_at)
    return output, report
