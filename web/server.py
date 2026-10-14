"""Velum 4.10 web test bench backend.

Serves:
  GET  /                    the test bench UI
  GET  /api/status          running state + available params + defaults
  POST /api/grade           run the pipeline, return result PNG
  POST /api/compare         return a side-by-side contact sheet
  POST /api/set-images      upload source + reference (multipart or raw)
  POST /api/reset           clear cached images/results

Every response carries the audit numbers as JSON headers (X-Audit-*), so the
frontend can display them without a second round-trip.

Run:  python server.py [--port 8765]
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import io
import ipaddress
import json
import os
import sys
import secrets
import threading
import time
import urllib.request
import weakref
from email import policy
from email.parser import BytesParser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

# This server is a separate process. Configure its native runtime before imports.
os.environ.update(NUMBA_THREADING_LAYER='workqueue',NUMBA_NUM_THREADS='8',
                  OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np

WEB_ROOT = Path(__file__).resolve().parents[1]
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine_adapter import VelumParams, grade, CPU_ONLY       # noqa: E402
from baselines import hm_mkl_hm, mkl_transfer, reinhard_lab  # noqa: E402
from web_io import decode_rgb, load_rgb, quantise, resize_max_side, SRGB_ICC  # noqa: E402
import lut
import lut_save
# Load the NumPy-only display/audit helper without importing the native engine package.
import importlib.util
_oklab_spec = importlib.util.spec_from_file_location('velum_web_oklab', WEB_ROOT/'_velum410'/'legacy25'/'oklab.py')
if _oklab_spec is None or _oklab_spec.loader is None: raise RuntimeError('Missing bundled Oklab helper')
_oklab = importlib.util.module_from_spec(_oklab_spec)
_oklab_spec.loader.exec_module(_oklab)
chroma, hue, oklab_to_srgb, srgb_to_oklab = _oklab.chroma, _oklab.hue, _oklab.oklab_to_srgb, _oklab.srgb_to_oklab

STATE: dict[str, "Any"] = {
    "src": None,      # float32 (H,W,3) in [0,1]
    "ref": None,
    "src_meta": None,
    "ref_meta": None,
    "src_digest": None,
    "ref_digest": None,
}

# ThreadingHTTPServer is necessary for SSE/static responsiveness, but OpenCV,
# ONNXRuntime and some NumPy kernels are not safe/reliable when several heavy
# grade requests enter simultaneously. On a shared LAN, multiple colleagues can
# otherwise trigger native computations concurrently and crash python.exe with
# 0xC0000005 (-1073741819). Serialize all state-changing/native POST work and
# grade streams while keeping ordinary GET/static requests concurrent.
COMPUTE_LOCK = threading.RLock()
_SCALED_PAIR_CACHE_LIMIT = 4
_VELUM_OUTPUT_CACHE_LIMIT = 3
_BODY_LIMIT = 256 * 1024 * 1024
_LUT_TOKEN = secrets.token_urlsafe(32)
_LUT_LOCK = threading.Lock()

try:
    import cv2 as _cv2_runtime
    _cv2_runtime.setNumThreads(1)
except Exception:
    pass


# --------------------------------------------------------------------------
# audit (same numbers as run.py)
# --------------------------------------------------------------------------
def audit(img: np.ndarray, ref: np.ndarray) -> dict:
    lab = srgb_to_oklab(img)
    rlab = srgb_to_oklab(ref)
    L = lab[..., 0]
    C = chroma(lab).ravel()
    h = (np.degrees(hue(lab)).ravel() % 360.0)
    rC = chroma(rlab).ravel()
    rh = (np.degrees(hue(rlab)).ravel() % 360.0)

    bm = (h >= 200) & (h <= 280)
    rbm = (rh >= 200) & (rh <= 280)
    blue_max = float(C[bm].max()) if bm.any() else 0.0
    ref_blue_max = float(rC[rbm].max()) if rbm.any() else 0.0
    cobalt = float(((h >= 200) & (h <= 280) & (C > 0.10)).mean() * 100.0)

    p1, p5, thr, p95 = np.percentile(L, [1, 5, 10, 95])
    span = p95 - p5
    rp5, rp95 = np.percentile(rlab[..., 0], [5, 95])

    dark = L[L <= thr]
    lv = int(len(np.unique(np.rint(dark * 255))))
    avail = max(int(round(float(dark.max() - dark.min()) * 255)), 1)

    import cv2
    Lf = L.astype(np.float32)
    hp = Lf - cv2.GaussianBlur(Lf, (0, 0), 1.6)
    mask = L <= thr
    return {
        "blue_maxC": round(blue_max, 4),
        "blue_over": round(blue_max / max(ref_blue_max, 1e-6), 3),
        "cobalt_pct": round(cobalt, 2),
        "span": round(span, 4),
        "span_over": round(span / max(rp95 - rp5, 1e-6), 3),
        "L_std": round(float(L.std()), 4),
        "shadow_eff": round(lv / max(avail, 1), 3),
        "shadow_hf": round(float(hp[mask].std()) if mask.any() else 0.0, 4),
        "L_p01": round(float(p1), 4),
    }


def dominant_palette(img: np.ndarray, top_n: int = 10) -> list[dict]:
    """Return the dominant perceptual colour clusters of a VELUM result.

    Clustering is performed in Oklab, not RGB, so visually similar colours group
    together more naturally.  The image is deterministically stride-sampled to
    at most ~60k pixels, then a small weighted k-means (k <= 10) is run.  Returned
    shares are fractions of the full sampled image and remain unnormalised; the
    front end normalises the selected Top-N shares so their bar segments fill the
    result width exactly.
    """
    n = max(1, min(int(top_n), 10))
    pixels = np.asarray(img, np.float32).reshape(-1, 3)
    if pixels.size == 0:
        return []
    # 12k samples are ample for a global Top-10 palette and keep the cached
    # layer sliders responsive; this runs on every post-grade modulation.
    step = max(1, int(np.ceil(pixels.shape[0] / 12000)))
    # The same sample lattice; only convert colours actually used by clustering.
    x = srgb_to_oklab(pixels[::step]).astype(np.float32)
    k = min(n, x.shape[0])

    # Deterministic farthest-point initialisation: first centre is the global
    # mean's nearest sample, then each new centre is farthest from the existing
    # set. Stable across refreshes, unlike a random seed.
    mean = x.mean(0)
    first = int(np.argmin(((x - mean) ** 2).sum(1)))
    centres = [x[first].copy()]
    nearest = ((x - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        idx = int(np.argmax(nearest))
        centres.append(x[idx].copy())
        nearest = np.minimum(nearest, ((x - centres[-1]) ** 2).sum(1))
    c = np.asarray(centres, np.float32)

    labels = np.zeros(x.shape[0], np.int32)
    for _ in range(12):
        d2 = ((x[:, None, :] - c[None, :, :]) ** 2).sum(2)
        new_labels = d2.argmin(1).astype(np.int32)
        new_c = c.copy()
        for j in range(k):
            m = new_labels == j
            if m.any():
                new_c[j] = x[m].mean(0)
        if np.array_equal(new_labels, labels) and np.max(np.abs(new_c - c)) < 1e-5:
            labels = new_labels
            c = new_c
            break
        labels = new_labels
        c = new_c

    counts = np.bincount(labels, minlength=k).astype(np.float64)
    shares = counts / max(float(counts.sum()), 1.0)
    order = np.argsort(-shares)
    rgb = np.clip(oklab_to_srgb(c[order]), 0.0, 1.0)
    out = []
    for col, share in zip(rgb, shares[order]):
        u8 = np.rint(col * 255.0).astype(np.uint8)
        out.append({
            "hex": "#%02x%02x%02x" % (int(u8[0]), int(u8[1]), int(u8[2])),
            "share": round(float(share), 6),
        })
    return out


# --------------------------------------------------------------------------
# tiny multipart parser (no external deps)
# --------------------------------------------------------------------------
def parse_multipart(body: bytes, ctype: str) -> dict[str, bytes]:
    """Parse multipart without trimming binary payload endings."""
    if not ctype or "multipart/form-data" not in ctype.lower():
        raise ValueError("expected multipart/form-data")
    header = (f"Content-Type: {ctype}\r\nMIME-Version: 1.0\r\n\r\n"
              ).encode("latin-1") + body
    message = BytesParser(policy=policy.default).parsebytes(header)
    if not message.is_multipart():
        raise ValueError("invalid multipart body")
    parts: dict[str, bytes] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            payload = part.get_payload(decode=True)
            if payload is None: payload = b''
            if not isinstance(payload,bytes): raise ValueError('invalid multipart payload')
            parts[str(name)] = payload
    return parts


def _encode_png(rgb01: np.ndarray, dither: bool = False) -> bytes:
    q = quantise(rgb01, dither=dither)
    from PIL import Image
    buf = io.BytesIO()
    # Level 1 is still lossless, but avoids the expensive default deflate search.
    Image.fromarray(q).save(buf, format="PNG", compress_level=1, icc_profile=SRGB_ICC)
    return buf.getvalue()


def _encode_preview(rgb01: np.ndarray, quality: int = 88,
                    max_side: int = 1400) -> tuple[bytes, str]:
    """Fast JPEG encode for INTERACTIVE frames (slider drags).

    A lossless PNG of a 2K frame costs ~1.3 s to deflate, which alone makes a
    real-time drag impossible; measured at 1024px it is 328 ms of the 725 ms
    per-frame budget. While the user is dragging, the frame is a PREVIEW that
    will be replaced by the next one within milliseconds, so paying for lossless
    compression buys nothing the eye can use. cv2's JPEG encoder is roughly an
    order of magnitude cheaper and q92 is visually transparent for this purpose.

    It is also DOWNSCALED to `max_side`. A drag frame is displayed inside a result
    pane that is well under 1400 px on screen, so encoding and shipping a full 2K/4K
    buffer spends time on pixels the browser immediately throws away in its own
    downscale; at 2K this was ~100 ms of the frame, the largest remaining cost once
    the blends moved to the GPU. The committed render is always full-resolution.

    The final/committed render still goes through _encode_png, so nothing the
    user saves or inspects is ever lossy or reduced -- only in-flight drag frames.
    Returns (bytes, content_type).
    """
    import cv2  # noqa: PLC0415
    img = rgb01
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        sc = max_side / float(max(h, w))
        img = cv2.resize(img, (max(1, int(round(w * sc))), max(1, int(round(h * sc)))),
                         interpolation=cv2.INTER_AREA)
    q = quantise(img, dither=False)            # dithering is invisible in a preview
    ok, enc = cv2.imencode(".jpg", q[..., ::-1],
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return _encode_png(rgb01, dither=False), "image/png"
    return enc.tobytes(), "image/jpeg"


# ---------------------------------------------------------------------------
# Photoshop-style post-grade layer modulation
# ---------------------------------------------------------------------------
def _lum(rgb: np.ndarray) -> np.ndarray:
    """Photoshop/W3C non-separable blend luminance on gamma-encoded RGB."""
    return 0.30 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]


def _sat(rgb: np.ndarray) -> np.ndarray:
    return rgb.max(axis=-1) - rgb.min(axis=-1)


def _clip_color(rgb: np.ndarray) -> np.ndarray:
    """W3C ClipColor used by Color/Luminosity blend modes.

    Mathematically identical to the straightforward np.where formulation, but it
    only touches the pixels that are actually out of gamut. That matters a lot:
    this function is the single most expensive step of the interactive layer path
    (measured 125 ms at 1K and 481 ms at 2K), because the np.where version paid
    for several full-size temporaries per branch even when a handful of pixels
    needed correction. In a real grade the vast majority of pixels are in gamut,
    so gathering the offending ones with a boolean index and scattering the fixed
    values back is dramatically cheaper while producing the same result.
    """
    c = np.asarray(rgb, np.float32).copy()
    l = _lum(c)

    n = c.min(axis=-1)
    below = n < 0.0
    if below.any():
        lb = l[below]
        cb = c[below]                                  # (k,3) gather
        den = np.maximum(lb - n[below], 1e-8)
        c[below] = lb[:, None] + (cb - lb[:, None]) * (lb / den)[:, None]

    x = c.max(axis=-1)
    above = x > 1.0
    if above.any():
        la = l[above]
        ca = c[above]
        den = np.maximum(x[above] - la, 1e-8)
        c[above] = la[:, None] + (ca - la[:, None]) * ((1.0 - la) / den)[:, None]

    return np.clip(c, 0.0, 1.0)


def _set_lum(rgb: np.ndarray, lum: np.ndarray) -> np.ndarray:
    return _clip_color(np.asarray(rgb, np.float32) + (lum - _lum(rgb))[..., None])


def _set_sat(rgb: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """W3C SetSat: preserve hue ordering while replacing saturation."""
    src = np.asarray(rgb, np.float32)
    order = np.argsort(src, axis=-1)
    sorted_c = np.take_along_axis(src, order, axis=-1)
    lo, mid, hi = sorted_c[..., 0], sorted_c[..., 1], sorted_c[..., 2]
    den = hi - lo
    nz = den > 1e-8
    new_mid = np.where(nz, (mid - lo) * sat / np.maximum(den, 1e-8), 0.0)
    new_hi = np.where(nz, sat, 0.0)
    vals = np.stack([np.zeros_like(new_mid), new_mid, new_hi], axis=-1)
    inv = np.argsort(order, axis=-1)
    return np.take_along_axis(vals, inv, axis=-1).astype(np.float32)


# Per-image memo for the parts of the layer blends that depend ONLY on the blend
# source, not on any slider. During a 亮度/色彩匹配 drag these were recomputed
# identically on every frame -- including an argsort over every pixel inside
# _set_sat -- which dominated the composite cost (measured 610 ms of a 717 ms 2K
# frame). Keyed by the array's identity + shape, cleared when the image changes.
_BLEND_MEMO: dict = {}


def _memo_key(arr: np.ndarray) -> tuple:
    return (id(arr), arr.shape, int(STATE.get("epoch", 0)))


def _set_sat_self(blend: np.ndarray) -> np.ndarray:
    """_set_sat(blend, _sat(blend)) -- a pure function of `blend`, so memoised."""
    k = ("set_sat_self",) + _memo_key(blend)
    hit = _BLEND_MEMO.get(k)
    if hit is not None and hit[0]() is blend:
        return hit[1]
    val = _set_sat(blend, _sat(blend))
    if len(_BLEND_MEMO) > 8:
        _BLEND_MEMO.clear()
    _BLEND_MEMO[k] = (weakref.ref(blend), val)
    return val


def _lum_memo(arr: np.ndarray) -> np.ndarray:
    """_lum(arr) memoised -- also slider-independent for the source layer."""
    k = ("lum",) + _memo_key(arr)
    hit = _BLEND_MEMO.get(k)
    if hit is not None and hit[0]() is arr:
        return hit[1]
    val = _lum(arr)
    if len(_BLEND_MEMO) > 8:
        _BLEND_MEMO.clear()
    _BLEND_MEMO[k] = (weakref.ref(arr), val)
    return val


def _blend_color(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Photoshop Color: blend hue+saturation, preserve base luminosity."""
    return _set_lum(_set_sat_self(blend), _lum(base))


def _blend_luminosity(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Photoshop Luminosity: blend luminosity, preserve base colour."""
    return _set_lum(base, _lum_memo(blend))


def _blend_overlay(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    base = np.clip(np.asarray(base, np.float32), 0.0, 1.0)
    blend = np.clip(np.asarray(blend, np.float32), 0.0, 1.0)
    return np.where(base <= 0.5, 2.0 * base * blend,
                    1.0 - 2.0 * (1.0 - base) * (1.0 - blend)).astype(np.float32)


def _blend_screen(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    base = np.clip(np.asarray(base, np.float32), 0.0, 1.0)
    blend = np.clip(np.asarray(blend, np.float32), 0.0, 1.0)
    return (1.0 - (1.0 - base) * (1.0 - blend)).astype(np.float32)


def _blend_multiply(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(base, np.float32), 0.0, 1.0) *
            np.clip(np.asarray(blend, np.float32), 0.0, 1.0)).astype(np.float32)


def _layer_settings(raw: dict | None) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    channel = str(raw.get("overlay_channel", "source")).lower()
    if channel not in ("source", "reinhard", "hm"):
        channel = "source"
    mode = str(raw.get("overlay_mode", "overlay")).lower()
    if mode not in ("overlay", "screen", "multiply", "color"):
        mode = "overlay"
    return {
        "luma_match": float(np.clip(float(raw.get("luma_match", 1.0)), 0.0, 1.0)),
        "color_match": float(np.clip(float(raw.get("color_match", 1.0)), 0.0, 1.0)),
        "overlay_amount": float(np.clip(float(raw.get("overlay_amount", 0.0)), 0.0, 1.0)),
        "overlay_channel": channel,
        "overlay_mode": mode,
    }


def _parse_layer_json(text: str | bytes | None) -> dict:
    try:
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        return _layer_settings(json.loads(text or "{}"))
    except Exception:
        return _layer_settings({})


def _compose_layers(src: np.ndarray, ref: np.ndarray, velum: np.ndarray,
                    settings: dict) -> np.ndarray:
    """Apply the requested layers bottom -> top, without re-running VELUM.

    4 normal VELUM (100%)
    3 selected channel, Overlay, opacity=overlay_amount
    2 VELUM, Luminosity, 100% (pull luminosity back after Overlay)
    1 source, Luminosity, opacity=1-luma_match
    0 source, Color, opacity=1-color_match

    Defaults (1, 1, 0, source) are exactly identity: result == raw VELUM.
    亮度匹配=0 and 色彩匹配=0 together are exactly the source image.

    These are DISPLAY layers, not grade stages: they only ever pull the finished
    result back toward the source, so they are the user's own aesthetic override
    and never part of the reference match.
    """
    s = _layer_settings(settings)
    out = np.asarray(velum, np.float32).copy()
    overlay_a = s["overlay_amount"]
    # Fixed layer 3 below (_blend_luminosity(out, velum)) exists only to restore
    # VELUM's luminosity AFTER the Overlay layer perturbed it. With no overlay,
    # out IS velum, so that call is a mathematical no-op -- but it still costs a
    # full _clip_color pass over every pixel (a sort-free but branchy float
    # traversal), which at 2K measured a large slice of the 876 ms compose time.
    # Skipping it when the overlay did not run is exactly equivalent and is the
    # common case during a 亮度/色彩匹配 drag.
    did_overlay = False
    if overlay_a > 0.0:
        channel = s["overlay_channel"]
        if channel == "reinhard":
            layer = reinhard_lab(src, ref)
        elif channel == "hm":
            layer = hm_mkl_hm(src, ref)
        else:
            layer = src
        mode = s["overlay_mode"]
        if mode == "screen":
            over = _blend_screen(out, layer)
        elif mode == "multiply":
            over = _blend_multiply(out, layer)
        elif mode == "color":
            over = _blend_color(out, layer)
        else:
            over = _blend_overlay(out, layer)
        out = out * (1.0 - overlay_a) + over * overlay_a
        did_overlay = True

    # Fixed layer 3: restore the raw VELUM luminosity after Overlay. Only needed
    # if the Overlay actually ran -- otherwise out is still velum and this is an
    # identity operation (see the note above).
    if did_overlay:
        out = _blend_luminosity(out, velum)

    # Luminosity before Color: Color keeps the luminosity it is applied to, so
    # source-Color over source-Luminosity is an exact round trip and 0%/0%
    # returns the source. The reverse order puts source chroma on VELUM's
    # luminosity first, where ClipColor desaturates out-of-gamut colours that
    # the later Luminosity layer can no longer restore (visibly greyer).
    luma_a = 1.0 - s["luma_match"]
    if luma_a > 0.0:
        lum = _blend_luminosity(out, src)
        out = out * (1.0 - luma_a) + lum * luma_a

    color_a = 1.0 - s["color_match"]
    if color_a > 0.0:
        col = _blend_color(out, src)
        out = out * (1.0 - color_a) + col * color_a

    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)


def _identity_layers(settings):
    return settings['overlay_amount'] == 0 and settings['luma_match'] == 1 and settings['color_match'] == 1


def _display_palette(image, settings, cached_audit):
    if _identity_layers(settings) and 'palette' in cached_audit:
        return cached_audit['palette']
    return dominant_palette(image, top_n=10)


def _sheet_theme(palette: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame RGB, text RGB) using the same rules as the web theme.

    Frame starts at share Top-2, walks to Top-3/... if it collides with the
    maximum-lightness-distance text colour, then keeps H/L, uses 75% saturation,
    and is composited at 75% opacity over the adaptive page background.
    """
    if not palette:
        return (np.array([239, 226, 211], np.float32) / 255.0,
                np.array([55, 52, 47], np.float32) / 255.0)

    def rgb01(item):
        h = item["hex"].lstrip("#")
        return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], np.float32) / 255.0

    rows = []
    for i, item in enumerate(palette[:10]):
        rgb = rgb01(item)
        h, l, s = colorsys.rgb_to_hls(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        rows.append({"i": i, "rgb": rgb, "h": h, "s": s, "l": l})
    top1 = rows[0]
    text = max(rows, key=lambda x: abs(x["l"] - top1["l"]))
    frame_src = rows[1] if len(rows) > 1 else top1
    if frame_src["i"] == text["i"]:
        frame_src = next((x for x in rows[2:] if x["i"] != text["i"]), top1)

    # Web background: Top-1 H, S*75%, L shifted five points toward HSL midpoint.
    bg_l = top1["l"] - 0.05 if top1["l"] >= 0.5 else top1["l"] + 0.05
    bg = np.array(colorsys.hls_to_rgb(top1["h"], np.clip(bg_l, 0, 1),
                                     top1["s"] * 0.75), np.float32)
    frame_raw = np.array(colorsys.hls_to_rgb(frame_src["h"], frame_src["l"],
                                            frame_src["s"] * 0.75), np.float32)
    frame = frame_raw * 0.75 + bg * 0.25
    return np.clip(frame, 0, 1), text["rgb"]


def _compose_sheet(src_g, ref_g, params, out, palette=None):
    """Build the two-row six-panel comparison sheet.

    row 1  source    | reinhard | mkl
    row 2  reference | hm-mkl-hm | VELUM

    The SOURCE defines one shared image box. Source and all four transfer results
    fill it (they have the source's aspect ratio); reference is aspect-fit and
    centred inside it, never stretched, cropped or allowed to exceed it. This is
    equivalent to matching compatible long edges and, for incompatible
    orientations, matching the reference's limiting edge to the source's short
    edge. All six tiles therefore have exactly one width and form a strict 2x3
    grid. Label font, size, bar height and row gap remain unchanged. Shared by
    /api/compare and the SSE grade stream.
    """
    import cv2
    TILE_H = 340
    LABEL_H = 28
    GAP = 4
    src_h, src_w = src_g.shape[:2]
    TILE_W = max(1, int(round(src_w * TILE_H / max(src_h, 1))))
    rh = reinhard_lab(src_g, ref_g)
    mkl = mkl_transfer(src_g, ref_g)
    hm = hm_mkl_hm(src_g, ref_g)
    frame_rgb, text_rgb = _sheet_theme(palette or dominant_palette(out, top_n=10))
    # Arrays in this bench stay RGB all the way to PIL; cv2.putText simply writes
    # the supplied channel tuple, so provide RGB (do not reverse to BGR).
    text_u8 = tuple(int(x) for x in np.rint(text_rgb * 255.0))

    def make_tile(img, lbl, *, contain=False):
        h, w = img.shape[:2]
        if contain:
            scale = min(TILE_W / max(w, 1), TILE_H / max(h, 1))
            fit_w = max(1, int(round(w * scale)))
            fit_h = max(1, int(round(h * scale)))
            fitted = cv2.resize(img, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
            r = np.empty((TILE_H, TILE_W, 3), np.float32)
            r[...] = frame_rgb
            x = (TILE_W - fit_w) // 2
            y = (TILE_H - fit_h) // 2
            r[y:y + fit_h, x:x + fit_w] = fitted
        else:
            r = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        bar = np.empty((LABEL_H, TILE_W, 3), np.uint8)
        bar[...] = np.rint(frame_rgb * 255.0).astype(np.uint8)
        cv2.putText(bar, lbl, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, text_u8, 1, cv2.LINE_AA)
        return np.vstack([bar.astype(np.float32) / 255.0, r])

    row1_tiles = [make_tile(src_g, "source"), make_tile(rh, "reinhard"),
                  make_tile(mkl, "mkl")]
    row2_tiles = [make_tile(ref_g, "reference", contain=True), make_tile(hm, "hm-mkl-hm"),
                  make_tile(out, "VELUM")]
    row1 = np.hstack(row1_tiles)
    row2 = np.hstack(row2_tiles)
    target_w = TILE_W * 3
    gap = np.empty((GAP, target_w, 3), np.float32)
    gap[...] = frame_rgb
    return np.vstack([row1, gap, row2])


def _lru_get(name: str, key):
    cache = STATE.get(name)
    if not isinstance(cache, OrderedDict):
        return None
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _lru_put(name: str, key, value, limit: int):
    cache = STATE.get(name)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        STATE[name] = cache
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def _canonical_grade_key(params: VelumParams) -> str:
    """Canonical parameters that actually affect grade() pixels.

    Excludes debug output and PNG-only dithering. Manual deband amount is irrelevant
    when auto mode owns it, so equivalent requests share one output cache entry.
    """
    import dataclasses
    values = {}
    for field in dataclasses.fields(params):
        if field.name in ("debug", "dither"):
            continue
        value = getattr(params, field.name)
        if field.name == "deband" and params.deband_auto:
            value = None
        values[field.name] = value
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _cached_velum(side, params, params_json, on_progress=None) -> tuple[np.ndarray, dict[str, Any], bool]:
    """Return (velum_rgb, audit, cache_hit) for the current images at `side`.

    ONE cache level, because grade() is now a single indivisible pass. The old
    three-level split (body / stage-9 / final) existed so that flipping the two
    tail-stage toggles could reuse the expensive body; both of those stages were
    measured to cost reference fidelity and were removed, so there is nothing left
    to toggle and nothing to split. The key is the complete parameter payload plus
    the epoch and resolution, so a new image, a new size or any parameter change
    invalidates it, and a repeated identical request is free.
    """
    a: dict[str, Any]
    epoch = STATE.get("epoch", 0)
    side = int(side)
    key = (epoch, side, _canonical_grade_key(params))
    src_g, ref_g = _scaled_pair(side)

    cache = _lru_get("velum_cache_lru", key)
    if cache is not None:
        a = dict(cache['audit'], cache_hit=True, algorithm_seconds=0.0,
                 cached_algorithm_seconds=cache['audit'].get('algorithm_seconds',0.0),
                 rpc_seconds=0.0, cached_rpc_seconds=cache['audit'].get('rpc_seconds',0.0))
        return cache["out"], a, True

    # Only 4.10 transfer owns reference analysis; no 2.5 reference cache is used.
    out = grade(src_g, ref_g, params, progress=on_progress,
                reference_stats=None)

    a = audit(out, ref_g)
    a['engine_version'] = '4.10'
    a['algorithm_seconds'] = params.debug['algorithm_seconds']
    a['rpc_seconds'] = params.debug.get('rpc_seconds',a['algorithm_seconds'])
    a['cache_hit'] = False
    if params.debug.get('compatibility_notes'):
        a['compatibility_notes'] = params.debug['compatibility_notes']
    a["palette"] = dominant_palette(out, top_n=10)
    # The normal web path is /api/grade_stream, which returns this cached audit
    # directly. Keep stage metadata here (rather than only in /api/grade), so the
    # front end can reflect auto-solved controls such as Continuous Mapping's
    # actual amount after every render and on subsequent cache hits.
    for stage in ("compute_backend", "joint_transport", "continuous_mapping"):
        value = params.debug.get(stage)
        if isinstance(value, dict):
            a[stage] = value

    _lru_put("velum_cache_lru", key,
             {"out": out, "audit": dict(a)}, _VELUM_OUTPUT_CACHE_LIMIT)
    return out, a, False


def _params_from_form(fields: dict[str, bytes]) -> VelumParams:
    """Decode the JSON 'params' field into a VelumParams, ignoring unknown keys.

    Filtering against the dataclass's OWN field list is the point. Previously the
    payload was splatted into the constructor unfiltered, so a single stale key
    from a cached page raised TypeError inside the request handler and the whole
    connection died with a 500 and no result image -- which is exactly what
    happens whenever a parameter is renamed or removed: the browser keeps posting
    the old name from its cached JS. A test bench must not be brittle against its
    own front end being one version behind, so unknown keys are reported and
    skipped -- which is also what lets the large parameter cull land without
    breaking any still-cached page.

    Values are also coerced to the declared type, because HTML form fields arrive
    as numbers-or-strings and a bool slider posts 0/1, which would otherwise make
    `dither` a float and silently disable the branch that tests it.
    """
    import dataclasses

    base = VelumParams()
    known = {f.name: f for f in dataclasses.fields(base) if f.name != "debug"}
    kw: dict = {}

    if "params" in fields:
        raw = {}
        try:
            parsed = json.loads(fields["params"].decode("utf-8"))
            if isinstance(parsed, dict):
                raw = parsed
        except Exception as e:
            print(f"[params] undecodable payload ignored: {e}")

        unknown = []
        for k, v in raw.items():
            if k not in known:
                unknown.append(k)
                continue
            cur = getattr(base, k)
            try:
                if isinstance(cur, bool):
                    kw[k] = bool(v) if not isinstance(v, str) else \
                        v.strip().lower() in ("1", "true", "yes", "on")
                elif isinstance(cur, int):
                    kw[k] = int(round(float(v)))
                elif isinstance(cur, float):
                    kw[k] = float(v)
                else:
                    kw[k] = v
            except (TypeError, ValueError):
                print(f"[params] {k}={v!r} not coercible to "
                      f"{type(cur).__name__}; using default")
        if unknown:
            print(f"[params] ignoring unknown keys (stale front end?): "
                  f"{', '.join(sorted(unknown))}")

    return VelumParams(**kw)


def _output_side_from_form(fields: dict[str, bytes]) -> int:
    """Read the 'max_output_side' form field (long-edge px cap), 0 = no cap.

    This is the resolution ladder that makes parameter iteration fast: capping the
    working images BEFORE grade() cuts the per-pixel pipeline cost, so 360p feels
    instant while 2160p is the final-quality render. It only ever SHRINKS -- a cap
    at or above the source is a no-op (resize_max_side returns the array
    unchanged) -- so picking a ladder rung larger than the upload costs nothing and
    never upscales. Unknown/blank/<=0 means "no cap" (grade at native size).
    """
    raw = fields.get("max_output_side")
    if raw is None:
        return 0
    try:
        return max(int(round(float(raw.decode("utf-8").strip()))), 0)
    except (ValueError, AttributeError):
        return 0


def _scaled_pair(side: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (src, ref) from STATE, capped to `side` on the long edge (0 = native).

    MEMOISED by (epoch, side). Two reasons, both about the interactive path:
      * the resize itself was repeated on every frame of a slider drag;
      * more importantly the result must be the SAME array object each time, or
        the slider-independent blend memo (_BLEND_MEMO, keyed on array identity)
        can never hit -- a fresh copy per frame silently defeats it.
    A new image bumps the epoch, which invalidates this along with everything else.
    """
    key = (int(STATE.get("epoch", 0)), int(side))
    hit = _lru_get("scaled_pair_lru", key)
    if hit is not None:
        return hit
    src = STATE["src"]
    ref = STATE["ref"]
    if side and side > 0:
        src = resize_max_side(src, side)
        ref = resize_max_side(ref, side)
    _lru_put("scaled_pair_lru", key, (src, ref), _SCALED_PAIR_CACHE_LIMIT)
    return src, ref


def _respond(handler, code: int, body: bytes, ctype: str, audit: dict | None = None):
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    if audit:
        handler.send_header("X-Audit", json.dumps(audit))
    handler.send_header("Access-Control-Allow-Origin", "*")
    if ctype.startswith('text/html'):
        handler.send_header('X-Frame-Options', 'DENY')
        handler.send_header('Content-Security-Policy', "frame-ancestors 'none'")
    # The bench UI is a single static index.html plus a live JSON API, and the
    # front end and back end must always agree on the parameter schema. A cached
    # copy is not just stale, it is BROKEN: after the parameters were renamed,
    # browsers kept posting the old names for days and the server (rightly, at
    # the time) refused them. No-store makes every load fetch the current page.
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
    handler.send_header("Pragma", "no-cache")
    handler.end_headers()
    handler.wfile.write(body)


def _reset_state():
    with COMPUTE_LOCK:
        STATE['src'] = STATE['ref'] = STATE['src_meta'] = STATE['ref_meta'] = None
        STATE['src_digest'] = STATE['ref_digest'] = None
        STATE['last_png'] = STATE['last_committed_png'] = STATE['last_audit'] = STATE['last_committed_audit'] = None
        STATE['velum_cache_lru'] = OrderedDict()
        STATE['scaled_pair_lru'] = OrderedDict()
        STATE['epoch'] = STATE.get('epoch',0)+1
        _BLEND_MEMO.clear()


class BenchHandler(BaseHTTPRequestHandler):
    server_version = "velum-bench/1.0"

    def log_message(self, format, *args):  # noqa: A002 - name fixed by the base class
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), format % args))

    def _lut_authorize(self, token=False):
        # File-dialog and image-export endpoints are NOT part of the public CORS API.
        if not ipaddress.ip_address(self.client_address[0]).is_loopback:
            raise PermissionError('LUT导出仅允许在运行Velum的本机操作。')
        host = urlparse('http://' + self.headers.get('Host', ''))
        if (host.username or host.password or host.path or host.query or host.fragment or
                host.hostname not in ('localhost', '127.0.0.1', '::1') or
                (host.port or 80) != self.server.server_address[1]):
            raise PermissionError('LUT导出要求本机同源访问。')
        origin = self.headers.get('Origin')
        if origin:
            parsed = urlparse(origin)
            if (parsed.scheme != 'http' or parsed.hostname != host.hostname or
                    (parsed.port or 80) != (host.port or 80) or parsed.username or parsed.password or
                    parsed.path or parsed.query or parsed.fragment):
                raise PermissionError('拒绝跨来源LUT导出请求。')
        if self.headers.get('Sec-Fetch-Site', '') not in ('', 'same-origin', 'none'):
            raise PermissionError('拒绝跨来源LUT导出请求。')
        provided = self.headers.get('X-Velum-Export', '')
        if token and (not provided.isascii() or not secrets.compare_digest(provided, _LUT_TOKEN)):
            raise PermissionError('导出会话已失效，请刷新网页后重试。')

    def _lut_respond(self, code, value, audit=None):
        binary = isinstance(value, bytes)
        body = value if binary else json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/octet-stream' if binary else 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if binary:
            self.send_header('Content-Disposition', 'attachment; filename="Velum-Look.cube"')
        if audit:
            self.send_header('X-Velum-LUT-Audit', json.dumps(audit, ensure_ascii=True))
        self.end_headers()
        self.wfile.write(body)

    def _export_lut(self, native):
        acquired = False
        destination = None
        try:
            self._lut_authorize(token=True)
            acquired = _LUT_LOCK.acquire(blocking=False)
            if not acquired:
                self._lut_respond(409, {'error': '已有LUT导出或保存对话框，请先完成或取消。'})
                return
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                raise ValueError('LUT导出需要JSON参数。')
            if not 0 < int(self.headers.get('Content-Length', '0')) <= 16384:
                raise ValueError('LUT导出参数大小无效。')
            data = json.loads(self._read_body())
            if not isinstance(data, dict) or set(data) - {'params', 'layers', 'side', 'epoch', 'size'}:
                raise ValueError('未知LUT导出参数；不接受文件路径或自定义文件内容。')
            if not isinstance(data.get('params'), dict) or not isinstance(data.get('layers'), dict):
                raise ValueError('缺少调色或图层参数。')
            epoch, side, size = data.get('epoch'), data.get('side'), data.get('size', 65)
            if type(epoch) is not int or type(side) is not int or side < 0 or type(size) is not int or size not in (33, 65):
                raise ValueError('图片状态、分辨率或LUT尺寸无效，请刷新并重新上传图片。')
            params_json = json.dumps(data['params'], allow_nan=False)
            params = _params_from_form({'params': params_json.encode()})
            layers = _layer_settings(data['layers'])
            if not all(np.isfinite(layers[k]) for k in ('luma_match', 'color_match', 'overlay_amount')):
                raise ValueError('图层参数必须为有限数值。')
            with COMPUTE_LOCK:
                if STATE.get('epoch', 0) != epoch or STATE['src'] is None or STATE['ref'] is None:
                    raise ValueError('图片已变化，请等待新图完成后重新导出。')
            if native:
                destination = lut_save.choose_path()
                if destination is None:
                    self._lut_respond(200, {'cancelled': True})
                    return
            with COMPUTE_LOCK:
                if STATE.get('epoch', 0) != epoch:
                    raise ValueError('选择保存路径期间图片已变化，未写入文件，请重新导出。')
                src, ref = _scaled_pair(side)
                raw, _, _ = _cached_velum(side, params, params_json)
                target = _compose_layers(src, ref, raw, layers)
                grid, audit_info = lut.fit_lut(src, target, size=size)
                audit_info['input_space'] = 'sRGB encoded RGB, full range 0..1'
                audit_info['output_space'] = 'sRGB encoded RGB, full range 0..1'
                audit_info['source_size'] = [src.shape[1], src.shape[0]]
                audit_info['layers'] = layers
                audit_info['grade'] = json.loads(_canonical_grade_key(params))
                audit_info['analysis_long_edge_limit'] = side
                content = lut.cube_bytes(grid, metadata=audit_info)
                if native:
                    assert destination is not None
                    saved = lut_save.write_cube(destination, content)
                    self._lut_respond(200, dict(saved, ok=True, audit=audit_info))
                else:
                    self._lut_respond(200, content, audit=audit_info)
        except Exception as exc:
            code = 403 if isinstance(exc, PermissionError) else 400 if isinstance(exc, (ValueError, TypeError)) else 500
            self._lut_respond(code, {'error': str(exc)})
        finally:
            if destination is not None:
                lut_save.discard_path(destination)
            if acquired:
                _LUT_LOCK.release()

    # ------------------------------------------------------------ helpers
    def _read_body(self) -> bytes:
        raw = self.headers.get("Content-Length")
        try:
            length = int(raw) if raw is not None else 0
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > _BODY_LIMIT:
            raise ValueError("request body too large")
        return self.rfile.read(length) if length else b""

    def _serve_ui(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "index.html"), "rb") as f:
            return f.read()

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        if urlparse(self.path).path == '/lut_export.js':
            _respond(self, 200, (Path(__file__).parent / 'lut_export.js').read_bytes(), 'application/javascript; charset=utf-8')
            return
        if urlparse(self.path).path == '/api/lut/capabilities':
            try:
                self._lut_authorize()
                self._lut_respond(200, {'token': _LUT_TOKEN, 'native_save': os.name == 'nt', 'size': 65})
            except (ValueError, PermissionError) as exc:
                self._lut_respond(403, {'error': str(exc)})
            return
        path = urlparse(self.path).path
        if path == "/":
            _respond(self, 200, self._serve_ui(), "text/html; charset=utf-8")
        elif path in ("/velum-title-warm.png", "/velum_icon.png", "/favicon.ico"):
            # Explicit UI assets. Browsers commonly request /favicon.ico even when a
            # PNG icon is declared, so serve the same designer-supplied icon there too.
            here = os.path.dirname(os.path.abspath(__file__))
            filename = ("velum-title-warm.png" if path == "/velum-title-warm.png"
                        else "velum_icon.png")
            with open(os.path.join(here, filename), "rb") as f:
                _respond(self, 200, f.read(), "image/png")
        elif path == "/api/status":
            params = VelumParams()
            spec = {}
            import dataclasses
            for f in dataclasses.fields(params):
                k, v = f.name, getattr(params, f.name)
                if k.startswith("_"):
                    continue
                spec[k] = {"value": v, "type": type(v).__name__}
            spec["engine_version"] = {"value": "4.10", "type": "str"}
            spec["standalone_cpu_only"] = {"value": CPU_ONLY, "type": "bool"}
            _respond(self, 200, json.dumps(spec).encode(), "application/json")
        elif path == "/api/reset":
            _reset_state()
            _respond(self, 200, b'{"ok":true}', "application/json")
        elif path == "/api/grade_stream":
            self._grade_stream(urlparse(self.path).query)
        elif path == "/api/last_result":
            # fetch the PNG produced by the most recent /api/grade_stream run
            png = STATE.get("last_committed_png")
            if png is None:
                _respond(self, 404, b'{"error":"no result"}', "application/json")
            else:
                _respond(self, 200, png, "image/png",
                          audit=STATE.get("last_committed_audit"))
        else:
            _respond(self, 404, b"not found", "text/plain")

    def _grade_stream(self, query: str):
        with COMPUTE_LOCK:
            try:
                return self._grade_stream_locked(query)
            except (ValueError, TypeError) as exc:
                _respond(self,400,json.dumps({'error':str(exc)}).encode(),'application/json')

    def _grade_stream_locked(self, query: str):
        """Server-Sent Events: stream real per-STAGE progress of one grade.

        Why SSE rather than a normal POST: /api/compare and /api/grade block until
        the whole pipeline is done, so the client can only show a spinner. grade()
        now takes a progress(fraction, label) callback; here we forward every
        stage boundary to the browser as it happens, so the bar advances step by
        step and names the stage that is running -- the point being debuggability
        (you can see WHICH stage is slow).

        The final PNG is NOT sent inline (SSE is a text protocol and a full-res
        PNG is large/base64-bloated). Instead it is cached in STATE and the client
        fetches it from /api/last_result on the 'done' event. mode=grid uses the
        six-panel sheet; mode=single uses the plain VELUM output.
        """
        from urllib.parse import parse_qs

        qs = parse_qs(query)
        params_json = qs.get("params", ["{}"]) [0]
        layer_json = qs.get("layers", ["{}"]) [0]
        layers = _parse_layer_json(layer_json)
        mode = qs.get("mode", ["single"])[0]
        try:
            side = max(int(float(qs.get("max_output_side", ["0"])[0])), 0)
        except (ValueError, TypeError):
            side = 0

        if STATE["src"] is None or STATE["ref"] is None:
            _respond(self, 400, b'{"error":"set images first"}', "application/json")
            return

        params = _params_from_form({"params": params_json.encode("utf-8")})

        # SSE headers (write directly; this response is a live stream, not _respond)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")   # disable proxy buffering
        self.end_headers()

        def send(event: str, data: dict):
            try:
                payload = (f"event: {event}\ndata: {json.dumps(data)}\n\n"
                           ).encode("utf-8")
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise   # client navigated away; abort the grade

        import time as _t
        src_g, ref_g = _scaled_pair(side)
        t0 = _t.time()

        def on_progress(frac, label):
            info = params.debug.get("compute_backend", {})
            effective = info.get("effective", "pending") if isinstance(info, dict) else "pending"
            device_label = {"cpu": "CPU", "gpu": "CPU + GPU映射", "not_used": "CPU"}.get(effective, "设备待定")
            send("progress", {"frac": round(float(frac), 4),
                              "label": f"[{device_label}] {label}",
                              "backend": effective,
                              "elapsed_ms": int((_t.time() - t0) * 1000)})

        # VELUM RESULT CACHE (the expensive part) is handled by _cached_velum,
        # shared with /api/ab so EVERY mode reuses the same computed VELUM.
        try:
            out, a, hit = _cached_velum(side, params, params_json, on_progress)
            if hit:
                effective = a.get('compute_backend',{}).get('effective','not_used')
                device_label = 'GPU结果缓存' if effective=='gpu' else 'CPU结果缓存'
                send("progress", {"frac": 0.95,
                                   "label": f"[{device_label}] 复用 VELUM 缓存",
                                   "backend": effective,
                                   "elapsed_ms": int((_t.time() - t0) * 1000)})

            display_out = _compose_layers(src_g, ref_g, out, layers)
            # The palette/theme must describe what the user actually sees after
            # layer modulation, not the cached raw VELUM underneath it.
            a = dict(a)
            a["palette"] = _display_palette(display_out, layers, a)
            if mode == "grid":
                send("progress", {"frac": 0.97, "label": "拼接六格对比图 (reinhard+mkl+hm)",
                                  "elapsed_ms": int((_t.time() - t0) * 1000)})
                sheet = _compose_sheet(src_g, ref_g, params, display_out,
                                       palette=a.get("palette"))
                png = _encode_png(sheet, dither=False)
            else:
                png = _encode_png(display_out, dither=params.dither)
            a["time_ms"] = int((_t.time() - t0) * 1000)
            STATE["last_png"] = png
            STATE["last_committed_png"] = png
            STATE["last_committed_audit"] = a
            STATE["last_audit"] = a
            send("done", {"time_ms": a["time_ms"], "audit": a})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                send("error", {"error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass

    # --------------------------------------------------------------- POST
    def do_POST(self):
        path = urlparse(self.path).path
        if path in ('/api/lut/save', '/api/lut/cube'):
            return self._export_lut(native=path.endswith('/save'))
        with COMPUTE_LOCK:
            return self._do_POST_locked()

    def _do_POST_locked(self):
        """Dispatch, with a blanket guard so no request can kill the connection.

        Without this, any unhandled exception propagates into socketserver, which
        prints a traceback and drops the socket -- the browser then shows a bare
        network error with no clue what went wrong, and the bench looks broken
        rather than merely misconfigured. Returning the message as JSON turns
        every failure into something diagnosable from the UI.
        """
        try:
            self._do_post()
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                _respond(self, 400 if isinstance(e, (ValueError, json.JSONDecodeError)) else 500,
                         json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(),
                         "application/json")
            except Exception:
                pass          # client already gone; nothing useful left to do

    def _do_post(self):
        path = urlparse(self.path).path
        ctype = self.headers.get("Content-Type", "")
        body = self._read_body()

        if path == '/api/reset':
            _reset_state()
            _respond(self,200,b'{"ok":true}','application/json')
        elif path == "/api/set-images":
            fields = parse_multipart(body, ctype) if "multipart" in ctype else {}
            sys.stderr.write(f"[set-images] ctype={ctype!r} body_len={len(body)} "
                             f"fields={list(fields.keys())}\n")
            ok_src = ok_ref = False
            submitted_src = submitted_ref = False
            decoded = {}
            digests = {}
            for name in ("source", "reference"):
                if name in fields:
                    if name == "source":
                        submitted_src = True
                    else:
                        submitted_ref = True
                    digest = hashlib.sha256(fields[name]).hexdigest()
                    digests[name] = digest
                    digest_key = "src_digest" if name == "source" else "ref_digest"
                    if STATE.get(digest_key) == digest:
                        continue
                    try:
                        decoded[name] = decode_rgb(fields[name])
                    except (ValueError, TypeError) as exc:
                        _respond(self, 400, json.dumps({"error": str(exc)}).encode(), "application/json")
                        return
            if "source" in decoded:
                img = decoded["source"]
                sys.stderr.write(f"[set-images] source decode: "
                                 f"{None if img is None else img.shape}, "
                                 f"len={len(fields['source'])}\n")
                if img is not None:
                    STATE["src"] = img
                    STATE["src_meta"] = f"{img.shape[1]}x{img.shape[0]}"
                    ok_src = True
                    STATE["src_digest"] = digests["source"]
            if "reference" in decoded:
                img = decoded["reference"]
                if img is not None:
                    STATE["ref"] = img
                    STATE["ref_meta"] = f"{img.shape[1]}x{img.shape[0]}"
                    ok_ref = True
                    STATE["ref_digest"] = digests["reference"]
            if ok_src or ok_ref:
                # New image(s) -> the cached VELUM is stale. Bump the epoch and
                # drop the cache so the next grade recomputes. (Bumping an epoch
                # is more robust than relying on id() of the numpy arrays, which
                # a GC could recycle.)
                STATE["epoch"] = STATE.get("epoch", 0) + 1
                # Raw outputs and scaled SOURCE images are tied to the image pair epoch.
                # Reference statistics are content-addressed and deliberately survive
                # source swaps and repeated uploads of identical reference pixels.
                STATE["velum_cache_lru"] = OrderedDict()
                STATE["scaled_pair_lru"] = OrderedDict()
                STATE['last_png'] = STATE['last_committed_png'] = STATE['last_audit'] = STATE['last_committed_audit'] = None
                _BLEND_MEMO.clear()
            _respond(self, 200, json.dumps({
                "ok": submitted_src or submitted_ref,
                "source": ok_src, "reference": ok_ref,
                "epoch": STATE['epoch'],
                "meta": {"source": STATE["src_meta"], "reference": STATE["ref_meta"]},
            }).encode(), "application/json")

        elif path == "/api/grade":
            fields = parse_multipart(body, ctype) if "multipart" in ctype else {}
            params = _params_from_form(fields)
            layers = _parse_layer_json(fields.get("layers", b"{}"))
            if STATE["src"] is None or STATE["ref"] is None:
                _respond(self, 400, b'{"error":"set images first"}', "application/json")
                return
            side = _output_side_from_form(fields)
            src_g, ref_g = _scaled_pair(side)
            t0 = time.time()
            # Go through the cache, so a following /api/render_layers (which only
            # recomposes display layers) finds this grade instead of 409-ing and
            # forcing a full recompute.
            params_json = fields.get("params", b"{}").decode("utf-8", "replace")
            try:
                out, _cached_audit, _hit = _cached_velum(side, params, params_json)
            except Exception as e:
                _respond(self, 500, json.dumps({"error": str(e)}).encode(), "application/json")
                return
            display_out = _compose_layers(src_g, ref_g, out, layers)
            dt = time.time() - t0
            a = dict(_cached_audit)
            if not _identity_layers(layers):
                a.update(audit(display_out, ref_g))
            a["palette"] = _display_palette(display_out, layers, a)
            a["time_ms"] = int(dt * 1000)
            a["out_w"] = int(out.shape[1])
            a["out_h"] = int(out.shape[0])
            # Surface the transport's own record so the bench can see the derived
            # gain bound and what the solve actually did on this pair.
            for key in ("joint_transport", "continuous_mapping"):
                v = params.debug.get(key)
                if isinstance(v, dict):
                    a[key] = v
            png = _encode_png(display_out, dither=params.dither)
            a["time_ms"] = int((time.time() - t0) * 1000)
            STATE["last_committed_png"] = png
            STATE['last_png'] = png
            STATE['last_audit'] = STATE['last_committed_audit'] = a
            _respond(self, 200, png,
                     "image/png", audit=a)

        elif path == "/api/compare":
            fields = parse_multipart(body, ctype) if "multipart" in ctype else {}
            params = _params_from_form(fields)
            if STATE["src"] is None or STATE["ref"] is None:
                _respond(self, 400, b'{"error":"set images first"}', "application/json")
                return
            side = _output_side_from_form(fields)
            src_g, ref_g = _scaled_pair(side)
            t0 = time.time()
            params_json = fields.get("params", b"{}").decode("utf-8", "replace")
            out, cached_audit, _hit = _cached_velum(
                side, params, params_json)
            palette = cached_audit["palette"]
            sheet = _compose_sheet(src_g, ref_g, params, out, palette=palette)
            a = dict(cached_audit)
            a["time_ms"] = int((time.time() - t0) * 1000)
            a["out_w"] = int(out.shape[1])
            a["out_h"] = int(out.shape[0])
            a["palette"] = palette
            for stage in ("compute_backend", "joint_transport", "continuous_mapping"):
                if stage in cached_audit:
                    a[stage] = cached_audit[stage]
            _respond(self, 200, _encode_png(sheet, dither=False), "image/png", audit=a)

        elif path == "/api/render_layers":
            """Fast post-grade render: reuse cached raw VELUM, never call grade()."""
            fields = parse_multipart(body, ctype) if "multipart" in ctype else {}
            if STATE["src"] is None or STATE["ref"] is None:
                _respond(self, 400, b'{"error":"set images first"}', "application/json")
                return
            layers = _parse_layer_json(fields.get("layers", b"{}"))
            mode = fields.get("mode", b"single").decode("utf-8", "replace").lower()
            params = _params_from_form(fields)
            params_json = fields.get("params", b"{}").decode("utf-8", "replace")
            side = _output_side_from_form(fields)
            expected_key = (STATE.get("epoch", 0), int(side),
                            _canonical_grade_key(params))
            cached = _lru_get("velum_cache_lru", expected_key)
            if cached is None:
                _respond(self, 409, b'{"error":"velum cache parameters mismatch"}',
                         "application/json")
                return
            src_g, ref_g = _scaled_pair(side)
            raw = cached["out"]
            # Refuse mismatched resolution instead of silently resizing an old
            # cached grade. The client will fall back to the normal grade path.
            if raw.shape[:2] != src_g.shape[:2]:
                _respond(self, 409, b'{"error":"velum cache resolution mismatch"}',
                         "application/json")
                return
            # `fast=1` marks an INTERACTIVE frame (a slider drag in progress).
            # Two per-frame costs are then skipped, both of which the client does
            # not consume mid-drag anyway:
            #   * dominant_palette -- a full-image quantisation (117 ms @1K,
            #     265 ms @2K). The UI now renders the palette bar on a trailing
            #     debounce, so a per-frame palette is computed and thrown away.
            #   * lossless PNG -- replaced by a q92 JPEG (see _encode_preview).
            # The committed (non-fast) render still produces PNG + palette, so
            # nothing the user keeps is lossy or missing data.
            fast = fields.get("fast", b"0").decode("utf-8", "replace").strip() in ("1", "true")
            t0 = time.time()
            display_out = _compose_layers(src_g, ref_g, raw, layers)
            a = dict(cached.get("audit", {}))
            a.update(cache_hit=True,cached_algorithm_seconds=a.get('algorithm_seconds',0.0),algorithm_seconds=0.0)
            a.update(cached_rpc_seconds=a.get('rpc_seconds',0.0),rpc_seconds=0.0)
            if mode == "grid":
                a["palette"] = _display_palette(display_out, layers, a)
                payload_img = _compose_sheet(src_g, ref_g, params, display_out,
                                             palette=a.get("palette"))
                png = _encode_png(payload_img, dither=False)
                ctype_out = "image/png"
            elif fast:
                a.pop("palette", None)          # client keeps the previous one
                a["fast"] = True
                png, ctype_out = _encode_preview(display_out)
            else:
                a["palette"] = _display_palette(display_out, layers, a)
                png = _encode_png(display_out, dither=params.dither)
                ctype_out = "image/png"
            a["time_ms"] = int((time.time() - t0) * 1000)
            STATE["last_png"] = png
            STATE["last_audit"] = a
            if ctype_out == 'image/png':
                STATE['last_committed_png'] = png
                STATE['last_committed_audit'] = a
            _respond(self, 200, png, ctype_out, audit=a)

        elif path == "/api/ab":
            """A/B compare: return VELUM and a chosen BASELINE as one JSON payload.

            Both images are produced in the SAME request with the SAME params, so
            they are perfectly comparable (same downscales, same autotune, same
            moment). Base64 PNG in a JSON body -- for a local bench this is simple
            and avoids two round-trips that could disagree. The front end overlays
            the two images and drives a vertical divider between them. `baseline`
            (form field) selects which reference algorithm to compare against.
            """
            fields = parse_multipart(body, ctype) if "multipart" in ctype else {}
            params = _params_from_form(fields)
            layers = _parse_layer_json(fields.get("layers", b"{}"))
            if STATE["src"] is None or STATE["ref"] is None:
                _respond(self, 400, b'{"error":"set images first"}', "application/json")
                return
            baseline = "hm"
            if "baseline" in fields:
                try:
                    baseline = fields["baseline"].decode("utf-8").strip().lower()
                except Exception:
                    baseline = "hm"
            import base64
            # Use the SAME resolution rung as the single/grid stream so the A/B
            # view reuses the cached VELUM instead of recomputing it -- switching
            # into A/B was the one mode that still felt slow because it used a
            # different (1920-capped) size, which never matched the cache key.
            # grade() is the cost; reinhard/hm/source are cheap, so dropping the
            # separate cap costs almost nothing on a cache hit and is worth the
            # instant switch. (params_json is rebuilt from the same field so the
            # key matches exactly.)
            params_json = fields.get("params", b"{}").decode("utf-8", "replace")
            side = _output_side_from_form(fields)
            t0 = time.time()
            out, a, hit = _cached_velum(side, params, params_json)
            src_ab, ref_ab = _scaled_pair(side)
            display_out = _compose_layers(src_ab, ref_ab, out, layers)
            lut_audit = None
            if baseline == "reinhard":
                base_img = reinhard_lab(src_ab, ref_ab)
            elif baseline == "mkl":
                base_img = mkl_transfer(src_ab, ref_ab)
            elif baseline == "cube":
                # Exactly what 导出.cube writes for these settings, replayed on
                # the source: native VELUM vs its 65^3 LUT approximation.
                grid, lut_audit = lut.fit_lut(src_ab, display_out, size=65)
                base_img = lut.apply_lut(src_ab, grid)
            elif baseline == "source":
                # "与源图相比" -- the source image itself is the baseline; velum is
                # shown on the right so the user can see the colour change directly.
                # No extra algorithm needed: just encode the (possibly downscaled)
                # source. Size-match to the output so the split view aligns.
                import cv2
                base_img = cv2.resize(
                    src_ab, (out.shape[1], out.shape[0]),
                    interpolation=cv2.INTER_AREA) if src_ab.shape[:2] != out.shape[:2] \
                    else src_ab
            else:
                baseline = "hm"
                base_img = hm_mkl_hm(src_ab, ref_ab)
            dt = time.time() - t0

            def b64(img):
                return base64.b64encode(_encode_png(img, dither=params.dither)).decode("ascii")

            # `a` (incl. auto) comes from _cached_velum; on a cache hit it carries
            # the auto values that params.debug no longer has. Just stamp timing.
            a = dict(a)
            a["palette"] = _display_palette(display_out, layers, a)
            a["time_ms"] = int(dt * 1000)
            payload = {
                "base": b64(base_img),
                "baseline": baseline,
                "velum": b64(display_out),
                "w": int(display_out.shape[1]),
                "h": int(display_out.shape[0]),
            }
            if lut_audit is not None:
                payload["lut"] = {k: lut_audit[k] for k in
                                  ("rmse_255", "p95_error_255", "max_error_255", "size", "fit_seconds")}
            a["time_ms"] = int((time.time() - t0) * 1000)
            _respond(self, 200, json.dumps(payload).encode(), "application/json", audit=a)

        else:
            _respond(self, 404, b"not found", "text/plain")

    # ---------------------------------------------------------- OPTIONS/CORS
    def do_OPTIONS(self):
        if urlparse(self.path).path.startswith('/api/lut/'):
            self._lut_respond(403, {'error': 'Cross-origin LUT access is disabled.'})
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--open-browser", action="store_true",
                    help="open the bench in the default browser once the server "
                         "is actually listening (after the port is chosen)")
    ap.add_argument("--no-port-jump", action="store_true",
                    help="fail instead of auto-incrementing when the port is busy")
    args = ap.parse_args()
    if not 0 <= args.port <= 65535:
        ap.error('--port must be 0..65535')

    # Bind the port ourselves so we can (a) pick a free port when the requested
    # one is taken, and (b) know the REAL port before telling the browser. The
    # old behaviour opened the browser before the server was listening, which
    # produced ERR_CONNECTION_REFUSED on every launch.
    import socket
    port = args.port
    last_err = None
    sock = None
    for attempt in range(50):
        if port > 65535:
            print('No available port in the permitted range',flush=True)
            return 2
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((args.host, port))
            sock.listen(128)
            port = sock.getsockname()[1]
            break
        except OSError as e:
            sock.close()
            if args.no_port_jump:
                print(f"velum bench: port {port} is busy "
                      f"({e}); use --port to pick another", flush=True)
                return 2
            last_err = e
            port += 1
    else:
        print(f"velum bench: could not find a free port near {args.port} "
              f"(last: {last_err})", flush=True)
        return 2

    assert sock is not None
    browser_host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    print(f"velum 4.10 -> http://{browser_host}:{port}", flush=True)
    if args.open_browser:
        import threading
        import webbrowser
        # Defer a hair so the socket is fully accepting; then the page loads on
        # the FIRST attempt instead of after a refresh.
        threading.Timer(0.3, lambda: webbrowser.open(f"http://{browser_host}:{port}"),
                        ).start()

    httpd = ThreadingHTTPServer((args.host, port), BenchHandler, bind_and_activate=False)
    httpd.socket = sock
    httpd.server_bind = lambda: None
    httpd.server_activate = lambda: None
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
