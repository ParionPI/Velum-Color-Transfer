"""Web image adaptation; RGB8 sRGB and deterministic, non-dithered output."""
import io
import warnings
import numpy as np
from PIL import Image, ImageCms, ImageOps

SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()


def decode_rgb(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as opened:
                if opened.width*opened.height > 64*1024*1024:
                    raise ValueError('图片超过64MP，请先降低分辨率')
                if opened.mode in ('I','F') or opened.mode.startswith('I;16'):
                    raise ValueError('当前内核使用8位RGB，请先显式转换16位/浮点图像')
                profile = opened.info.get('icc_profile')
                image = ImageOps.exif_transpose(opened)
                # Preserve the historical RGB upload contract: alpha is not a mask.
                # Palette/grayscale input is expanded; CMYK retains its profile input mode.
                if image.mode not in ('RGB','RGBA','CMYK'):
                    image = image.convert('RGB')
                if profile:
                    image = ImageCms.profileToProfile(image,ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                                     ImageCms.createProfile('sRGB'),outputMode='RGB')
                else:
                    image = image.convert('RGB')
                return np.asarray(image,dtype=np.float32).copy()/255.0
    except Exception as exc:
        raise ValueError(f'无法读取图片：{exc}') from exc


def load_rgb(path):
    with open(path,'rb') as stream:
        return decode_rgb(stream.read()), None


def quantise(rgb01,bits=8,dither=False,seed=0):
    if bits != 8:
        raise ValueError('4.10网页输出为8位RGB PNG')
    pixels=np.asarray(rgb01,dtype=np.float32)
    if not np.all(np.isfinite(pixels)):
        raise ValueError('输出含NaN或Inf')
    # Compatibility signature only: the accepted engine never adds output noise.
    return np.rint(np.clip(pixels,0,1)*255).astype(np.uint8)


def resize_max_side(rgb01,max_side):
    import cv2
    h,w=rgb01.shape[:2]
    if max_side <= 0 or max(h,w) <= max_side:
        return rgb01
    ratio=max_side/max(h,w)
    return cv2.resize(rgb01,(max(1,round(w*ratio)),max(1,round(h*ratio))),interpolation=cv2.INTER_AREA)
