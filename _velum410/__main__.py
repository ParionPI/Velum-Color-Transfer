"""4.10 CLI with safe-worker integration and CUDA acceleration."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms, ImageOps
from . import transfer, Performance, DebandOptions


def load_srgb(path):
    with Image.open(path) as image:
        if ('A' in image.getbands() or 'transparency' in image.info) and np.any(np.asarray(image.convert('RGBA'))[..., 3] != 255):
            raise ValueError('Non-opaque input requires an explicit compositing policy.')
        profile = image.info.get('icc_profile')
        image = ImageOps.exif_transpose(image).convert('RGB')
        if profile:
            image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                              ImageCms.createProfile('sRGB'), outputMode='RGB')
        return np.array(image, np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Velum 4.10: safe-worker integration with GPU fallback')
    parser.add_argument('source', type=Path); parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--deband', choices=['off', 'auto', 'manual'], default='auto')
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--backend', choices=['auto', 'cpu', 'gpu'], default='auto',
                        help='auto falls back safely; cpu skips CUDA; gpu requires CUDA')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.output.suffix.lower() != '.png':
        parser.error('Output must be a new lossless PNG.')
    if args.output.exists() or (args.report and args.report.exists()):
        parser.error('Refusing to overwrite an existing artifact.')
    if args.report and args.report.resolve() == args.output.resolve():
        parser.error('Report and PNG must have different paths.')
    result = transfer(load_srgb(args.source), load_srgb(args.reference),
                      performance=Performance(workers=args.workers, backend=args.backend),
                      deband=DebandOptions(enabled=args.deband != 'off', auto=args.deband == 'auto', strength=args.strength))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(result.rgb).save(args.output, compress_level=3, icc_profile=profile)
    text = json.dumps(result.stats, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
