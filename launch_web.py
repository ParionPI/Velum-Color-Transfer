"""Offline standalone entry point. ComfyUI never imports this launcher."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import traceback
import warnings

ROOT = Path(__file__).resolve().parent
DEPENDENCIES = [('numpy', 'numpy'), ('scipy', 'scipy'), ('numba', 'numba'),
                ('llvmlite', 'llvmlite'), ('cv2', 'opencv-python'), ('PIL', 'Pillow'),
                ('colour', 'colour-science'), ('imageio', 'imageio'),
                ('typing_extensions', 'typing_extensions')]


def self_test(work_root):
    import numpy as np
    from isolated_client import IsolatedVelumClient
    y, x = np.mgrid[:48, :64]
    source = np.stack((x * 4, y * 5, (x + y) * 2), axis=-1).astype(np.uint8)
    reference = np.ascontiguousarray(source[..., ::-1])
    client = IsolatedVelumClient(runtime_root=work_root / 'self-test', idle_seconds=0)
    try:
        result, response = client.call(source, reference, deband_mode='auto', manual_strength=1,
                                       backend='cpu', cpu_workers=2)
        assert result.shape == source.shape and result.dtype == np.uint8
        assert response['backend']['effective'] == 'cpu'
        return {'shape': list(result.shape), 'backend': response['backend'],
                'algorithm_seconds': response['algorithm_seconds'],
                'worker_log_dir': response['worker_log_dir']}
    finally:
        client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Velum offline standalone launcher')
    parser.add_argument('port', type=int, nargs='?', default=8899)
    parser.add_argument('--cpu', action='store_true', help='use the CPU-only standalone mode')
    parser.add_argument('--check-only', action='store_true', help='check real native imports and exit')
    parser.add_argument('--self-test', action='store_true', help='also run a small isolated CPU grade and exit')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--no-port-jump', action='store_true')
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        configure = getattr(stream, 'reconfigure', None)
        if callable(configure):
            configure(encoding='utf-8')
    work_root = Path(os.environ.get('LOCALAPPDATA') or tempfile.gettempdir()) / 'Velum' / 'standalone'
    report = {'python': sys.executable, 'version': sys.version, 'root': str(ROOT),
              'cpu_only': args.cpu, 'isolated': bool(sys.flags.isolated), 'dependencies': {},
              'bundled_runtime': Path(sys.prefix).resolve() == (ROOT / 'runtime').resolve()}
    log = work_root / f'startup-{os.getpid()}.json'
    phase = 'startup'
    try:
        work_root.mkdir(parents=True, exist_ok=True)
        if not sys.flags.isolated:
            raise RuntimeError('请通过一键启动.bat启动，或使用 python -I -B launch_web.py。')
        os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                          NUMBA_THREADING_LAYER='workqueue', NUMBA_NUM_THREADS='8',
                          NUMBA_CACHE_DIR=str(work_root / 'cache410'),
                          VELUM_WEB_RUNTIME_ROOT=str(work_root / 'workers'))
        if args.cpu:
            os.environ['VELUM_STANDALONE_CPU'] = '1'
            os.environ['VELUM_DISABLE_GPU'] = '1'
            # Plotting is optional and deliberately absent from the CPU runtime.
            warnings.filterwarnings('ignore', message=r'"Matplotlib" related API features are not available:.*',
                                    module=r'colour\.utilities\.verbose')
        sys.path.insert(0, str(ROOT))
        print(f'Python: {sys.executable}', flush=True)
        print('模式：独立 CPU 整合包' if args.cpu else '模式：已有 Python 环境', flush=True)
        for module, distribution in DEPENDENCIES:
            phase = module
            importlib.import_module(module)
            report['dependencies'][distribution] = importlib.metadata.version(distribution)
        phase = 'native image libraries'
        import numpy as np
        import cv2
        from PIL import ImageCms
        cv2.setNumThreads(1)
        ImageCms.createProfile('sRGB')
        cv2.cvtColor(np.zeros((2, 2, 3), np.uint8), cv2.COLOR_RGB2BGR)
        for name in ('worker.py', '_velum410/__init__.py', 'web/server.py', 'web/index.html'):
            if not (ROOT / name).is_file():
                raise FileNotFoundError(f'整合包不完整：缺少 {name}；请完整解压，不要在压缩包中直接运行。')
        if args.self_test:
            phase = 'isolated CPU grade'
            report['self_test'] = self_test(work_root)
        report['ok'] = True
        log.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'自检通过。诊断记录：{log}', flush=True)
        if args.check_only or args.self_test:
            return 0
        phase = 'web server'
        sys.argv = [str(ROOT / 'web' / 'server.py'), '--port', str(args.port)]
        if not args.no_browser:
            sys.argv.append('--open-browser')
        if args.no_port_jump:
            sys.argv.append('--no-port-jump')
        runpy.run_path(str(ROOT / 'web' / 'server.py'), run_name='__main__')
        return 0
    except Exception:
        report.update(ok=False, failed_phase=phase, error=traceback.format_exc())
        try:
            log.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError:
            pass
        print(f'启动失败（{phase}）。解释器：{sys.executable}\n{report["error"]}', file=sys.stderr)
        print(f'请完整解压到本地目录，并将此诊断记录发给维护者：{log}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
