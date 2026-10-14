"""Only this child main thread imports and executes the native Velum engine."""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import threading


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--cpu-workers',type=int,required=True)
    parser.add_argument('--cuda-device',type=int)
    args=parser.parse_args()
    runtime=args.runtime_dir.resolve(); runtime.mkdir(parents=True,exist_ok=True)
    for stream in (sys.stdout,sys.stderr):
        configure=getattr(stream,'reconfigure',None)
        if callable(configure): configure(encoding='utf-8')
    def runtime_info(phase):
        info={'phase':phase,'pid':os.getpid(),'executable':sys.executable,'python':sys.version,
              'isolated':sys.flags.isolated,'ignore_environment':sys.flags.ignore_environment,
              'user_site_disabled':sys.flags.no_user_site,'threads':[t.name for t in threading.enumerate()],
              'thread_modules':{t.name:getattr(t.run,'__module__',None) for t in threading.enumerate()},
               'scorer_backend':getattr(sys.modules.get('_velum410.legacy25.auto_score'),'SCORER_BACKEND',None),
               'native_modules':{name:getattr(sys.modules.get(name),'__file__',None)
                                 for name in ('numpy','cv2','numba','scipy','colour','torch')},
              'startup_modules':{name:getattr(module,'__file__',None) for name,module in tuple(sys.modules.items())
                                 if name.startswith('enhanced_experience') or name in ('sitecustomize','usercustomize')},
              'native_environment':{k:os.environ.get(k) for k in ('NUMBA_THREADING_LAYER','NUMBA_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}}
        (runtime/'runtime_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    runtime_info('before_native_imports')
    if not sys.flags.isolated:
        raise RuntimeError('Worker requires Python -I. Restart ComfyUI or the standalone webpage to load the updated client.')
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    if not 1<=args.cpu_workers<=64: raise ValueError('Invalid worker count')
    os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        NUMBA_NUM_THREADS=str(min(args.cpu_workers,os.cpu_count() or 1)),NUMBA_THREADING_LAYER='workqueue',
        NUMBA_CACHE_DIR=str(args.cache_dir.resolve()),PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    faulthandler.enable(sys.stderr,all_threads=True)
    bundle = Path(__file__).resolve().parent
    portable = bundle / 'runtime'
    dll_directory = None
    if os.name == 'nt' and sys.version_info[:2] == (3, 12) and (portable / 'python312._pth').is_file():
        # Only the isolated child uses these ABI-matched CPU dependencies.
        # Torch/CUDA still comes from the ComfyUI interpreter's normal site-packages.
        sys.path.insert(0, str(portable / 'Lib' / 'site-packages'))
        dll_directory = os.add_dll_directory(str(portable))  # Keep handle alive for the worker lifetime.
    import numpy as np
    import cv2
    import numba
    cv2.setNumThreads(1)
    # -I removes script/PYTHONPATH search entries. Add only our trusted bundle.
    sys.path.insert(0,str(bundle))
    import _velum410 as engine
    if engine.__version__!='4.10': raise RuntimeError('Incorrect bundled engine')
    from _velum410.legacy25 import auto_score
    if auto_score.SCORER_BACKEND!='numpy_no_jit' or hasattr(auto_score.score_amplitudes,'py_func'):
        raise RuntimeError('Unsafe/stale JIT scorer loaded; restart with the NumPy scoring hotfix')
    runtime_info('engine_loaded_numpy_scorer')
    reference_cache=None
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line)
            if request.get('op')=='stop': return 0
            directory=Path(request['request_dir']).resolve()
            if directory.parent!=runtime or not directory.name.startswith('request_'): raise ValueError('Request escaped private runtime')
            for name in ('source','reference'):
                if Path(request[name]).resolve()!=directory/(name+'.npy'): raise ValueError('Unexpected input path')
            source=np.load(request['source'],allow_pickle=False)
            ref_hash=hashlib.sha256(Path(request['reference']).read_bytes()).hexdigest()
            if reference_cache is None or reference_cache[0]!=ref_hash:
                reference_cache=(ref_hash,np.load(request['reference'],allow_pickle=False))
            reference=reference_cache[1]
            for array in (source,reference):
                if array.dtype!=np.uint8 or array.ndim!=3 or array.shape[2]!=3 or min(array.shape)<=0: raise ValueError('Expected uint8 HWC3')
            mode,backend=request['deband_mode'],request['backend']
            if mode not in {'off','auto','manual'} or backend not in {'auto','cpu','gpu'}: raise ValueError('Invalid mode/backend')
            options=engine.DebandOptions(enabled=mode!='off',auto=mode=='auto',strength=float(request['manual_strength']) if mode=='manual' else 1.0,
                                        structure_field=request.get('structure_field','gradient_energy'))
            if backend!='cpu' and options.active and os.environ.get('VELUM_DISABLE_GPU')!='1':
                try:
                    import torch
                    torch.set_num_threads(1)
                    if args.cuda_device is not None and torch.cuda.is_available(): torch.cuda.set_device(args.cuda_device)
                except Exception as exc:
                    if backend=='gpu': raise
                    # Keep automatic fallback even when CUDA setup itself fails.
                    from _velum410 import gpu_mapping
                    with gpu_mapping._PROBE_LOCK:
                        gpu_mapping._PROBE_DONE=True; gpu_mapping._PROBE_TORCH=None
                        gpu_mapping._PROBE_REASON=f'worker CUDA setup: {type(exc).__name__}: {str(exc)[:200]}'
            faulthandler.dump_traceback_later(120,repeat=False)
            runtime_info('before_transfer_numpy_scorer')
            started=time.perf_counter()
            result=engine.transfer(source,reference,performance=engine.Performance(workers=args.cpu_workers,backend=backend),deband=options)
            elapsed=time.perf_counter()-started
            faulthandler.cancel_dump_traceback_later()
            runtime_info('after_transfer_numpy_scorer')
            if result.rgb.dtype!=np.uint8 or result.rgb.shape!=source.shape: raise RuntimeError('Invalid engine output')
            output=directory/'output.npy'; np.save(output,result.rgb,allow_pickle=False)
            print('VELUM_JSON:'+json.dumps({'id':request['id'],'ok':True,'output':str(output),'version':engine.__version__,
                'stability_revision':'numpy_scorer_r1',
                'worker_pid':os.getpid(),'threading_layer':getattr(numba,'threading_layer')(),'algorithm_seconds':elapsed,
                'backend':result.stats['performance']['backend'],'stats':result.stats},allow_nan=False),flush=True)
        except Exception as exc:
            faulthandler.cancel_dump_traceback_later()
            traceback.print_exc(file=sys.stderr)
            print('VELUM_JSON:'+json.dumps({'id':request.get('id'),'ok':False,'error':f'{type(exc).__name__}: {exc}'}),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
