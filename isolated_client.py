"""Persistent process boundary: no engine, Numba, SciPy, OpenCV or Torch here."""
from __future__ import annotations
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import numpy as np


class IsolatedVelumClient:
    def __init__(self, *, worker_script=None, timeout_seconds=None, runtime_root=None, idle_seconds=None):
        self.worker_script=Path(worker_script or Path(__file__).with_name('worker.py')).resolve()
        self.timeout_seconds=float(timeout_seconds if timeout_seconds is not None else os.environ.get('VELUM410_WORKER_TIMEOUT','180'))
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds<=0: raise ValueError('Worker timeout must be finite and positive')
        self.idle_seconds=float(idle_seconds if idle_seconds is not None else os.environ.get('VELUM410_WORKER_IDLE_SECONDS','120'))
        if not math.isfinite(self.idle_seconds) or self.idle_seconds<0: raise ValueError('Worker idle time must be finite and nonnegative')
        self.runtime_root=Path(runtime_root or Path(__file__).parent/'.runtime')
        self._lock=threading.Lock(); self._proc=None; self._reader=None; self._settings=None
        self._responses=None; self._worker_dir=None
        self._idle_timer=None
        atexit.register(self.close)

    @staticmethod
    def _read(proc, responses, log_path):
        assert proc.stdout is not None
        with log_path.open('ab') as log:
            for line in iter(proc.stdout.readline,b''):
                if log.tell()<10*1024*1024: log.write(line); log.flush()
                if line.startswith(b'VELUM_JSON:'):
                    try: responses.put(json.loads(line[len(b'VELUM_JSON:'):].decode('utf-8')))
                    except Exception: responses.put({'protocol_error':'invalid JSON response'})
        code=proc.wait()
        responses.put({'worker_exited':True,'exit_code':code})

    def _start(self, settings):
        workers,device=settings
        self.runtime_root.mkdir(parents=True,exist_ok=True)
        self._worker_dir=self.runtime_root/('worker_'+uuid.uuid4().hex)
        self._worker_dir.mkdir()
        args=[sys.executable,'-I','-B','-u',str(self.worker_script),'--runtime-dir',str(self._worker_dir.resolve()),
              '--cache-dir',str((self.runtime_root/'cache410').resolve()),'--cpu-workers',str(workers)]
        if device is not None: args += ['--cuda-device',str(device)]
        env=os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                   NUMBA_NUM_THREADS=str(min(workers,os.cpu_count() or 1)),NUMBA_THREADING_LAYER='workqueue',
                   PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        env.pop('KMP_DUPLICATE_LIB_OK',None)
        responses=queue.Queue()
        with (self._worker_dir/'stderr.log').open('ab',buffering=0) as err:
            proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,env=env,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),cwd=str(self.worker_script.parent))
        self._proc=proc; self._responses=responses; self._settings=settings
        self._reader=threading.Thread(target=self._read,args=(proc,responses,self._worker_dir/'stdout.log'),daemon=True)
        self._reader.start()

    def _stop(self):
        if self._idle_timer is not None:
            self._idle_timer.cancel(); self._idle_timer=None
        proc,reader=self._proc,self._reader
        self._proc=None; self._reader=None; self._settings=None; self._responses=None
        if proc is not None:
            if proc.poll() is None:
                try: proc.kill()
                except OSError: pass
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
            if reader is not None: reader.join(timeout=5)
            for stream in (proc.stdin,proc.stdout):
                if stream is not None:
                    try: stream.close()
                    except OSError: pass

    def _schedule_idle(self, proc):
        if not self.idle_seconds: return
        def retire():
            if self._lock.acquire(blocking=False):
                try:
                    if self._proc is proc: self._stop()
                finally: self._lock.release()
        self._idle_timer=threading.Timer(self.idle_seconds,retire)
        self._idle_timer.daemon=True; self._idle_timer.start()

    def _failure(self,text,code=None):
        suffix='' if code is None else f' exit code {code} (0x{code & 0xffffffff:08X})'
        return RuntimeError(f'Velum 4.10 isolated worker: {text}{suffix}. ComfyUI host is not the worker. Logs: {self._worker_dir}')

    @staticmethod
    def _array(value,name):
        value=np.asarray(value)
        if value.dtype!=np.uint8 or value.ndim!=3 or value.shape[2]!=3 or min(value.shape)<=0:
            raise ValueError(f'{name} must be nonempty uint8 HWC3')
        return np.ascontiguousarray(value)

    def call(self,source,reference,*,deband_mode,manual_strength,backend,cpu_workers,cuda_device=None,cancel=None,structure_field='gradient_energy'):
        source=self._array(source,'source'); reference=self._array(reference,'reference')
        workers=int(cpu_workers)
        if not 1<=workers<=64: raise ValueError('cpu_workers must be 1..64')
        if structure_field not in {'gradient_energy','dog_guided','dog_raw'}: raise ValueError('Invalid structure field')
        def check():
            if cancel is not None and cancel(): raise RuntimeError('Velum request cancelled')
        while not self._lock.acquire(timeout=.1): check()
        try:
            check()
            if self._idle_timer is not None: self._idle_timer.cancel(); self._idle_timer=None
            settings=(workers,None if cuda_device is None else int(cuda_device))
            if self._proc is None or self._proc.poll() is not None or self._settings!=settings:
                self._stop(); self._start(settings)
            proc,responses=self._proc,self._responses
            assert proc is not None and proc.stdin is not None and responses is not None and self._worker_dir is not None
            started=time.perf_counter()
            with tempfile.TemporaryDirectory(prefix='request_',dir=self._worker_dir) as temp:
                directory=Path(temp).resolve()
                np.save(directory/'source.npy',source,allow_pickle=False)
                np.save(directory/'reference.npy',reference,allow_pickle=False)
                ident=uuid.uuid4().hex
                payload={'id':ident,'request_dir':str(directory),'source':str(directory/'source.npy'),
                         'reference':str(directory/'reference.npy'),'deband_mode':deband_mode,
                         'manual_strength':float(manual_strength),'backend':backend,'structure_field':structure_field}
                (self._worker_dir/'last_request.json').write_text(json.dumps(dict(payload,source_shape=list(source.shape),reference_shape=list(reference.shape),
                    source_pixel_sha256=hashlib.sha256(source).hexdigest(),reference_pixel_sha256=hashlib.sha256(reference).hexdigest()),indent=2),encoding='utf-8')
                try:
                    proc.stdin.write((json.dumps(payload)+'\n').encode()); proc.stdin.flush()
                except (OSError,ValueError) as exc:
                    code=proc.poll(); raise self._failure(f'pipe closed: {exc}',code) from exc
                deadline=time.monotonic()+self.timeout_seconds
                try:
                    while True:
                        check()
                        remaining=deadline-time.monotonic()
                        if remaining<=0: raise self._failure(f'timeout after {self.timeout_seconds:g}s')
                        try: response=responses.get(timeout=min(.1,remaining))
                        except queue.Empty: continue
                        if response.get('worker_exited'): raise self._failure('native/process exit',response['exit_code'])
                        if response.get('protocol_error'): raise self._failure(response['protocol_error'])
                        if response.get('id')!=ident: raise self._failure('response id mismatch')
                        if not response.get('ok'): raise self._failure(response.get('error','worker failed'))
                        if response.get('version')!='4.10': raise self._failure('wrong engine version')
                        output=Path(response['output']).resolve()
                        if output.parent!=directory or output.name!='output.npy': raise self._failure('unexpected output path')
                        result=self._array(np.load(output,allow_pickle=False),'output')
                        if result.shape!=source.shape: raise self._failure('output shape changed')
                        response['rpc_seconds']=time.perf_counter()-started
                        response['worker_log_dir']=str(self._worker_dir)
                        self._schedule_idle(proc)
                        return result,response
                except BaseException:
                    # In particular preserve Comfy's cancellation exception, but stop its native work first.
                    self._stop()
                    raise
        except BaseException:
            self._stop()
            raise
        finally:
            self._lock.release()

    def close(self):
        acquired=self._lock.acquire(timeout=.5)
        try: self._stop()
        finally:
            if acquired: self._lock.release()
