import os
import traceback
from .log import logger


if os.name == "nt":
    _USE_POSIX = False
else:
    _USE_POSIX = True


if _USE_POSIX:
    import posix_ipc

    class Semaphore(object):
        def __init__(self, name: str, create: bool = False):
            self._name = name
            self._create = create
            self._closed = False
            self._unlinked = False
            
            self._sem = None
            if create:
                try:
                    self._sem = posix_ipc.Semaphore(
                        name, posix_ipc.O_CREX | posix_ipc.O_EXCL, initial_value=1)
                except posix_ipc.ExistentialError:
                    logger.info("Semaphore with name {} already exists".format(name))
                    logger.error(traceback.format_exc())
                    logger.info("Unlink and recreate it")
                    sem = posix_ipc.Semaphore(name); sem.unlink(); del sem
                    self._sem = posix_ipc.Semaphore(
                        name, posix_ipc.O_CREX | posix_ipc.O_EXCL, initial_value=1)
            else:
                self._sem = posix_ipc.Semaphore(name)
        
        @property
        def name(self):
            return self._name
            
        def acquire(self):
            return self._sem.acquire()
        
        def release(self):
            return self._sem.release()
        
        def __enter__(self):
            self._sem.__enter__()
            return self

        def __exit__(self):
            self._sem.__exit__()
            return self
        
        def close(self):
            if (not self._closed) and (self._sem is not None):
                self._sem.close()
                self._closed = True
        
        def unlink(self):
            if (not self._unlinked) and (self._sem is not None):
                self._sem.unlink()
                self._unlinked = True
        
        def __del__(self):
            if self._create:
                self.unlink()
            else:
                self.close()

else:
    import semaphore_win_ctypes

    class Semaphore(object):
        def __init__(self, name: str, create: bool = False):
            self._create = create
            self._closed = False
            self._unlinked = False

            self._sem = semaphore_win_ctypes.Semaphore(name)
            if create:
                self._sem.create(maximum_count=1, initial_count=1)
            else:
                self._sem.open()
        
        def acquire(self):
            self._sem.acquire()
        
        def release(self):
            self._sem.release()
        
        def __enter__(self):
            self._sem.acquire()
        
        def __exit__(self):
            self._sem.release()
        
        def close(self):
            if not self._closed:
                self._sem.close()
                self._closed = True
        
        def unlink(self):
            self.close()
        
        def __del__(self):
            self.close()

