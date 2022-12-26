import multiprocessing
import multiprocessing.pool


class NoDaemonProcess(multiprocessing.Process):
    """A process that can spawn other processes"""

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(multiprocessing.pool.Pool):
    """A multiprocessing pool whose processes can spawn other processes"""

    Process = NoDaemonProcess

    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc
