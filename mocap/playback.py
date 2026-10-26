"""Frame selection from elapsed media time, independent of timer jitter."""
import math
import sys


class WindowsTimerPolicy:
    """Honor this process's timer request during playback, restoring its policy."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        class State(ctypes.Structure):
            _fields_ = [("Version", wintypes.DWORD), ("ControlMask", wintypes.DWORD),
                        ("StateMask", wintypes.DWORD)]
        self.State = State
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        for name in ("GetProcessInformation", "SetProcessInformation"):
            function = getattr(self.kernel, name)
            function.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            function.restype = wintypes.BOOL
        self.saved = None

    def acquire(self):
        state = self.State(1, 0, 0)
        handle = self.kernel.GetCurrentProcess()
        if self.saved is None and self.kernel.GetProcessInformation(handle, 4, self.ctypes.byref(state), self.ctypes.sizeof(state)):
            changed = self.State(1, state.ControlMask | 4, state.StateMask & ~4)
            if self.kernel.SetProcessInformation(handle, 4, self.ctypes.byref(changed), self.ctypes.sizeof(changed)):
                self.saved = state

    def release(self):
        if self.saved is not None:
            self.kernel.SetProcessInformation(self.kernel.GetCurrentProcess(), 4,
                self.ctypes.byref(self.saved), self.ctypes.sizeof(self.saved))
            self.saved = None


class PlaybackTimerResolution:
    """Request Windows media timer precision only while playback is active.

    Every successful timeBeginPeriod call is paired with timeEndPeriod.
    No polling or busy waiting is used.
    """
    def __init__(self, backend=None, policy=None):
        self.backend = backend
        self.policy = policy
        self.active = False
        if self.backend is None and sys.platform == "win32":
            import ctypes
            try:
                self.policy = WindowsTimerPolicy()
            except (OSError, AttributeError):
                self.policy = None  # Older Windows can still use timeBeginPeriod.
            self.backend = ctypes.WinDLL("winmm")
            for name in ("timeBeginPeriod", "timeEndPeriod"):
                function = getattr(self.backend, name)
                function.argtypes = [ctypes.c_uint]
                function.restype = ctypes.c_uint

    def acquire(self):
        if self.backend is not None and not self.active:
            if self.policy:
                self.policy.acquire()
            self.active = self.backend.timeBeginPeriod(1) == 0
            if not self.active and self.policy:
                self.policy.release()

    def release(self):
        if self.active:
            self.backend.timeEndPeriod(1)
            self.active = False
            if self.policy:
                self.policy.release()


def playback_frame(anchor, elapsed_seconds, fps, frame_count, speed=1.0):
    if fps <= 0 or frame_count <= 0 or speed <= 0:
        raise ValueError("Playback requires a positive frame rate, length and speed")
    return (anchor + math.floor(max(0, elapsed_seconds) * fps * speed + 1e-8)) % frame_count
