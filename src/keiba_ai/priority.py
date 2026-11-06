"""Run heavy batch work (nightly update, training, experiments) behind the browser app.

The process gets a lower CPU priority and low disk priority, so pages and one-race jobs answer first while a
long job keeps the rest of the machine.  When nothing else is running the job is about as fast as before.
Windows: BELOW_NORMAL priority class (inherited by child processes) and low I/O priority; elsewhere: nice 10.
Set KEIBA_PRIORITY=normal to keep the normal priority.
"""
from __future__ import annotations

import os
import sys

_done = False


def background():
    """Lower this process's CPU and disk priority (once; failures are ignored)."""
    global _done
    if _done or os.environ.get('KEIBA_PRIORITY', '').lower() == 'normal':
        return
    _done = True
    try:
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            kernel32.SetPriorityClass(handle, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
            # ProcessIoPriority (33) = IoPriorityLow (1); the same call psutil uses.  Working-set trimming of
            # PROCESS_MODE_BACKGROUND_BEGIN is avoided on purpose: it makes large numpy work page heavily.
            value = ctypes.c_ulong(1)
            ctypes.windll.ntdll.NtSetInformationProcess(handle, 33, ctypes.byref(value), ctypes.sizeof(value))
        else:
            os.nice(10)
    except Exception:  # noqa: BLE001 - priority is best-effort
        pass

