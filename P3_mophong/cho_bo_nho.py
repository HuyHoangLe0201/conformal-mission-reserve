"""Block until the free virtual memory (available commit charge) of this Windows machine reaches a given size, so that
a run started from a shell script waits instead of failing on a shared machine; prints the free memory when it starts.

Run: python -I cho_bo_nho.py <GB> [label]
"""
import ctypes
import os
import sys
import time


def free_virtual_gb():
    """Available commit charge (free virtual memory) in GB on Windows; None elsewhere."""
    if os.name != "nt":
        return None

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                   [(k, ctypes.c_ulonglong) for k in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.apf / 1e9


if __name__ == "__main__":
    need, label = float(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else "")
    free, t0, said = free_virtual_gb(), time.time(), -1
    while free is not None and free < need:
        k = int((time.time() - t0) // 1800)
        if k > said:
            print(f"{label}: waiting for {need:.1f} GB of free virtual memory (now {free:.1f} GB), "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)
            said = k
        time.sleep(60)
        free = free_virtual_gb()
    print(f"{label}: {free if free is None else round(free, 1)} GB free virtual memory, "
          f"{time.strftime('%d/%m %H:%M')}", flush=True)
