"""Process setup that must run before anything else is imported. Every entry point imports this module first.

- ONNX Runtime and ONNX Runtime GenAI send usage telemetry unless ORT_DISABLE_TELEMETRY=1 is set before they start
  (docs/Privacy.md of both projects). This chat sets it here, which turns their usage telemetry off.
- Windows slows down a process it thinks runs in the background (power throttling, "EcoQoS"). Opting out keeps
  token generation at full speed when the window is not in front.
"""
import os
import sys

os.environ["ORT_DISABLE_TELEMETRY"] = "1"

POWER_THROTTLING = "not set"


def _high_qos():
    """Opt this process out of execution-speed power throttling (SetProcessInformation, ProcessPowerThrottling)."""
    global POWER_THROTTLING
    if sys.platform != "win32":
        POWER_THROTTLING = "not Windows"
        return
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    # version 1, PROCESS_POWER_THROTTLING_EXECUTION_SPEED in the control mask, cleared in the state mask
    state = PowerThrottlingState(1, 0x1, 0x0)
    ok = kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4,  # ProcessPowerThrottling
                                        ctypes.byref(state), ctypes.sizeof(state))
    POWER_THROTTLING = "off" if ok else f"could not set (error {ctypes.get_last_error()})"


_high_qos()
