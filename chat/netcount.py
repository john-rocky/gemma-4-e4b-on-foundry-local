"""Count the network endpoints (TCP connections, TCP listeners, UDP endpoints) that this process and its child
processes hold, from Windows' own tables (iphlpapi GetExtendedTcpTable / GetExtendedUdpTable, IPv4 and IPv6).

No administrator rights are needed, and nothing is sent anywhere: the tables are read from the local system.

    import netcount
    netcount.count()   # {"tcp": 0, "udp": 0, "total": 0, "pids": [1234], "rows": []}

When a table or the process list cannot be read, count() returns "total": None and an "error": not measured, never 0.
"""
import ctypes
import os
import sys
from ctypes import wintypes

AF_INET, AF_INET6 = 2, 23
TCP_TABLE_OWNER_PID_ALL = 5
UDP_TABLE_OWNER_PID = 1
ERROR_INSUFFICIENT_BUFFER = 122
ERROR_BAD_LENGTH = 24   # CreateToolhelp32Snapshot can fail with it for a moment; Microsoft says to retry
TCP_STATES = {1: "CLOSED", 2: "LISTEN", 3: "SYN_SENT", 4: "SYN_RCVD", 5: "ESTABLISHED", 6: "FIN_WAIT1",
              7: "FIN_WAIT2", 8: "CLOSE_WAIT", 9: "CLOSING", 10: "LAST_ACK", 11: "TIME_WAIT", 12: "DELETE_TCB"}

# row layouts after the 4-byte entry count: (row size, offset of the owning pid, offset of the local port)
_TCP4 = (24, 20, 8)   # MIB_TCPROW_OWNER_PID: state, local addr, local port, remote addr, remote port, pid
_TCP6 = (56, 52, 20)  # MIB_TCP6ROW_OWNER_PID: local addr[16], scope, port, remote addr[16], scope, port, state, pid
_UDP4 = (12, 8, 4)    # MIB_UDPROW_OWNER_PID: local addr, local port, pid
_UDP6 = (28, 24, 20)  # MIB_UDP6ROW_OWNER_PID: local addr[16], scope, port, pid


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]


def own_pids():
    """This process and every process under it (children, grandchildren), from one process snapshot. Raises OSError
    when the snapshot cannot be read (the count is then unknown, not "this process only")."""
    me = os.getpid()
    if sys.platform != "win32":
        return [me]
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    snap, error = None, 0
    for _ in range(5):
        snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        if snap not in (None, wintypes.HANDLE(-1).value):
            break
        error = ctypes.get_last_error()
        if error != ERROR_BAD_LENGTH:
            break
    if snap in (None, wintypes.HANDLE(-1).value):
        raise OSError(error, "the process list could not be read")
    children = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        if not ok:
            raise OSError(ctypes.get_last_error(), "the process list could not be read")
        while ok:
            children.setdefault(entry.th32ParentProcessID, []).append(entry.th32ProcessID)
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    found, todo = [me], [me]
    while todo:
        for child in children.get(todo.pop(), []):
            if child not in found and child != 0:
                found.append(child)
                todo.append(child)
    return found


def _table(func, family, table_class):
    size = wintypes.DWORD(0)
    buf = None
    for _ in range(8):  # the table can grow between the size call and the read
        buf = ctypes.create_string_buffer(max(size.value, 4))
        rc = func(buf, ctypes.byref(size), False, family, table_class, 0)
        if rc == 0:
            return buf.raw
        if rc != ERROR_INSUFFICIENT_BUFFER:
            raise OSError(rc, f"{func.__name__} failed")
    raise OSError(ERROR_INSUFFICIENT_BUFFER, f"{func.__name__}: the table kept growing")


def _rows(raw, layout):
    size, pid_at, port_at = layout
    n = int.from_bytes(raw[0:4], "little")
    for i in range(n):
        base = 4 + i * size
        row = raw[base:base + size]
        pid = int.from_bytes(row[pid_at:pid_at + 4], "little")
        port = int.from_bytes(row[port_at:port_at + 2], "big")  # network byte order in the low 16 bits
        yield pid, port, row


def count(pids=None):
    """Endpoints held by `pids` (default: this process and its children): TCP rows of every state Windows lists under
    these processes (listeners and connections) and bound UDP endpoints. Windows lists a closed connection's TIME_WAIT
    row under process 0, not under the process that opened it (seen on Windows 11), so a connection opened and closed
    between two readings is not counted; traffic Windows makes for a process (a DNS lookup) belongs to other
    processes."""
    if sys.platform != "win32":
        return {"tcp": None, "udp": None, "total": None, "pids": [], "rows": [], "error": "only on Windows"}
    try:
        pids = set(pids if pids is not None else own_pids())
    except OSError as error:
        return {"tcp": None, "udp": None, "total": None, "pids": [], "rows": [], "error": str(error)}
    out = {"tcp": 0, "udp": 0, "total": 0, "pids": sorted(pids), "rows": []}
    iphlpapi = ctypes.WinDLL("iphlpapi")
    tcp, udp = iphlpapi.GetExtendedTcpTable, iphlpapi.GetExtendedUdpTable
    for f in (tcp, udp):
        f.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL, wintypes.ULONG, ctypes.c_int,
                      wintypes.ULONG]
        f.restype = wintypes.DWORD
    for proto, func, table_class, layouts in (("tcp", tcp, TCP_TABLE_OWNER_PID_ALL, ((AF_INET, _TCP4), (AF_INET6, _TCP6))),
                                              ("udp", udp, UDP_TABLE_OWNER_PID, ((AF_INET, _UDP4), (AF_INET6, _UDP6)))):
        for family, layout in layouts:
            try:
                raw = _table(func, family, table_class)
            except OSError as error:
                return {"tcp": None, "udp": None, "total": None, "pids": sorted(pids), "rows": [], "error": str(error)}
            for pid, port, row in _rows(raw, layout):
                if pid in pids:
                    out[proto] += 1
                    item = {"proto": proto, "ip": 6 if family == AF_INET6 else 4, "pid": pid, "local_port": port}
                    if proto == "tcp":
                        state_at = 0 if family == AF_INET else 48
                        item["state"] = TCP_STATES.get(int.from_bytes(row[state_at:state_at + 4], "little"), "?")
                    out["rows"].append(item)
    out["total"] = out["tcp"] + out["udp"]
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(count(), indent=1))
