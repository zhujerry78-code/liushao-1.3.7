"""Dependency-free, conservative Windows process ownership inspection."""
from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class ServiceState:
    state: str
    reason: str

    def __bool__(self):
        if self.state == 'unknown':
            raise RuntimeError(self.reason)
        return self.state == 'running'


def windows_processes():
    """Yield Python process name, image and creation time; None means access denied."""
    if os.name != 'nt':
        raise OSError('需要 Windows 进程核查接口')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    class Entry(ctypes.Structure):
        _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
                    ('th32ProcessID', wintypes.DWORD), ('th32DefaultHeapID', ctypes.c_size_t),
                    ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD),
                    ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', wintypes.LONG),
                    ('dwFlags', wintypes.DWORD), ('szExeFile', wintypes.WCHAR * 260)]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                               wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(entry)
        ok = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        while ok:
            if entry.szExeFile.casefold() in ('python.exe', 'pythonw.exe'):
                pid = int(entry.th32ProcessID)
                handle = kernel.OpenProcess(0x1000, False, pid)
                image, created = None, None
                if handle:
                    try:
                        size = wintypes.DWORD(32768)
                        text = ctypes.create_unicode_buffer(size.value)
                        times = [wintypes.FILETIME() for _ in range(4)]
                        if (kernel.QueryFullProcessImageNameW(handle, 0, text, ctypes.byref(size))
                                and kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times])):
                            image = text.value
                            created = ((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime) / 1e7 - 11644473600
                    finally:
                        kernel.CloseHandle(handle)
                yield pid, image, created
            ok = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)


def native_service_state(root):
    root = Path(root).resolve()
    target = str(root / '.venv/Scripts/python.exe').casefold()
    session_path = root / 'demo-state/desktop-session.json'
    state = None
    if session_path.exists():
        try:
            state = json.loads(session_path.read_bytes())
            if (Path(state['root']).resolve() != root or type(state['server_pid']) is not int
                    or not isinstance(state['server_created'], (int, float))):
                raise ValueError('session identity')
        except (OSError, ValueError, KeyError, TypeError):
            return ServiceState('unknown', f'运行记录无法核实：{session_path}。旧环境已保留；确认本目录程序已关闭后，将此文件移到交付目录外备份，再重试。无法确认时请运行环境自检。')
    uncertain = False
    try:
        for pid, image, created in windows_processes():
            if image is None:
                uncertain = True
                continue
            if str(Path(image).resolve()).casefold() == target:
                return ServiceState('running', '本目录虚拟环境仍有 Python 进程，请先关闭。')
            if state and pid == state['server_pid'] and abs(created - state['server_created']) < .01:
                if Path(image).resolve().is_relative_to(root):
                    return ServiceState('running', '本目录服务进程仍在运行，请先关闭。')
                uncertain = True
    except OSError as error:
        return ServiceState('unknown', f'无法核实进程：{error}。旧环境已保留；请关闭本目录程序后重试。')
    if uncertain:
        return ServiceState('unknown', '部分 Python 进程权限不足，旧环境已保留。请关闭本目录程序后重试或运行环境自检。')
    return ServiceState('stopped', 'Windows 进程核查确认本目录环境未运行。')
