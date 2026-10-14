"""Windows-only native destination selection and safe .cube writing."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
from ctypes import wintypes
from datetime import datetime
from pathlib import Path


_approved_paths: set[Path] = set()
_approved_state: dict[Path, tuple[bool, int, int]] = {}
_BUF_SIZE = 32768


def _prefs_path() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    if not root:
        root = str(Path.home() / "AppData" / "Local")
    return Path(root) / "Velum" / "lut_export.json"


def _fallback_directory() -> Path:
    candidates = [Path.home() / "Documents", Path.home()]
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        candidates.insert(0, Path(userprofile) / "Documents")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return Path.home()


def last_directory() -> Path:
    """Return the remembered directory, or a safe existing user directory."""
    try:
        data = json.loads(_prefs_path().read_text(encoding="utf-8"))
        value = data.get("last_directory")
        if isinstance(value, str) and "\x00" not in value:
            candidate = Path(value)
            if candidate.is_absolute() and not candidate.is_symlink() and candidate.is_dir():
                return candidate
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return _fallback_directory()


class _OPENFILENAMEW(ctypes.Structure):
    _fields_ = [
        ("lStructSize", wintypes.DWORD),
        ("hwndOwner", wintypes.HWND),
        ("hInstance", wintypes.HINSTANCE),
        ("lpstrFilter", wintypes.LPCWSTR),
        ("lpstrCustomFilter", wintypes.LPWSTR),
        ("nMaxCustFilter", wintypes.DWORD),
        ("nFilterIndex", wintypes.DWORD),
        ("lpstrFile", wintypes.LPWSTR),
        ("nMaxFile", wintypes.DWORD),
        ("lpstrFileTitle", wintypes.LPWSTR),
        ("nMaxFileTitle", wintypes.DWORD),
        ("lpstrInitialDir", wintypes.LPCWSTR),
        ("lpstrTitle", wintypes.LPCWSTR),
        ("Flags", wintypes.DWORD),
        ("nFileOffset", wintypes.WORD),
        ("nFileExtension", wintypes.WORD),
        ("lpstrDefExt", wintypes.LPCWSTR),
        ("lCustData", wintypes.LPARAM),
        ("lpfnHook", wintypes.LPVOID),
        ("lpTemplateName", wintypes.LPCWSTR),
        ("pvReserved", wintypes.LPVOID),
        ("dwReserved", wintypes.DWORD),
        ("FlagsEx", wintypes.DWORD),
    ]


def _native_dialog() -> Path | None:
    if os.name != "nt":
        raise NotImplementedError("native LUT save dialog is supported on Windows only")

    OFN_OVERWRITEPROMPT = 0x00000002
    OFN_HIDEREADONLY = 0x00000004
    OFN_NOCHANGEDIR = 0x00000008
    OFN_PATHMUSTEXIST = 0x00000800
    OFN_EXPLORER = 0x00080000
    OFN_DONTADDTORECENT = 0x02000000
    buffer = ctypes.create_unicode_buffer(_BUF_SIZE)
    name = datetime.now().strftime("Velum-%Y%m%d-%H%M%S.cube")
    buffer.value = name
    initial = str(last_directory())
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    dialog = _OPENFILENAMEW(
        ctypes.sizeof(_OPENFILENAMEW), 0, 0,
        "3D LUT (*.cube)\0*.cube\0\0", 0, 0, 1,
        ctypes.cast(buffer, wintypes.LPWSTR), _BUF_SIZE, 0, 0, initial, "Velum - Export .cube",
        OFN_OVERWRITEPROMPT | OFN_HIDEREADONLY | OFN_NOCHANGEDIR |
        OFN_PATHMUSTEXIST | OFN_EXPLORER | OFN_DONTADDTORECENT,
         0, 0, "cube", 0, 0, 0, 0, 0, 0,
    )
    dialog.hwndOwner = user32.GetForegroundWindow()
    get_save = ctypes.windll.comdlg32.GetSaveFileNameW
    get_save.argtypes = [ctypes.POINTER(_OPENFILENAMEW)]
    get_save.restype = wintypes.BOOL
    if not get_save(ctypes.byref(dialog)):
        extended_error = ctypes.windll.comdlg32.CommDlgExtendedError()
        if extended_error == 0:
            return None
        raise OSError(int(extended_error), "GetSaveFileNameW failed")
    return Path(buffer.value)


def _valid_target(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if "\x00" in str(path) or not path.is_absolute() or path.suffix.lower() != ".cube":
        raise ValueError("LUT destination must be a .cube file")
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("LUT destination cannot be a symbolic link")
    if path.exists() and (path.is_dir() or not path.is_file()):
        raise ValueError("LUT destination must be a regular file")
    if not path.parent.is_dir():
        raise ValueError("LUT destination directory does not exist")
    return path


def choose_path() -> Path | None:
    """Ask a separate Windows process for a path and approve that path."""
    if os.name != "nt":
        raise NotImplementedError("native LUT save dialog is supported on Windows only")
    command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--dialog"]
    completed = subprocess.run(command, shell=False, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "LUT save dialog failed")
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("LUT save dialog returned invalid JSON") from exc
    if result.get("cancelled") is True:
        return None
    raw = result.get("path")
    if not isinstance(raw, str):
        raise RuntimeError("LUT save dialog returned no path")
    path = _valid_target(raw).absolute()
    _approved_paths.add(path)
    try:
        stat = path.stat()
        _approved_state[path] = (True, stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        _approved_state[path] = (False, 0, 0)
    return path


def discard_path(path: Path) -> None:
    """Revoke a pending dialog capability without writing it."""
    try:
        target = Path(path).absolute()
    except (TypeError, ValueError):
        return
    _approved_paths.discard(target)
    _approved_state.pop(target, None)


def _remember(directory: Path) -> None:
    target = _prefs_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".lut_export-", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump({"last_directory": str(directory)}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_cube(path: Path, content: bytes) -> dict[str, object]:
    """Atomically write bytes to a path returned by :func:`choose_path`."""
    target = _valid_target(path)
    try:
        if target not in _approved_paths:
            raise PermissionError("LUT destination was not selected by the native dialog")
        approved = _approved_state.get(target)
        try:
            stat = target.stat()
            current = (True, stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            current = (False, 0, 0)
        if approved is not None and current != approved:
            raise RuntimeError("LUT destination changed after dialog confirmation")
        if not isinstance(content, bytes):
            raise TypeError("cube content must be bytes")
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp",
                                         dir=str(target.parent))
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            # A new destination must not replace a file that appeared after the dialog.
            if not target.exists():
                os.rename(temporary, target)
                temporary = None
            else:
                before = target.stat()
                if approved is not None and (True, before.st_mtime_ns, before.st_size) != approved:
                    raise RuntimeError("LUT destination changed before commit")
                os.replace(temporary, target)
                temporary = None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        try:
            _remember(target.parent)
        except OSError as exc:
            return {"path": str(target), "directory_remembered": False,
                    "directory_error": str(exc)}
        return {"path": str(target), "directory_remembered": True}
    finally:
        discard_path(target)


def _dialog_main() -> int:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="strict")
        selected = _native_dialog()
        payload = {"cancelled": True} if selected is None else {"path": str(selected)}
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        print(f"lut_save: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__" and "--dialog" in sys.argv[1:]:
    raise SystemExit(_dialog_main())
