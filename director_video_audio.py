"""Optional audio handling for Director video references."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _discover_ffprobe() -> str | None:
    """Find ffprobe beside the FFmpeg selected by VideoHelperSuite, if available."""
    try:
        from videohelpersuite.utils import ffmpeg_path
    except Exception:
        ffmpeg_path = None

    if ffmpeg_path:
        ffmpeg = Path(str(ffmpeg_path))
        suffix = ffmpeg.suffix or (".exe" if shutil.which("ffmpeg") else "")
        candidate = ffmpeg.with_name(f"ffprobe{suffix}")
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffprobe")


def probe_video_audio_stream(
    path: str,
    *,
    ffprobe_executable: str | None = None,
    timeout_seconds: float = 15.0,
) -> bool | None:
    """Return whether a file has an audio stream, or ``None`` if probing is unavailable."""
    executable = ffprobe_executable or _discover_ffprobe()
    if not executable:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(0.1, float(timeout_seconds)),
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _is_missing_audio_stream_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "output file does not contain any stream",
            "does not contain any audio stream",
            "matches no streams",
        )
    )


def resolve_optional_video_audio(audio: Any, path: str) -> tuple[Any, str]:
    """Materialize a VHS lazy audio output, treating a genuinely absent track as normal."""
    if audio is None:
        return None, "absent"
    if not isinstance(audio, Mapping):
        return audio, "unvalidated_output"
    if isinstance(audio, dict):
        return audio, "present"

    has_audio = probe_video_audio_stream(path)
    if has_audio is False:
        return None, "absent"
    try:
        return dict(audio), "present"
    except Exception as exc:
        if _is_missing_audio_stream_error(exc):
            return None, "absent"
        raise RuntimeError(
            f"Could not materialize the audio stream from video reference {path!r}: {exc}"
        ) from exc
