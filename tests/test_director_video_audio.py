from __future__ import annotations

import unittest
from collections.abc import Mapping
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import director_video_audio as video_audio


class _LazyAudio(Mapping):
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.iterated = False

    def __iter__(self):
        self.iterated = True
        if self.error:
            raise self.error
        return iter(self.payload)

    def __len__(self):
        if self.error:
            raise self.error
        return len(self.payload)

    def __getitem__(self, key):
        if self.error:
            raise self.error
        return self.payload[key]


class DirectorVideoAudioTests(unittest.TestCase):
    def test_probe_reports_absent_audio_stream(self):
        result = type("Result", (), {"returncode": 0, "stdout": ""})()
        with patch.object(video_audio.subprocess, "run", return_value=result):
            self.assertIs(video_audio.probe_video_audio_stream("silent.mp4", ffprobe_executable="ffprobe"), False)

    def test_probe_reports_present_audio_stream(self):
        result = type("Result", (), {"returncode": 0, "stdout": "1\n"})()
        with patch.object(video_audio.subprocess, "run", return_value=result):
            self.assertIs(video_audio.probe_video_audio_stream("sound.mp4", ffprobe_executable="ffprobe"), True)

    def test_no_audio_stream_drops_lazy_mapping_without_iterating_it(self):
        lazy_audio = _LazyAudio(error=RuntimeError("FFmpeg: Output file does not contain any stream"))
        with patch.object(video_audio, "probe_video_audio_stream", return_value=False):
            resolved, status = video_audio.resolve_optional_video_audio(lazy_audio, "silent.mp4")
        self.assertIsNone(resolved)
        self.assertEqual(status, "absent")
        self.assertFalse(lazy_audio.iterated)

    def test_audio_stream_materializes_lazy_mapping_before_return(self):
        expected = {"waveform": object(), "sample_rate": 44100}
        lazy_audio = _LazyAudio(payload=expected)
        with patch.object(video_audio, "probe_video_audio_stream", return_value=True):
            resolved, status = video_audio.resolve_optional_video_audio(lazy_audio, "sound.mp4")
        self.assertEqual(status, "present")
        self.assertIsInstance(resolved, dict)
        self.assertIs(resolved["waveform"], expected["waveform"])
        self.assertTrue(lazy_audio.iterated)

    def test_known_missing_stream_error_is_graceful_when_probe_unavailable(self):
        lazy_audio = _LazyAudio(error=RuntimeError("Output file does not contain any stream"))
        with patch.object(video_audio, "probe_video_audio_stream", return_value=None):
            resolved, status = video_audio.resolve_optional_video_audio(lazy_audio, "silent.mp4")
        self.assertIsNone(resolved)
        self.assertEqual(status, "absent")

    def test_unrelated_audio_decode_errors_still_raise(self):
        lazy_audio = _LazyAudio(error=RuntimeError("decoder crashed while reading audio"))
        with patch.object(video_audio, "probe_video_audio_stream", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Could not materialize"):
                video_audio.resolve_optional_video_audio(lazy_audio, "broken-audio.mp4")


if __name__ == "__main__":
    unittest.main()
