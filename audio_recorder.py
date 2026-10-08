"""
audio_recorder.py -- Microphone capture module for GlideText.

Rebuilt from scratch with:
  - Callback-based streaming via sounddevice (16 kHz, mono, int16)
  - Energy-based Voice Activity Detection (VAD) for auto-stop
  - OS-level audio ducking via pycaw (with proper COM init)
  - Active-window context detection (Win32 API)
  - Robust error handling -- never crashes the host process

The microphone device index is configurable via the GUI settings panel
and persisted in config.txt.
"""

import logging
import os
import tempfile
import queue
import threading
import time
import numpy as np
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception as e:
    sd = None
    HAS_SOUNDDEVICE = False
    logging.error(f"  ⚠  Failed to import sounddevice: {e}")
try:
    import wavio
    HAS_WAVIO = True
except ImportError:
    HAS_WAVIO = False


# ---------------------------------------------------------------------------
# Optional: pycaw for audio ducking
# ---------------------------------------------------------------------------
try:
    if sys.platform == "win32":
        from pycaw.pycaw import AudioUtilities
        HAS_PYCAW = True
    else:
        HAS_PYCAW = False
except Exception:
    HAS_PYCAW = False

# ---------------------------------------------------------------------------
# Optional: win32gui for active window detection
# ---------------------------------------------------------------------------
try:
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes
        HAS_WIN32 = True
    else:
        HAS_WIN32 = False
except Exception:
    HAS_WIN32 = False


# ---------------------------------------------------------------------------
# Optional: webrtcvad for enterprise voice activity detection
# ---------------------------------------------------------------------------
try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False


# ---------------------------------------------------------------------------
# Optional: noisereduce for DSP noise suppression
# ---------------------------------------------------------------------------
try:
    import noisereduce as nr
    HAS_NOISE_REDUCE = True
    logging.info("  [Recorder] Noise suppression: ACTIVE (noisereduce loaded)")
except Exception as e:
    HAS_NOISE_REDUCE = False
    logging.warning(f"  [Recorder] Noise suppression: DISABLED (noisereduce import failed: {e})")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000       # 16 kHz -- optimal for speech models
CHANNELS = 1               # Mono
DTYPE = "int16"            # 16-bit PCM
BLOCK_SIZE = 256           # Reduced block size for low-latency streaming-style capture
DEFAULT_DEVICE_INDEX = None  # None = system default input device

# VAD configuration
VAD_ENERGY_THRESHOLD = 300      # RMS energy threshold to consider "speech"
VAD_SILENCE_DURATION = 1.8      # Seconds of silence before auto-stop
VAD_MIN_SPEECH_DURATION = 0.4   # Minimum speech duration before VAD kicks in


# ═══════════════════════════════════════════════════════════════
#  Context Awareness -- Active Window Detection
# ═══════════════════════════════════════════════════════════════

def get_active_window_info() -> dict:
    """Detect the currently focused application and window context.

    Returns a dict with keys:
        title                -- window title string (cleared if sensitive context)
        exe_name             -- executable/app basename (e.g. 'Code.exe', 'Terminal.app')
        app_hint             -- simplified privacy-safe app name / bundle ID
        app_category         -- canonical AppCategory string ('IDE/code editor', 'terminal', etc.)
        is_sensitive_context -- True if password manager / credential prompt detected
        hwnd                 -- opaque window token (Win32 HWND or macOS token)

    Returns safe empty/unknown values on failure or unsupported platforms.
    """
    import platform_compat
    return platform_compat.get_active_window_info()


# ═══════════════════════════════════════════════════════════════
#  Audio Ducker -- OS-level volume control during recording
# ═══════════════════════════════════════════════════════════════

class AudioDucker:
    """Ducks playback volume during recording to prevent bleed-through."""

    def __init__(self):
        pass

    def duck(self):
        """Duck active application or system playback volume asynchronously."""
        import platform_compat
        platform_compat.duck_audio()

    def restore(self):
        """Restore application or system playback volume asynchronously."""
        import platform_compat
        platform_compat.restore_audio()


# ═══════════════════════════════════════════════════════════════
#  Audio Recorder -- Core recording engine with VAD
# ═══════════════════════════════════════════════════════════════

class AudioRecorder:
    """Hold-to-record and auto-stop microphone capture engine.

    Modes:
        - Hold-to-record:  call start() / stop() manually.
        - Auto-stop (VAD): call start(auto_stop_callback=fn) and the
          recorder will invoke fn(audio_path) when silence is detected.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        device_index: int | None = DEFAULT_DEVICE_INDEX,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self._queue: queue.Queue = queue.Queue()
        self._vad_queue: queue.Queue = queue.Queue()
        self._stream = None
        self._is_recording = False
        self._lock = threading.Lock()
        self._ducker = AudioDucker()

        # VAD state (speech / silence activity detector)
        self._vad_enabled = False
        self._auto_stop_callback = None
        self._on_speech_callback = None
        self._on_silence_callback = None
        self._chunk_callback = None
        self._speech_detected = False
        self._is_in_silence_pause = False
        self._speech_start_time = 0.0
        self._last_speech_time = 0.0

        self._vad_thread = None
        
        self._current_rms = 0.0

        self._vad_instance = None
        self._warmed_up = False
        self._accumulated_frames = []

    def _log_device_info(self):
        """Print the selected input device name (ASCII-safe)."""
        if not HAS_SOUNDDEVICE or sd is None:
            logging.info("  [Recorder] sounddevice library is not available. Audio capture is disabled.")
            return
        try:
            if self.device_index is not None:
                dev_info = sd.query_devices(self.device_index)
            else:
                dev_info = sd.query_devices(kind="input")
            idx = self.device_index if self.device_index is not None else "default"
            logging.info(f"  [Recorder] Input device: [{idx}] {dev_info['name']}")
        except Exception as e:
            logging.info(f"  [Recorder] Could not query device {self.device_index}: {e}")

    def warmup(self) -> None:
        """Asynchronously initialize the audio device cache and the VAD engine in a low-priority thread."""
        import sys
        with self._lock:
            if getattr(self, "_warmed_up", False) or self._is_recording:
                return
            self._warmed_up = True

        def _warmup_impl():
            import platform_compat
            # Run warmup at below normal priority to not impact GUI thread
            platform_compat.set_thread_priority(-1)

            # Pre-warm PortAudio by querying device info
            if HAS_SOUNDDEVICE and sd is not None:
                try:
                    if self.device_index is not None:
                        sd.query_devices(self.device_index)
                    else:
                        sd.query_devices(kind="input")
                    self._log_device_info()
                except Exception as e:
                    logging.info(f"  [Warmup] sounddevice device query failed: {e}")

            # Pre-warm VAD engine
            if HAS_VAD and self._vad_instance is None:
                try:
                    self._vad_instance = webrtcvad.Vad(3)
                    # Feed a dummy frame to pre-warm internal C memory/buffers
                    dummy_frame = b"\x00" * 960  # 30ms of silence at 16kHz
                    self._vad_instance.is_speech(dummy_frame, 16000)
                except Exception as e:
                    logging.info(f"  [Warmup] VAD warmup failed: {e}")
            logging.info("  [Warmup] Audio pre-warming completed.")

        threading.Thread(target=_warmup_impl, daemon=True).start()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # PortAudio callback -- runs on the audio thread
    # ------------------------------------------------------------------
    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status) -> None:
        """Push audio blocks into the queue; update VAD energy tracking.

        Memory discipline:
          - Allocates a single `block_copy` (`int16`) per callback and shares
            that immutable block reference across `_queue`, `_chunk_callback`,
            and `_vad_queue` without duplicating buffers.
        """
        if status:
            logging.info(f"  [Recorder] Stream status: {status}")
        block_copy = indata.copy()
        self._queue.put(block_copy)
        if getattr(self, "_chunk_callback", None) is not None:
            try:
                self._chunk_callback(block_copy)
            except Exception:
                pass
        if self._vad_enabled and self._vad_queue.qsize() < 250:
            self._vad_queue.put(block_copy)

        # Track RMS for UI Live Waveform using float32 (half the memory of float64)
        try:
            f32 = indata.astype(np.float32, copy=False)
            self._current_rms = float(np.sqrt(np.mean(f32 * f32)))
        except Exception:
            self._current_rms = 0.0

    # ------------------------------------------------------------------
    # VAD monitor thread (speech / silence activity detector)
    # ------------------------------------------------------------------
    def _vad_monitor(self):
        """Background thread that monitors speech/silence activity states.

        CRITICAL ARCHITECTURE:
        VAD detects speech activity and silence/thinking pauses to update
        session state and UI diagnostics, but NEVER stops the audio stream
        or finalizes a continuous dictation session on silence.
        """
        import platform_compat
        # Set thread priority to HIGHEST for real-time monitoring
        platform_compat.set_thread_priority(2)

        vad = getattr(self, "_vad_instance", None)
        if vad is None and HAS_VAD:
            try:
                vad = webrtcvad.Vad(3)
                self._vad_instance = vad
            except Exception as e:
                logging.error(f"  [VAD] Failed to initialize webrtcvad Vad: {e}")

        frame_duration_ms = 30
        frame_samples = int(self.sample_rate * (frame_duration_ms / 1000.0))
        audio_buffer = np.empty(0, dtype=np.int16)

        def _eval_frame(frame_1d: np.ndarray) -> None:
            is_speech = False
            if vad:
                try:
                    is_speech = vad.is_speech(frame_1d.tobytes(), self.sample_rate)
                except Exception:
                    pass
            else:
                f32 = frame_1d.astype(np.float32, copy=False)
                rms = float(np.sqrt(np.mean(f32 * f32)))
                is_speech = rms > VAD_ENERGY_THRESHOLD
            self._process_vad_frame_state(is_speech, time.time())

        while self._is_recording and self._vad_enabled:
            processed_any = False
            while not self._vad_queue.empty():
                try:
                    block = self._vad_queue.get_nowait()
                except queue.Empty:
                    break
                processed_any = True
                flat = block.ravel()
                # Fast path: standard 30ms block with empty remainder buffer (zero concatenation)
                if len(audio_buffer) == 0 and len(flat) == frame_samples:
                    _eval_frame(flat)
                else:
                    audio_buffer = np.concatenate((audio_buffer, flat)) if len(audio_buffer) else flat
                    offset = 0
                    total_len = len(audio_buffer)
                    while total_len - offset >= frame_samples:
                        _eval_frame(audio_buffer[offset:offset + frame_samples])
                        offset += frame_samples
                    audio_buffer = (
                        audio_buffer[offset:].copy()
                        if offset < total_len
                        else np.empty(0, dtype=np.int16)
                    )
            if not processed_any:
                time.sleep(0.01)

    def _process_vad_frame_state(self, is_speech: bool, now: float) -> None:
        """Process a single VAD decision and fire speech/silence activity callbacks.

        Does NOT stop recording or finalize the session.
        """
        if is_speech:
            if not self._speech_detected or self._is_in_silence_pause:
                self._speech_detected = True
                self._is_in_silence_pause = False
                self._speech_start_time = now
                if self._on_speech_callback:
                    try:
                        self._on_speech_callback()
                    except Exception as e:
                        logging.error(f"  [VAD] Speech callback error: {e}")
            self._last_speech_time = now
        else:
            if self._speech_detected and not self._is_in_silence_pause:
                speech_duration = self._last_speech_time - self._speech_start_time
                silence_duration = now - self._last_speech_time

                if (speech_duration >= VAD_MIN_SPEECH_DURATION
                        and silence_duration >= VAD_SILENCE_DURATION):
                    self._is_in_silence_pause = True
                    logging.info("  [VAD] Silence detected -- session paused (waiting for more speech or explicit stop).")
                    self._trigger_auto_stop(silence_duration=silence_duration)

    def _trigger_auto_stop(self, silence_duration: float = VAD_SILENCE_DURATION):
        """Notify listeners that silence/pause was detected.

        NOTE: Retained for backward compatibility in naming, but its semantic
        role is now strictly a non-terminating silence-activity notification.
        It does NOT call `self.stop()` and does NOT end the recording session.
        """
        if self._on_silence_callback:
            try:
                self._on_silence_callback(silence_duration)
            except Exception as e:
                logging.error(f"  [VAD] Silence callback error: {e}")
        if self._auto_stop_callback:
            try:
                self._auto_stop_callback(None)
            except TypeError:
                try:
                    self._auto_stop_callback()
                except Exception as e:
                    logging.error(f"  [VAD] Activity callback error: {e}")
            except Exception as e:
                logging.error(f"  [VAD] Activity callback error: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(
        self,
        auto_stop_callback=None,
        on_speech_callback=None,
        on_silence_callback=None,
        chunk_callback=None,
    ) -> None:
        """Begin recording from the selected microphone.

        Args:
            auto_stop_callback: Optional legacy VAD silence-activity callback.
                                Does NOT stop the recorder; notifies on silence.
            on_speech_callback: Optional callback invoked when speech starts or resumes.
            on_silence_callback: Optional callback(silence_duration) invoked when
                                 silence pause is detected after speech.
            chunk_callback: Optional callback(np.ndarray) invoked for each captured audio block.
        """
        with self._lock:
            if self._is_recording:
                return  # Guard against double-start

            # Duck system audio
            self._ducker.duck()

            # Drain any leftover frames
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            while not self._vad_queue.empty():
                try:
                    self._vad_queue.get_nowait()
                except queue.Empty:
                    break

            # Reset VAD state
            self._accumulated_frames = []
            self._auto_stop_callback = auto_stop_callback
            self._on_speech_callback = on_speech_callback
            self._on_silence_callback = on_silence_callback
            self._chunk_callback = chunk_callback
            self._vad_enabled = any(
                cb is not None
                for cb in (auto_stop_callback, on_speech_callback, on_silence_callback)
            )
            self._speech_detected = False
            self._is_in_silence_pause = False
            self._speech_start_time = 0.0
            self._last_speech_time = time.time()

            if not HAS_SOUNDDEVICE or sd is None:
                logging.info("  [Recorder] Cannot start recording: sounddevice library is not available.")
                self._ducker.restore()
                self._is_recording = False
                self._stream = None
                return

            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=DTYPE,
                    blocksize=BLOCK_SIZE,
                    device=self.device_index,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._is_recording = True
                logging.info("  [Recorder] Recording started.")
            except Exception as e:
                logging.error(f"  [Recorder] Failed to start stream: {e}")
                self._ducker.restore()
                self._is_recording = False
                self._stream = None
                return

        # Start VAD monitor if needed
        if self._vad_enabled:
            self._vad_thread = threading.Thread(
                target=self._vad_monitor, daemon=True
            )
            self._vad_thread.start()

    # Maximum audio duration (seconds) for full-array spectral noise reduction.
    # Above this threshold, skipping full-buffer STFT prevents large float64 RAM/CPU spikes
    # (faster-whisper already applies Silero VAD and log-mel filtering).
    MAX_NOISE_REDUCE_SECONDS: float = 30.0

    def cancel(self) -> None:
        """Immediately stop recording and discard all queued audio buffers without writing to disk."""
        with self._lock:
            self._is_recording = False
            self._vad_enabled = False
            self._chunk_callback = None
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logging.error(f"  [Recorder] Stream close error on cancel: {e}")
                self._stream = None
            self._ducker.restore()
            self._accumulated_frames = []

        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        while not self._vad_queue.empty():
            try:
                self._vad_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> str | None:
        """Stop recording and return the path to the saved .wav file.

        Returns:
            Path to the .wav file, or None if nothing was recorded or disk write failed.
            Guaranteed to release internal recorder queues so raw audio is never
            retained indefinitely on the recorder instance.
        """
        with self._lock:
            if not self._is_recording:
                return None
            self._is_recording = False
            self._vad_enabled = False
            self._chunk_callback = None
            self._accumulated_frames = []

            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logging.error(f"  [Recorder] Stream close error: {e}")
                self._stream = None

            # Restore system audio
            self._ducker.restore()

        # Drain VAD queue immediately so it holds no references
        while not self._vad_queue.empty():
            try:
                self._vad_queue.get_nowait()
            except queue.Empty:
                break

        # Stitch all queued frames and immediately release the frame list
        frames: list[np.ndarray] = []
        while not self._queue.empty():
            try:
                frames.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if not frames:
            logging.info("  [Recorder] No frames captured.")
            return None

        audio_data = np.concatenate(frames, axis=0)
        frames.clear()

        # Discard very short recordings (accidental taps, < 0.3s)
        min_samples = int(self.sample_rate * 0.3)
        if len(audio_data) < min_samples:
            logging.info("  [Recorder] Recording too short -- discarded.")
            return None

        duration = len(audio_data) / float(self.sample_rate)

        # Apply DSP noise suppression only on short/medium clips to prevent
        # multi-hundred-MB float64 STFT RAM & CPU spikes on multi-minute sessions
        if HAS_NOISE_REDUCE and duration <= self.MAX_NOISE_REDUCE_SECONDS:
            logging.info("  [Recorder] Noise suppression is active. Applying reduction...")
            try:
                flat_audio = audio_data.ravel()
                reduced = nr.reduce_noise(y=flat_audio, sr=self.sample_rate)
                clipped = np.clip(reduced, -32768.0, 32767.0)
                audio_data = clipped.astype(np.int16).reshape(-1, 1)
                del reduced, clipped, flat_audio
            except Exception as e:
                logging.error(f"  [Recorder] Noise suppression processing failed: {e}")
        elif HAS_NOISE_REDUCE:
            logging.info(
                f"  [Recorder] Long session ({duration:.1f}s > {self.MAX_NOISE_REDUCE_SECONDS}s) "
                "-- bypassing full-buffer STFT noise reduction to preserve low RAM/CPU."
            )
        else:
            logging.info("  [Recorder] Noise suppression is disabled.")

        # Write to a temp .wav file with a unique name to avoid races in continuous mode
        import uuid
        import wave

        try:
            tmp_dir = os.path.join(tempfile.gettempdir(), "glidetext")
            os.makedirs(tmp_dir, exist_ok=True)
            filepath = os.path.join(tmp_dir, f"rec_{uuid.uuid4().hex}.wav")

            if HAS_WAVIO and wavio is not None:
                wavio.write(filepath, audio_data, self.sample_rate, sampwidth=2)
            else:
                with wave.open(filepath, "wb") as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(np.asarray(audio_data, dtype=np.int16).tobytes())

            logging.info(f"  [Recorder] Saved {duration:.1f}s recording to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"  [Recorder] Failed to write WAV (disk error): {e}")
            return None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def current_rms(self) -> float:
        return self._current_rms

    def get_accumulated_audio(self) -> np.ndarray | None:
        """Get a concatenated view of the audio data currently queued in the active session."""
        with self._lock:
            if getattr(self, "_accumulated_frames", None):
                try:
                    return np.concatenate(self._accumulated_frames, axis=0)
                except Exception:
                    return None
            with self._queue.mutex:
                queued = list(self._queue.queue)
            if not queued:
                return None
            try:
                return np.concatenate(queued, axis=0)
            except Exception as e:
                logging.error(f"  [Recorder] Failed to concatenate queued frames: {e}")
                return None
