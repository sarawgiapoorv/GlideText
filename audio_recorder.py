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
    from pycaw.pycaw import AudioUtilities
    HAS_PYCAW = True
except ImportError:
    HAS_PYCAW = False

# ---------------------------------------------------------------------------
# Optional: win32gui for active window detection
# ---------------------------------------------------------------------------
try:
    import ctypes
    import ctypes.wintypes
    HAS_WIN32 = True
except ImportError:
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
    """Detect the currently focused Windows application.

    Returns a dict with keys:
        title    -- window title string
        exe_name -- executable basename (e.g. 'Code.exe', 'chrome.exe')
        app_hint -- simplified app name for context prompts

    Returns empty-string values on failure or non-Windows platforms.
    """
    result = {"title": "", "exe_name": "", "app_hint": ""}
    if not HAS_WIN32:
        return result

    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        result["hwnd"] = hwnd

        # Window title
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        result["title"] = buf.value

        # Process executable
        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        h_process = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
        )
        if h_process:
            exe_buf = ctypes.create_unicode_buffer(512)
            size = ctypes.wintypes.DWORD(512)
            kernel32.QueryFullProcessImageNameW(
                h_process, 0, exe_buf, ctypes.byref(size)
            )
            kernel32.CloseHandle(h_process)
            exe_path = exe_buf.value
            exe_name = os.path.basename(exe_path) if exe_path else ""
            result["exe_name"] = exe_name

            # Simplify to app hint
            app_map = {
                "Code.exe": "VS Code",
                "code.exe": "VS Code",
                "chrome.exe": "Chrome Browser",
                "msedge.exe": "Edge Browser",
                "firefox.exe": "Firefox Browser",
                "slack.exe": "Slack",
                "Teams.exe": "Microsoft Teams",
                "OUTLOOK.EXE": "Outlook",
                "WINWORD.EXE": "Microsoft Word",
                "EXCEL.EXE": "Microsoft Excel",
                "POWERPNT.EXE": "Microsoft PowerPoint",
                "notepad.exe": "Notepad",
                "WindowsTerminal.exe": "Windows Terminal",
                "Discord.exe": "Discord",
                "Telegram.exe": "Telegram",
                "WhatsApp.exe": "WhatsApp",
            }
            result["app_hint"] = app_map.get(exe_name, exe_name.replace(".exe", ""))
    except Exception:
        pass  # Never crash on context detection failure

    return result


# ═══════════════════════════════════════════════════════════════
#  Audio Ducker -- OS-level volume control during recording
# ═══════════════════════════════════════════════════════════════

class AudioDucker:
    """Ducks system master volume during recording to prevent bleed-through."""

    def __init__(self):
        self._lock = threading.Lock()
        self._is_ducked = False
        self._original_volumes = {}

    def duck(self):
        """Duck active application volumes to 10% asynchronously."""
        if not HAS_PYCAW:
            return
        threading.Thread(target=self._duck_impl, daemon=True).start()

    def _duck_impl(self):
        with self._lock:
            if self._is_ducked:
                return
            import comtypes
            try:
                comtypes.CoInitialize()
                try:
                    sessions = AudioUtilities.GetAllSessions()
                    self._original_volumes.clear()
                    
                    for session in sessions:
                        volume = session.SimpleAudioVolume
                        if session.Process:
                            proc_id = session.Process.pid
                            vol = volume.GetMasterVolume()
                            self._original_volumes[proc_id] = vol
                            
                            # Duck volume to 10%
                            ducked_vol = max(0.0, vol * 0.10)
                            volume.SetMasterVolume(ducked_vol, None)
                    
                    self._is_ducked = True
                except Exception as e:
                    logging.info(f"  [AudioDucker] Duck failed: {e}")
            finally:
                try:
                    comtypes.CoUninitialize()
                except Exception:
                    pass

    def restore(self):
        """Restore application volumes asynchronously."""
        if not HAS_PYCAW:
            return
        threading.Thread(target=self._restore_impl, daemon=True).start()

    def _restore_impl(self):
        with self._lock:
            if not self._is_ducked:
                return
            import comtypes
            try:
                comtypes.CoInitialize()
                try:
                    sessions = AudioUtilities.GetAllSessions()
                    
                    for session in sessions:
                        volume = session.SimpleAudioVolume
                        if session.Process:
                            proc_id = session.Process.pid
                            if proc_id in self._original_volumes:
                                volume.SetMasterVolume(self._original_volumes[proc_id], None)
                    
                    self._original_volumes.clear()
                    self._is_ducked = False
                except Exception as e:
                    logging.info(f"  [AudioDucker] Restore failed: {e}")
            finally:
                try:
                    comtypes.CoUninitialize()
                except Exception:
                    pass


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

        # VAD state
        self._vad_enabled = False
        self._auto_stop_callback = None
        self._speech_detected = False
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
            if sys.platform == "win32":
                try:
                    import ctypes
                    # Run warmup at below normal priority to not impact GUI thread
                    ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), -1)
                except Exception:
                    pass

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
    # PortAudio callback -- runs on the audio thread
    # ------------------------------------------------------------------
    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status) -> None:
        """Push audio blocks into the queue; update VAD energy tracking."""
        if status:
            logging.info(f"  [Recorder] Stream status: {status}")
        self._queue.put(indata.copy())
        if hasattr(self, "_accumulated_frames"):
            self._accumulated_frames.append(indata.copy())
        if self._vad_enabled:
            self._vad_queue.put(indata.copy())
            
        # Track RMS for UI Live Waveform
        try:
            self._current_rms = np.sqrt(np.mean(indata.astype(np.float64) ** 2))
        except Exception:
            self._current_rms = 0.0

    # ------------------------------------------------------------------
    # VAD monitor thread
    # ------------------------------------------------------------------
    def _vad_monitor(self):
        """Background thread that watches energy levels for silence detection."""
        import sys
        if sys.platform == "win32":
            try:
                import ctypes
                # Set thread priority to HIGHEST for real-time monitoring
                ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), 2)
            except Exception as e:
                logging.error(f"  [VAD] Failed to set thread priority: {e}")

        vad = getattr(self, "_vad_instance", None)
        if vad is None and HAS_VAD:
            try:
                vad = webrtcvad.Vad(3)
                self._vad_instance = vad
            except Exception as e:
                logging.error(f"  [VAD] Failed to initialize webrtcvad Vad: {e}")

        frame_duration_ms = 30
        frame_samples = int(self.sample_rate * (frame_duration_ms / 1000.0))
        audio_buffer = np.array([], dtype=np.int16)

        while self._is_recording and self._vad_enabled:
            # Drain VAD queue
            blocks = []
            while not self._vad_queue.empty():
                try:
                    blocks.append(self._vad_queue.get_nowait())
                except queue.Empty:
                    break
            
            if blocks:
                # np.concatenate with 2D blocks requires flattening if it was 2D (it is mono but shape is (frames, 1))
                blocks_flat = [b.flatten() for b in blocks]
                audio_buffer = np.concatenate([audio_buffer] + blocks_flat)
            
            # Process complete frames
            while len(audio_buffer) >= frame_samples:
                frame = audio_buffer[:frame_samples]
                audio_buffer = audio_buffer[frame_samples:]
                
                is_speech = False
                if vad:
                    try:
                        # webrtcvad requires bytes
                        is_speech = vad.is_speech(frame.tobytes(), self.sample_rate)
                    except Exception:
                        pass
                else:
                    # Fallback RMS energy VAD
                    rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
                    is_speech = rms > VAD_ENERGY_THRESHOLD

                now = time.time()
                
                if is_speech:
                    if not self._speech_detected:
                        self._speech_detected = True
                        self._speech_start_time = now
                    self._last_speech_time = now
                else:
                    if self._speech_detected:
                        speech_duration = self._last_speech_time - self._speech_start_time
                        silence_duration = now - self._last_speech_time

                        if (speech_duration >= VAD_MIN_SPEECH_DURATION
                                and silence_duration >= VAD_SILENCE_DURATION):
                            logging.info("  [VAD] Silence detected -- auto-stopping.")
                            self._trigger_auto_stop()
                            return
            time.sleep(0.01)

    def _trigger_auto_stop(self):
        """Stop recording and invoke the auto-stop callback."""
        audio_path = self.stop()
        if audio_path and self._auto_stop_callback:
            try:
                self._auto_stop_callback(audio_path)
            except Exception as e:
                logging.error(f"  [VAD] Auto-stop callback error: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self, auto_stop_callback=None) -> None:
        """Begin recording from the selected microphone.

        Args:
            auto_stop_callback: If provided, enables VAD. Called with
                                the audio file path when silence is detected.
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
            self._vad_enabled = auto_stop_callback is not None
            self._auto_stop_callback = auto_stop_callback
            self._speech_detected = False
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

    def stop(self) -> str | None:
        """Stop recording and return the path to the saved .wav file.

        Returns:
            Path to the .wav file, or None if nothing was recorded.
        """
        with self._lock:
            if not self._is_recording:
                return None
            self._is_recording = False
            self._vad_enabled = False

            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logging.error(f"  [Recorder] Stream close error: {e}")
                self._stream = None

            # Restore system audio
            self._ducker.restore()

        # Stitch all queued frames
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

        # Discard very short recordings (accidental taps, < 0.3s)
        min_samples = int(self.sample_rate * 0.3)
        if len(audio_data) < min_samples:
            logging.info("  [Recorder] Recording too short -- discarded.")
            return None

        # Apply DSP noise suppression
        if HAS_NOISE_REDUCE:
            logging.info("  [Recorder] Noise suppression is active. Applying reduction...")
            try:
                # noisereduce expects flat array for mono
                flat_audio = audio_data.flatten()
                reduced = nr.reduce_noise(y=flat_audio, sr=self.sample_rate)
                # Clip to prevent overflow/distortion when casting float64 -> int16
                clipped = np.clip(reduced, -32768.0, 32767.0)
                audio_data = clipped.astype(np.int16).reshape(-1, 1)
            except Exception as e:
                logging.error(f"  [Recorder] Noise suppression processing failed: {e}")
        else:
            logging.info("  [Recorder] Noise suppression is disabled.")

        # Check if wavio is available before trying to write
        if not HAS_WAVIO:
            logging.info("  [Recorder] wavio library is not available. Cannot write WAV.")
            return None

        # Write to a temp .wav file with a unique name to avoid races in continuous mode
        import uuid
        tmp_dir = os.path.join(tempfile.gettempdir(), "glidetext")
        os.makedirs(tmp_dir, exist_ok=True)
        filepath = os.path.join(tmp_dir, f"rec_{uuid.uuid4().hex}.wav")

        try:
            wavio.write(filepath, audio_data, self.sample_rate, sampwidth=2)
            duration = len(audio_data) / self.sample_rate
            logging.info(f"  [Recorder] Saved {duration:.1f}s recording to {filepath}")
        except Exception as e:
            logging.error(f"  [Recorder] Failed to write WAV: {e}")
            return None

        return filepath

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def current_rms(self) -> float:
        return self._current_rms

    def get_accumulated_audio(self) -> np.ndarray | None:
        """Get a copy of the audio data accumulated so far in the session."""
        with self._lock:
            if not hasattr(self, "_accumulated_frames") or not self._accumulated_frames:
                return None
            try:
                return np.concatenate(self._accumulated_frames, axis=0)
            except Exception as e:
                logging.error(f"  [Recorder] Failed to concatenate accumulated frames: {e}")
                return None
