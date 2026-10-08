# GlideText 🎙️

An open, privacy-first, locally transcribed voice dictation tool for Windows & macOS — built as an offline-capable, developer-friendly alternative to Wispr Flow (formerly named **LocalFlow**). Hold a hotkey, speak naturally, and polished, context-aware text is typed directly into whichever application is currently focused.

---

## 🧠 LLM AI Brain: FreeLLMAPI & 7.4 Billion Free Tokens

GlideText uses an external open-source tool, **FreeLLMAPI**, as its primary AI brain and intelligent LLM routing layer.

* **7.4 Billion Free Tokens Access:**  
  By integrating with the [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) tool, GlideText routes text polishing through pooled free-tier quotas and community endpoints spanning multiple leading AI providers (including Gemini, Groq, Cerebras, OpenRouter, Mistral, Together, Cohere, and HuggingFace). Collectively, these pooled free quotas offer access to **over 7.4 billion free tokens**, allowing you to enjoy high-quality AI text polishing and grammar correction without paying for monthly OpenAI, Anthropic, or proprietary dictation SaaS subscriptions.
* **Single Local Endpoint:**  
  FreeLLMAPI runs as a lightweight local server (by default on `http://127.0.0.1:3001`), exposing an OpenAI-compatible `/v1/chat/completions` endpoint. GlideText sends raw transcripts locally to FreeLLMAPI, which dynamically selects healthy upstream models and automatically retries if a specific free model or provider rate-limits.
* **Zero Audio Upload to Cloud:**  
  FreeLLMAPI *only* receives the local text transcript. Your raw microphone audio is never uploaded to FreeLLMAPI, Gemini, or any cloud server.

---

## ⚡ Multi-Tier AI Architecture & Provider Priority

GlideText uses a strictly sequential, non-blocking 4-tier fallback pipeline. You will never lose spoken words because an API went down:

```
[ Microphone Audio ]
         │
         ▼
[ Local Speech-to-Text ] ── (faster-whisper on-device)
         │
         ▼
[ Raw Spoken Transcript ]
         │
         ├──► Tier 1: FreeLLMAPI (7.4B free tokens pool @ 127.0.0.1:3001)
         │       │ (if offline / rate-limited / error)
         │       ▼
         ├──► Tier 2: Google Gemini API (Direct Cloud API key fallback)
         │       │ (if no key / rate-limited / network down)
         │       ▼
         ├──► Tier 3: Local LLM (Ollama, e.g. llama3.2:3b, 100% offline)
         │       │ (if Ollama offline / model unavailable)
         │       ▼
         └──► Tier 4: Canonical Raw Transcript (Basic punctuation, zero AI loss)
                 │
                 ▼
     [ Safe Text Injection into Active App ]
```

1. **Priority 1 — FreeLLMAPI (`http://127.0.0.1:3001`)**  
   Primary AI brain utilizing pooled free-tier token routing.
2. **Priority 2 — Google Gemini API**  
   Direct cloud fallback using your own Google AI Studio key (`gemini-2.5-flash` / `gemini-1.5-flash`), stored securely in Windows Credential Manager or macOS Keychain via `keyring`.
3. **Priority 3 — Local LLM via Ollama (`http://127.0.0.1:11434`)**  
   Completely offline, on-device text cleanup (e.g., `llama3.2:3b`, `qwen2.5:3b`, or `mistral`).
4. **Degraded Tier — Raw Fallback**  
   If every AI provider fails or you have no network connection and no local LLM, GlideText capitalizes and lightly punctuates your raw Whisper transcript and types it immediately. You never speak into a void.

---

## 🛑 Brutal Honesty: Realities, Limitations & Trade-Offs

We believe in engineering transparency rather than marketing hype. Here is what you must understand before using GlideText:

### 1. Latency Is Real (It is Not Zero-Latency Streaming)
- Unlike commercial SaaS tools that run multi-million dollar cloud GPU streaming clusters, GlideText captures audio, finalizes the segment upon hotkey release, and then transcribes and polishes it.
- **ASR Latency (faster-whisper):** On a modern CPU with the default `base` model, transcription takes **0.4s to 2.0s** depending on utterance length. On older 4-core CPUs or laptops running on battery saver, this can take up to 3.5s. Running on an NVIDIA GPU via CUDA drastically cuts this to under 250ms, but requires manual CUDA/cuDNN configuration.
- **Polishing Latency:** FreeLLMAPI adds a network round-trip of **0.8s to 3.0s** depending on which free upstream provider is serving the request. If FreeLLMAPI fails and triggers a fallback to Gemini or Ollama, you will feel an extra delay while the fallback executes.
- **Total End-to-End Latency:** Expect typical dictation round-trips to take between **1.5s and 4.0s** on CPU.

### 2. Upstream Free Providers Have Quirks & Outages
- FreeLLMAPI leverages free-tier allocations across third-party services. Those free tiers are inherently subject to:
  - Sudden provider rate limits (HTTP 429).
  - Occasional provider downtime, maintenance windows, or cold-start latency.
  - Model drift or changes in provider API schemas.
- While GlideText has automated health checks, retries, and multi-tier fallbacks to insulate you from crashes, *upstream free services will never have a 99.99% commercial SLA*.

### 3. Local LLMs Require Real Hardware
- If you want 100% offline AI polishing with Ollama, you need enough RAM/VRAM to run at least a 3-billion-parameter model (`llama3.2:3b` requires ~2.5 GB RAM; 7B/8B models require 5–8 GB).
- On CPU-only systems without an dedicated GPU, local LLM inference can take **3 to 8 seconds** per sentence, which may feel sluggish for rapid conversational dictation.

### 4. Windows UAC & UIPI Privilege Boundary
- On Windows, a standard (unelevated) process **cannot** send synthetic keystrokes or clipboard pastes to an elevated Administrator process (such as Command Prompt or PowerShell run as Administrator, Task Manager, or Registry Editor) due to User Interface Privilege Isolation (UIPI).
- If you frequently dictate into Administrator terminals or tools, **you must launch GlideText as Administrator**.

### 5. Whisper Audio Realities & Hallucination Tendencies
- OpenAI Whisper's `base` model is surprisingly good, but in high-noise environments (mechanical keyboards clacking while speaking, fan noise, coffee shops), or if your microphone gain is turned up with total silence, Whisper can occasionally hallucinate filler repetitions (e.g. repeating phrases or typing subtitles from background videos). GlideText includes silence trimming and hallucination guards, but a clean microphone input remains essential.

### 6. The Privacy Boundary: Cloud vs. Local
- **Your audio never leaves your machine.** Audio is recorded to a temporary Windows file, processed by `faster-whisper` in local RAM, and deleted.
- **However, the resulting raw text transcript IS sent over HTTP to FreeLLMAPI or Gemini** if you use Tier 1 or Tier 2. If you are handling top-secret or HIPAA-restricted text and require total air-gapped isolation, you must configure GlideText to use **only Tier 3 (Ollama)** or raw transcription.

---

## 🎙️ Core Pipeline & Architecture

### 1. Local Speech-to-Text (On-Device)
- **Audio Capture:** Recorded at 16 kHz, 16-bit mono PCM via `sounddevice`.
- **Transcription:** Local offline transcription powered by `faster-whisper`.
- **Multilingual Support:** Supports English, Hindi, Hinglish, Spanish, French, German, and 90+ other languages automatically.

### 2. Context-Aware AI Polishing
- **Automatic App Classification:** Categorizes the active window into IDE/Code Editor, Terminal, Email, Slack/Chat, Teams, Document Editor, or Browser.
- **Coding Mode:** Detects IDEs (`VS Code`, `PyCharm`, `Cursor`, `Sublime`) and terminals (`Windows Terminal`, `PowerShell`, `cmd`, `bash`, `wsl`), preserving camelCase, snake_case, CLI flags (`--help`, `-rf`), and code syntax.
- **Tone Profiles:** Normal, Formal, Casual, and Developer modes selectable from the UI.
- **Prompt Injection Defense:** Surrounding editor text is sandboxed as passive `<untrusted_cursor_context_data>` to prevent malicious text inside opened files from hijacking LLM system prompts.

### 3. Safe Text Injection
- **Target Window Verification:** Confirms window focus matches the window active when recording started, preventing accidental text insertion into the wrong window if you switch applications.
- **Clipboard vs Typing Engine:** Standard single-line ASCII text is typed directly via keystroke simulation. Multiline text and complex Unicode scripts (Devanagari, emojis, non-Latin alphabets) are pasted safely via clipboard (`Ctrl+V` on Windows, `Cmd+V` on macOS), and your previous clipboard contents are restored immediately.
- **Terminal Guard:** In shell environments (Windows Terminal, PowerShell, Command Prompt, `Terminal.app`, `iTerm2`, `Warp`, `kitty`, `alacritty`), unprompted newlines are stripped to prevent accidental execution of unfinished commands.


---

## 🗣️ Voice Commands & Dynamic Vocabulary

GlideText includes a safe, local voice command layer that operates strictly on recent GlideText dictations without executing uncontrolled OS scripts:

| Voice Command | Action | Behavior |
|---|---|---|
| **"scratch that"** / **"delete that"** | Undoes the most recent GlideText insertion | Safely backspaces the exact characters typed by GlideText; stores a recovery copy in the history vault. |
| **"delete previous sentence"** | Deletes the last sentence of the recent dictation | Safely removes trailing sentence punctuation and text. |
| **"delete previous paragraph"** | Deletes the last paragraph of the recent dictation | Scoped to the most recent dictation chunk. |
| **"undo last dictation"** / **"clear last dictation"** | Clears the entire last text injection | No-op if target window changed or text was outside session. |
| **"add [word] to my dictionary"** | Adds custom words or names to vocabulary hints | Persists directly to `dictionary.json` for future Whisper priming. |

> **Anti-False-Positive Guard:** Spoken sentences containing command words (e.g. *"I scratched that idea yesterday"* or *"Please do not delete that report"*) are transcribed as normal text and will **never** trigger destructive deletions.

---

## 🔒 Security & Privacy

- **Zero Audio Cloud Telemetry:** Raw audio is processed strictly on-device in temporary directories (`%TEMP%\glidetext` on Windows, `$TMPDIR/glidetext` on macOS) and wiped immediately after transcription.
- **Credential Safety:** Google Gemini API keys are saved securely in Windows Credential Manager or macOS Keychain via `keyring`, never hardcoded in source files or plain-text config files.
- **Sensitive App & Password Field Defense:** Dictation and context capture are automatically refused when sensitive applications (e.g. 1Password, Bitwarden, Dashlane, Keychain Access, SecurityAgent) or password fields (macOS Carbon Secure Event Input, Windows credential prompts) are focused.
- **Log Sanitization:** All logs, exceptions, and crash reports automatically scrub API keys, bearer tokens, and credentials before writing to disk.
- **Git Hygiene:** Local configuration (`config.txt`), SQLite databases (`*.db`), audio recordings (`*.wav`), and logs (`*.log`) are strictly excluded in `.gitignore`.


---

## 🚀 Setup & Installation

### Prerequisites
- **Operating System:**
  - **Windows:** Windows 10 or Windows 11 (64-bit)
  - **macOS:** macOS 12 (Monterey), 13 (Ventura), 14 (Sonoma), or 15+ (Sequoia) on Apple Silicon (M1, M2, M3, M4, M5 — Base/Pro/Max) or Intel Macs.
- **Python:** Python 3.10 to 3.12
- **Audio:** Working microphone
- **Optional for FreeLLMAPI:** Node.js (v18+) and npm
- **Optional for Local LLM:** [Ollama](https://ollama.com/) installed with `ollama pull llama3.2:3b`

### Installation

#### Windows
```cmd
git clone https://github.com/sarawgiapoorv/GlideText.git
cd GlideText

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

#### macOS (Quickstart via Setup Script)
```bash
git clone https://github.com/sarawgiapoorv/GlideText.git
cd GlideText

chmod +x setup_macos.sh
./setup_macos.sh
```
The setup script will verify your Python and Node.js environment, build the virtual environment in `.venv`, install dependencies, make `Launch_GlideText.command` executable, and run a macOS permissions diagnostic.

#### macOS (Manual Setup)
```bash
git clone https://github.com/sarawgiapoorv/GlideText.git
cd GlideText

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### Setting Up FreeLLMAPI (Tier 1 AI Brain)
1. Clone and launch the FreeLLMAPI service (see [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi)):
   ```bash
   git clone https://github.com/tashfeenahmed/freellmapi.git
   cd freellmapi
   npm install
   npm start
   ```
   *(By default, FreeLLMAPI runs on `http://127.0.0.1:3001`)*
2. In GlideText, copy `config.example.txt` to `config.txt` and set the path to your FreeLLMAPI directory:
   - Windows: `FREELLMAPI_DIR=C:\path\to\freellmapi`
   - macOS: `FREELLMAPI_DIR=/path/to/freellmapi`
   GlideText can automatically start and manage the FreeLLMAPI background server for you.

### Running GlideText
* **Windows:**
  - Standard Launch: Double-click `Launch_GlideText.bat` or run `.\.venv\Scripts\python.exe main.py`
  - Silent Background / Tray Mode: `.\.venv\Scripts\python.exe main.py --silent`
* **macOS:**
  - Standard Launch: Double-click `Launch_GlideText.command` in Finder or run `python main.py`
  - Silent Background / Tray Mode: `python main.py --silent`

### Running Unit Tests
GlideText includes a comprehensive cross-platform test suite:
```bash
# Windows
.\.venv\Scripts\python.exe -m unittest discover tests

# macOS
python -m unittest discover tests
```

---

## 🍎 macOS Permissions Guide

macOS protects user privacy by requiring explicit user consent (TCC) for microphone input, global key listeners, and synthetic keystroke injection. **GlideText never requires and should never be run with `sudo`.**

When running GlideText for the first time on macOS, ensure the following permissions are granted to your host terminal or app (e.g. `Terminal.app`, `iTerm.app`, `Visual Studio Code`, or the Python launcher):

1. **Accessibility** (*Required for text injection & simulated paste*):
   - Path: **System Settings > Privacy & Security > Accessibility**
   - Enable your terminal emulator or Python.
2. **Input Monitoring** (*Required for global push-to-talk hotkey listener*):
   - Path: **System Settings > Privacy & Security > Input Monitoring**
   - Enable your terminal emulator or Python.
3. **Microphone** (*Required for voice recording*):
   - Path: **System Settings > Privacy & Security > Microphone**
   - Allow microphone access when prompted on first dictation.

### Permissions Diagnostic Tool
To verify permission status at any time from your command line:
```bash
python -m platform_compat.permissions_check
```
You can also click the **"Check Permissions"** button in GlideText's Settings UI at any time.

---

## ⌨️ Hotkeys Reference

| Hotkey | Platform | Mode | Function |
|---|---|---|---|
| **Right Alt** (Hold & Release) | Windows | Push-to-Talk | Hold to speak, release to transcribe, polish, and type into active window. |
| **Right Option** (Hold & Release) | macOS | Push-to-Talk | Hold to speak, release to transcribe, polish, and type into active window. |
| **Ctrl + Shift + A** | Both | Continuous Mode | Hands-free continuous dictation using Voice Activity Detection (VAD). |
| **Ctrl + Shift + W** | Both | Floating Widget | Toggle floating minimal desktop status widget / full GUI dashboard. |
| **Escape** | Both | Cancel | Press while recording or transcribing to abort without typing. |
| **Cmd + W** | macOS | Window | Hide / minimize main window (keeps running in background/tray). |
| **Cmd + ,** | macOS | Settings | Toggle Settings panel. |
| **Cmd + Q** | macOS | Quit | Cleanly shut down GlideText and background servers. |

> **Custom Hotkeys:** Hotkeys are customizable via `config.txt` (e.g., `HOTKEY_PUSH_TO_TALK=alt_r` or `HOTKEY_CONTINUOUS=ctrl+shift+a`).

---

## 🛠️ Troubleshooting & Common Issues

- **Nothing types into an Administrator command prompt or Task Manager (Windows):**  
  *Cause:* Windows UIPI blocks standard apps from sending keystrokes to elevated apps.  
  *Fix:* Right-click `Launch_GlideText.bat` and select **Run as Administrator**.
- **Keystrokes not typing or hotkey not responding (macOS):**  
  *Cause:* Accessibility or Input Monitoring permission is not granted to the terminal or Python host.  
  *Fix:* Run `python -m platform_compat.permissions_check` and enable permissions in **System Settings > Privacy & Security**. Restart the terminal after granting.
- **FreeLLMAPI status shows "Offline":**  
  *Cause:* Node.js process is not running on port 3001 or Node.js is not on PATH.  
  *Fix:* Start FreeLLMAPI manually (`npm start`), or verify the `FREELLMAPI_DIR` path in `config.txt` or GUI Settings.
- **First dictation is slow:**  
  *Cause:* Whisper downloads its model weights on the very first run (stored in HuggingFace cache).  
  *Fix:* Normal; subsequent runs load instantly from local disk cache.
- **Accidental double paste or clipboard not restored:**  
  *Cause:* Certain clipboard history tools aggressively intercept clipboard writes.  
  *Fix:* In GlideText settings, enable standard keystroke typing mode if your target app supports it.

