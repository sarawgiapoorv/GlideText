# GlideText 🎙️

A privacy-first voice dictation tool for Windows — built as a personal, in-progress alternative to Wispr Flow (formerly named **LocalFlow**). Hold a hotkey, speak, and polished text is typed directly into whatever window is focused.

---

## Real Architecture

1. **Local Speech-to-Text (Always On-Device)**  
   Audio capture uses `sounddevice` (16 kHz, 16-bit PCM). Speech is transcribed locally using `faster-whisper` (`base.en`, int8 quantization). Your raw voice audio is transcribed entirely on-device and never leaves your machine.

2. **3-Tier AI Polish Pipeline (Tried in Order)**  
   Once text is transcribed, it is polished for grammar, punctuation, filler removal, and tone:
   - **Tier 1 — FreeLLMAPI Proxy (`127.0.0.1:3001`):** A headless local proxy aggregating free LLM provider endpoints. Defaults to `model="auto"`, auto-queries `/v1/models` on HTTP 400/404/422 errors, and operates under a hard ~8-second budget cap.
   - **Tier 2 — Direct Gemini API:** Uses `gemini-2.5-flash` with fallback to `gemini-2.0-flash`. The API key is sent via the `x-goog-api-key` header.
   - **Tier 3 — Local Ollama:** Uses local models (`qwen2.5:3b`, `llama3.2:3b`) via Ollama's REST API (`127.0.0.1:11434`). If online tiers fail repeatedly (2+ consecutive failures), sticky local mode is activated. If Ollama is unavailable, GlideText falls back to lightly-punctuated raw text.

3. **Keystroke Injection**  
   Polished text is injected at your active cursor position via `keyboard`. In terminal windows (e.g. PowerShell, CMD, WSL), newlines are automatically sanitized to prevent accidental command execution.

---

## Gemini Key is Optional

A Gemini API key is **not required** to use GlideText:
- Recording, local transcription, Tier 1 FreeLLMAPI polish, Tier 3 Ollama polish, and raw text fallbacks function completely without a Gemini key.
- If provided, keys are validated only when typed and saved securely to Windows Credential Manager.

---

## Features

- **Push-to-Talk & Continuous Mode:** Hold `Right Alt` to record and release to dictate, or press `Ctrl+Shift+A` for hands-free continuous dictation.
- **Custom Voice Vocabulary:** Say "add Kubernetes to my dictionary" to append terms to `dictionary.json`.
- **Context-Aware Dictionaries:** Automatically loads app-specific dictionaries (`dictionary_coding.json` for IDEs/terminals, `dictionary_slack.json` for communication apps) based on the focused window.
- **Snippet Expansion:** Replaces keyword triggers defined in `snippets.json` (template provided with generic placeholders).
- **Tone Profiles:** Normal, Formal, Casual, and Developer styles incorporated into system prompts across all tiers.
- **Output Cleaning & Anti-Chatbot Guards:**
  - `_clean_model_output()` strips markdown fences, quote wrappers, conversational chatter (`"Sure!"`, `"Here is the polished text:"`), and trailing explanatory notes.
  - Word-overlap and refusal guards prevent LLM chatbot responses or execution attempts, falling back to clean transcriptions.
- **Lookback Context (Optional):** Controlled by `LOOKBACK_CONTEXT=0` (disabled by default). When set to `1`, inspects preceding words around the cursor, automatically skipped in terminal windows.
- **Telemetry & History Vault:** Logs polish attempts, model tiers, and latency to a local SQLite database (`glidetext_history.db`), viewable in the Settings panel.

---

## Security & Privacy

- **On-Device Audio:** Audio recordings are stored in temporary files (`%TEMP%\glidetext`) and transcribed locally. No raw audio is ever uploaded to external cloud servers.
- **Text-Only Cloud Requests:** Only raw transcribed text (never audio) is sent to external or local LLM polish tiers.
- **Credential Storage:** API keys live in Windows Credential Manager (`keyring`) or environment variables, never in source files or configuration logs.
- **Log Sanitization:** All log output, exception tracebacks, and crash reports sanitize Gemini (`AIza...`), OpenRouter (`sk-or-...`), and FreeLLMAPI keys before writing to disk.
- **Git Hygiene:** Local configuration (`config.txt`), SQLite databases (`*.db`), logs (`*.log`), audio recordings (`*.wav`), `.venv/`, and diagnostic markers (`deps_ok`) are strictly `.gitignore`d.

---

## Setup & Launch

### Requirements
- Windows 10/11 (64-bit)
- Python 3.10+
- Optional: FreeLLMAPI directory (auto-discovered if placed on Desktop or configured in `config.txt`)
- Optional: [Ollama](https://ollama.com/) with `ollama pull qwen2.5:3b` or `ollama pull llama3.2:3b`

### Installation (Dedicated Virtual Environment)
```cmd
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Configuration Template
Copy `config.example.txt` to `config.txt` if custom path overrides are needed:
```text
0
FREELLMAPI_DIR=C:\path\to\your\freellmapi
```
*(Line 1 represents the microphone device index; `FREELLMAPI_DIR` points to your FreeLLMAPI directory.)*

### Running GlideText
* **Standard Launch:** Double-click `Launch_GlideText.bat` (or `Launch_GlideText.vbs`)
* **Terminal Launch:**
  ```cmd
  .\.venv\Scripts\python.exe main.py
  ```
* **Silent Tray-Only Launch:**
  ```cmd
  .\.venv\Scripts\python.exe main.py --silent
  ```

### Desktop Shortcut
Recreate the Desktop shortcut pointing to the launcher:
```cmd
.\.venv\Scripts\python.exe create_shortcut.py
```

### FreeLLMAPI Diagnostic Utility
Verify FreeLLMAPI process management and catalog connectivity:
```cmd
.\.venv\Scripts\python.exe diagnose_freellmapi.py
```

### Running Unit Tests
```cmd
.\.venv\Scripts\python.exe tests/test_clean.py
```

---

## Hotkeys

| Hotkey | Action |
|---|---|
| **Right Alt** (Hold) | Push-to-talk recording |
| **Ctrl + Shift + A** | Toggle continuous VAD dictation mode |
| **Ctrl + Shift + W** | Toggle floating minimal widget / full dashboard |

---

## Known Gaps

- **Voice Editing Commands:** Complex voice editing commands ("delete previous paragraph", "scratch that") are not fully wired up yet.
- **Optional Dependencies:** DSP noise reduction (`noisereduce`) and WebRTC VAD (`webrtcvad`) are soft-dependencies; if C++ build tools are missing, GlideText uses built-in RMS energy threshold VAD seamlessly.
- **Windows-Only:** Built specifically for Windows OS APIs (Win32 window focus detection, Windows registry, pycaw audio ducking, COM interfaces).
