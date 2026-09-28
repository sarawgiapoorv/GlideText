# GlideText 🎙️

A privacy-first voice dictation tool for Windows — built as an open, offline-capable alternative to Wispr Flow (formerly named **LocalFlow**). Hold a hotkey, speak, and polished text is typed directly into whatever window is focused.

---

## AI Architecture & Provider Priority

GlideText strictly enforces a **3-tier AI polishing priority order**:

1. **Priority 1 — FreeLLMAPI** (`127.0.0.1:3001` or configured endpoint)
2. **Priority 2 — Gemini API** (Google Gemini cloud fallback)
3. **Priority 3 — Local LLM** (Local Ollama models, e.g. `llama3.2:3b`)
4. **Degraded Fallback — Raw Transcript** (Lightly punctuated raw text if all AI tiers fail)

> **In Simple Terms:**
> GlideText first tries **FreeLLMAPI**. If it is unavailable, it tries **Gemini**. If Gemini also fails, GlideText falls back to the **local LLM** on the computer. If all three fail, it types your raw speech transcript with basic punctuation so dictation never fails.

### FreeLLMAPI Integration
GlideText uses **FreeLLMAPI** as its primary LLM brain and routing layer.
- **External Project Reference:** [FreeLLMAPI GitHub repository](https://github.com/tashfeenahmed/freellmapi?utm_source=chatgpt.com) *(FreeLLMAPI is an independent external open-source dependency)*.
- **How FreeLLMAPI Works with GlideText:**
  FreeLLMAPI gives GlideText a single local API endpoint that automatically routes requests to available free LLM providers and models, with automatic fallback when a specific provider is overloaded.
- **GlideText's Role:**
  In GlideText, your speech is first converted to text locally on your device. FreeLLMAPI then sends that text to an available LLM to clean up and polish the wording before GlideText types the final text into the active application.

---

## Core Pipeline

1. **Local Speech-to-Text (Always On-Device)**  
   Audio capture uses `sounddevice` (16 kHz, 16-bit PCM). Speech is transcribed locally using `faster-whisper` (default model `base`, language `auto`). Your raw audio is transcribed entirely on-device and never leaves your computer. Multilingual recognition (Hindi, Hinglish, Spanish, French, German, etc.) is fully supported.

2. **AI Text Polishing**  
   The raw transcript is polished to fix grammatical errors, remove conversational filler ("um", "uh"), and apply the active tone style (Normal, Formal, Casual, Developer).

3. **Safe Text Injection**  
   - **Target Window Verification:** Verifies window focus against the target window captured when recording began, refusing injection if the user switched windows.
   - **Unicode & Multiline Safety:** Single-line ASCII text types via `keyboard.write`; Unicode scripts (e.g. Hindi, Devanagari, emojis) and multiline text inject via Windows clipboard (`Ctrl+V`), immediately restoring your previous clipboard content in the background.
   - **Terminal Guard:** In shell environments (PowerShell, CMD, Bash, WSL), unprompted newlines are stripped to prevent accidental command execution.

---

## Features

- **Push-to-Talk & Continuous Mode:** Hold `Right Alt` to record and release to dictate, or press `Ctrl+Shift+A` for hands-free continuous dictation with Voice Activity Detection (VAD).
- **Custom Voice Vocabulary:** Say "add Kubernetes to my dictionary" to append terms dynamically to your vocabulary dictionary.
- **Context-Aware Dictionaries:** Automatically loads app-specific dictionaries (`dictionary_coding.json` for IDEs/terminals, `dictionary_slack.json` for messaging apps) based on the active window.
- **Snippet Expansion:** Automatically expands shortcut triggers configured in `snippets.json`.
- **Tone Profiles:** Normal, Formal, Casual, and Developer styles selectable from the UI.
- **Output Cleaning & Anti-Chatbot Guards:**
  - Strips reasoning blocks (`<think>...</think>`), markdown code fences, and conversational preambles (`"Sure! Here is the text:"`).
  - Guards against AI refusal chatter, ensuring you always get polished dictation text.
- **Telemetry & History Vault:** Logs dictation history and latency to a local SQLite database (`glidetext_history.db`), securely viewable inside the Settings panel.

---

## Security & Privacy

- **On-Device Audio:** Audio recordings are stored in temporary files (`%TEMP%\glidetext`) and transcribed locally. No raw audio is ever uploaded to external cloud servers.
- **Text-Only Requests:** Only transcribed text is sent to the configured text polish provider.
- **Credential Storage:** API keys are stored securely in Windows Credential Manager (`keyring`), never in source files or repository commits.
- **Log Sanitization:** All log outputs, exception tracebacks, and error messages redact API keys and bearer tokens.
- **Git Hygiene:** Local configuration (`config.txt`), SQLite databases (`*.db`), logs (`*.log`), audio recordings (`*.wav`), `.venv/`, and diagnostic markers (`deps_ok`) are strictly `.gitignore`d.

---

## Setup & Launch

### Requirements
- Windows 10/11 (64-bit)
- Python 3.10+
- Optional: Node.js (for running FreeLLMAPI locally)
- Optional: [Ollama](https://ollama.com/) with `ollama pull llama3.2:3b`

### Installation
```cmd
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Configuration
Copy `config.example.txt` to `config.txt` to customize settings:
```text
0
FREELLMAPI_DIR=C:\path\to\your\freellmapi
WHISPER_MODEL=base
WHISPER_LANGUAGE=auto
```
Or copy `.env.example` to `.env` if configuring via environment variables.

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

### Running Unit Tests
```cmd
.\.venv\Scripts\python.exe -m unittest discover tests
```

---

## Hotkeys

| Hotkey | Action |
|---|---|
| **Right Alt** (Hold) | Push-to-talk recording |
| **Ctrl + Shift + A** | Toggle continuous VAD dictation mode |
| **Ctrl + Shift + W** | Toggle floating minimal widget / full dashboard |
