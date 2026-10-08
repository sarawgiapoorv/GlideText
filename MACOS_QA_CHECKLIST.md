# GlideText macOS Quality Assurance & Verification Checklist

This checklist provides a comprehensive, step-by-step verification guide for validating GlideText on macOS across Apple Silicon (M1, M2, M3, M4, M5 — Base/Pro/Max/Ultra) and Intel (x86_64) hardware architectures.

---

## 1. Supported Environments & Hardware Matrix

| Platform / Chip | OS Releases | Status | Notes |
|---|---|---|---|
| **Apple Silicon (M1–M5)** | macOS 12 (Monterey) through macOS 15+ (Sequoia) | Supported | Native `arm64` wheels for faster-whisper, PyObjC, and sounddevice |
| **Intel Mac (x86_64)** | macOS 12 (Monterey) through macOS 15+ (Sequoia) | Supported | Native `x86_64` wheels; ctranslate2 CPU `int8` / `default` compute type |

---

## 2. Automated & Manual Setup Verification

- [ ] **Automated Setup Script (`./setup_macos.sh`)**
  - [ ] Execute `chmod +x setup_macos.sh && ./setup_macos.sh`.
  - [ ] Verify script correctly detects architecture (`arm64` vs `x86_64`).
  - [ ] Verify detection of Homebrew, Python (>= 3.10), and optional Node.js / Ollama.
  - [ ] Verify `.venv` creation, dependency installation, and launcher permission update.
  - [ ] Verify automatic execution of permissions diagnostic at the end of setup.
- [ ] **Manual Setup**
  - [ ] Run `python3 -m venv .venv && source .venv/bin/activate`.
  - [ ] Run `pip install -r requirements.txt`.
  - [ ] Verify zero PyObjC or CTranslate2 wheel build errors.
- [ ] **Shortcut & Launcher Verification**
  - [ ] Run `python create_shortcut.py` on macOS: verify it exits cleanly with explanatory notice without throwing PowerShell/WScript COM errors.
  - [ ] Double-click `Launch_GlideText.command` from Finder: verify Terminal opens and launches GlideText.

---

## 3. macOS Privacy & Permissions (TCC)

> **Important:** Do NOT run GlideText using `sudo`. Permissions must be granted to the host process (e.g., Terminal, iTerm, VS Code, or Python executable) via macOS System Settings.

- [ ] **Permissions CLI Diagnostic**
  - [ ] Run `python -m platform_compat.permissions_check`.
  - [ ] Verify clean, formatted report detailing Microphone, Accessibility, and Input Monitoring states.
- [ ] **Microphone Permission**
  - [ ] When holding hotkey for first time, verify system microphone prompt displays.
  - [ ] Verify audio recording functions when granted.
  - [ ] Verify graceful UI error message if denied (directing to *System Settings > Privacy & Security > Microphone*).
- [ ] **Accessibility Permission (Synthetic Keystrokes & Paste)**
  - [ ] Verify startup check calls `AXIsProcessTrustedWithOptions` to prompt if missing.
  - [ ] Verify app guides user to *System Settings > Privacy & Security > Accessibility*.
  - [ ] Once granted, verify GlideText injects text into target applications without errors.
- [ ] **Input Monitoring Permission (Global Hotkey Listener)**
  - [ ] Verify `pynput` listener captures global keypresses.
  - [ ] If permission missing, verify app logs a warning and degrades gracefully rather than crashing.
- [ ] **GUI Permissions Button**
  - [ ] Open Settings in GlideText GUI.
  - [ ] Click "Check Permissions" button: verify non-blocking background check prints status to log or dialog.

---

## 4. Input & Hotkey Verification

- [ ] **Push-to-Talk (Hold & Release)**
  - [ ] Press and hold **Right Option** (`alt_r`): verify audio recording begins.
  - [ ] Release **Right Option**: verify recording finalizes, transcribes, polishes, and injects.
- [ ] **Continuous Mode**
  - [ ] Press **Ctrl + Shift + A**: verify continuous dictation session starts with VAD.
  - [ ] Press again to stop continuous mode.
- [ ] **Floating Widget Toggle**
  - [ ] Press **Ctrl + Shift + W**: verify floating widget toggles on/off.
- [ ] **Cancellation Guard**
  - [ ] Press and hold Right Option, speak, then press **Escape**: verify session cancels immediately without injecting any text.
- [ ] **Configurable Hotkeys**
  - [ ] Edit `config.txt` setting `HOTKEY_PUSH_TO_TALK=alt_r` or custom key: verify hotkey binds as specified.

---

## 5. Text Injection & Target Window Safety

- [ ] **Single-Line ASCII Keystrokes**
  - [ ] Dictate a short sentence (e.g. "Hello world") into TextEdit: verify characters type smoothly.
- [ ] **Multiline & Unicode Text (Devanagari, Emoji, Accents)**
  - [ ] Dictate text containing newlines or emojis (e.g., "Line one\nLine two 😀"): verify injection uses clipboard paste (`Cmd+V`).
  - [ ] Verify original system clipboard content is restored immediately after injection.
- [ ] **Window Token / Focus Verification**
  - [ ] Start dictation in App A, alt-tab/switch to App B before releasing: verify GlideText detects window change and safely handles or refuses injection to prevent mis-typing.
- [ ] **Terminal Guard (Newline Stripping)**
  - [ ] Dictate multiline text into `Terminal.app`, `iTerm2`, `Warp`, `kitty`, or `alacritty`: verify newlines are stripped or converted to prevent premature command execution.
- [ ] **Sensitive App Deny List**
  - [ ] Focus a password manager or secure prompt (`1Password`, `Bitwarden`, `Dashlane`, `Keychain Access`, `SecurityAgent`):
  - [ ] Verify GlideText refuses context capture and refuses text injection.
- [ ] **Carbon Secure Input Detection**
  - [ ] Focus a password field where `IsSecureEventInputEnabled()` returns true:
  - [ ] Verify context snapshot is blanked and injection is blocked.

---

## 6. Multi-Tier AI Polishing & Fallback

- [ ] **Local Speech-to-Text (`faster-whisper`)**
  - [ ] Test on Apple Silicon: verifies `compute_type="int8"` or falls back to `"default"`.
  - [ ] Verify local transcription occurs on-device without network calls.
- [ ] **Tier 1: FreeLLMAPI**
  - [ ] If Node.js is installed, start FreeLLMAPI or configure `FREELLMAPI_DIR`.
  - [ ] Verify `_find_npm` locates npm in `/opt/homebrew/bin`, `/usr/local/bin`, or NVM paths.
  - [ ] Verify clean shutdown on app exit via process group signal (`killpg`).
- [ ] **Tier 2: Google Gemini Fallback**
  - [ ] Store Gemini API key in Settings: verify safe storage in macOS Keychain via `keyring`.
  - [ ] Disconnect FreeLLMAPI: verify automatic fallback to Gemini cloud API.
- [ ] **Tier 3: Ollama Local LLM Fallback**
  - [ ] Launch or verify Ollama path (`/opt/homebrew/bin/ollama` or `/Applications/Ollama.app`).
  - [ ] Disconnect network: verify offline cleanup via local Ollama instance.
- [ ] **Tier 4: Canonical Raw Fallback**
  - [ ] Turn off all LLM tiers and network: verify spoken words are transcribed and capitalized directly into target app without data loss.

---

## 7. Voice Commands & Vocabulary

- [ ] **"scratch that" / "delete that"**
  - [ ] Dictate a phrase, then say "scratch that": verify previous insertion is undone using translated `Cmd+Z` or simulated backspaces.
- [ ] **"delete previous sentence"**
  - [ ] Verify last sentence is removed accurately.
- [ ] **Dynamic Dictionary**
  - [ ] Say "add [CustomTerm] to my dictionary": verify term is persisted in `dictionary.json`.

---

## 8. GUI, Window Management & Lifecycle

- [ ] **macOS Standard Shortcuts**
  - [ ] Press **Cmd + W**: verify main window hides or minimizes without terminating background process.
  - [ ] Press **Cmd + ,**: verify Settings dialog opens/toggles.
  - [ ] Press **Cmd + Q**: verify app exits cleanly, terminating all child threads and servers.
- [ ] **System Tray / Menu Bar**
  - [ ] Verify tray icon displays in macOS menu bar without crashing Tkinter event loop.
  - [ ] Click menu items: "Show GlideText", "Settings", "Quit".
- [ ] **Silent Mode (`--silent`)**
  - [ ] Launch `python main.py --silent`.
  - [ ] Verify window is hidden and Dock icon is hidden (`NSApplicationActivationPolicyAccessory`).
  - [ ] Unhide via tray icon: verify Dock icon and window are restored.
- [ ] **Start at Login (Autoboot)**
  - [ ] Open Settings > toggle "Start GlideText with macOS Login".
  - [ ] Verify `~/Library/LaunchAgents/com.glidetext.app.plist` is created with valid arguments.
  - [ ] Toggle off: verify `.plist` is unloaded and deleted.
- [ ] **Data & Logs Directory**
  - [ ] Verify app data is stored in `~/Library/Application Support/GlideText`.
  - [ ] Verify logs are stored in `~/Library/Logs/GlideText`.
