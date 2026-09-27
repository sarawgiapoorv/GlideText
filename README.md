# LocalFlow 🎙️

A voice dictation tool for Windows — a personal, in-progress alternative to Wispr Flow. Hold a hotkey, speak, and the cleaned-up text gets typed into whatever window is focused.

This is a solo side project. It works for my own daily use, but read the "Known gaps" section before assuming a feature works the way it sounds.

---

## What it actually does

1. **Local transcription (always).** Speech is transcribed on-device with `faster-whisper` (`base.en`, int8). This never touches the network — the "local ASR" part of the pitch is real and unconditional.
2. **Text polishing (three tiers, tried in order):**
   - **Tier 1 — FreeLLMAPI.** If you have a local [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) instance running (a self-hosted proxy that aggregates free tiers from multiple LLM providers behind one OpenAI-compatible endpoint), LocalFlow auto-discovers its directory, spawns `npm run dev` headlessly, and routes polish requests through it. Default model: `groq/llama-3.3-70b-versatile`, with a couple of hardcoded fallback model names before finally trying `model="auto"`.
   - **Tier 2 — Direct Gemini.** If FreeLLMAPI isn't reachable, it falls back to a direct call to `gemini-2.5-flash` using your own API key. **Note:** despite what earlier versions of this README implied, this is currently a single model, not a multi-model failover array — that's a TODO in the source, not a shipped feature.
   - **Tier 3 — Local Ollama (`llama3.2:3b`).** If both cloud tiers fail, it falls back to a local model via Ollama, auto-launched headlessly if installed. Once this fallback triggers, LocalFlow goes into "sticky local mode" and stays on the local model until it detects FreeLLMAPI is reachable again (it does *not* re-probe Gemini directly to decide when to come back).
3. **Injection.** The polished text is typed at your cursor via simulated keystrokes (`keyboard` library), not clipboard-paste.

## Features that genuinely work

- **Multi-key Gemini rotation** — comma-separated keys, auto-rotates on rate limits.
- **Custom vocabulary, added by voice** — saying "add Kubernetes to my dictionary" is the one voice command that's actually implemented end-to-end; it writes to `dictionary.json`.
- **Context-aware dictionaries** — separate vocab lists for coding apps vs. chat apps (`dictionary_coding.json`, `dictionary_slack.json`), auto-selected based on the focused window.
- **Snippet expansion** — defined in `snippets.json`, expanded on the polished text before typing.
- **Tone profiles** — Normal / Formal / Casual / Developer, which change the system prompt sent to whichever tier is active.
- **Reasoning-leak guard** — regex-based post-processing that strips `<think>` blocks, catches unlabeled chain-of-thought narration ("the user wants..."), and tries to recover the actual answer instead of typing the model's internal monologue. This is genuinely useful glue code, even if it's a stack of regexes rather than anything elegant.
- **Refusal guard** — if the LLM slips into "As an AI, I cannot..." mode, LocalFlow detects it and falls back to a lightly-punctuated version of the raw transcript instead of typing the refusal.
- **API telemetry** — every polish call (FreeLLMAPI, Gemini, local) is logged to SQLite with status/latency, viewable in Settings.
- **Windows Credential Manager storage** for both the Gemini key(s) and the FreeLLMAPI key, via `keyring` — not plaintext config.
- **Continuous / VAD mode** — hands-free dictation using `webrtcvad`, toggled with `Ctrl+Shift+A`.
- **Terminal safety guard** — strips newlines from injected text when the focused window looks like a terminal, so dictation can't accidentally submit a shell command.
- **Tray + floating widget mode**, noise reduction (`noisereduce`), audio ducking (`pycaw`) — all present and wired up.

## Known gaps / things that don't work as advertised

- **Voice editing commands mostly don't fire.** "Scratch that," "undo," "make that a bulleted list," "rewrite clipboard," punctuation-by-voice, etc. are defined as constants and have handler functions written for them, but the code path that would trigger them (`detect_editing_command` → `_execute_editing_command`) is never actually connected for anything except the dictionary-add phrase. Treat this as unimplemented until it's wired up.
- **Gemini fallback is single-model**, not the multi-model array the pipeline diagram implies.
- **FreeLLMAPI discovery is a heuristic file-system scan** (env var → saved config → a list of common folder names under Desktop/OneDrive/home). If your setup doesn't match one of those, it silently falls through to Gemini with no user-facing error — you'd only see it in the logs or `diagnose_freellmapi.py`.
- **No installer.** You run from source via `Launch_LocalFlow.bat`, which pip-installs from `requirements.txt` (plus `webrtcvad`, installed separately since it isn't actually listed in `requirements.txt`).
- **Test coverage is unverified from my end** — the README used to reference `test_suite.py` and a "14-test suite," but I don't have visibility into whether those files exist in the current state of the repo, so I'm not asserting a number here. Fill this section in yourself once you confirm what's actually there.
- **Windows-only**, single-user, not packaged for distribution.

## Setup

**Requirements:**
- Windows 10/11 (64-bit), Python 3.10+
- Optional: [Ollama](https://ollama.com/) with `ollama pull llama3.2:3b` (Tier 3 fallback)
- Optional: a running FreeLLMAPI instance (Tier 1) — see note above on how LocalFlow finds it

**Run:**

Double-click `Launch_LocalFlow.bat`

or:

```bash
python main.py
python main.py --silent  # tray only
```

**Diagnose FreeLLMAPI connectivity:**

```bash
python diagnose_freellmapi.py
```

## Hotkeys

- **Push-to-talk:** hold `Right Alt`, speak, release.
- **Continuous mode:** `Ctrl+Shift+A` toggles hands-free VAD dictation.
- **"add [word] to my dictionary"** — the only functioning voice command right now.

## Why this exists

Wispr Flow is paid and closed; I wanted something local-first, using my own (mostly free-tier) API access, that I could actually modify. It's a working daily driver for me, not a finished product — expect rough edges, and PRs/issues are welcome if you find something broken.
