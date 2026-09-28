"""
local_llm.py -- Local LLM Engine with Headless Ollama Management.

Features:
- Headless auto-discovery and background launch of Ollama on Windows (no console popups).
- Zero-configuration execution for local Stage 2 speech polishing using llama3.2:3b.
- Automatic model pre-warming and fast streaming/non-streaming inference.
- Clean text extraction with anti-chatbot quotation/preamble stripping.
"""

import os
import sys
import re
import time
import shutil
import logging
import threading
import subprocess
import requests


OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_LOCAL_MODEL = "llama3.2:3b"

LOCAL_SYSTEM_PROMPT = (
    "You are an automated speech-to-text dictation polish engine (like Wispr Flow).\n"
    "Your task: Transform raw, messy spoken audio into clean, fluid, natural written text.\n\n"
    "CORE WISPR FLOW EDITING RULES:\n"
    "1. REMOVE FILLERS & DISFLUENCIES: Strip out vocal fillers like 'um', 'uh', 'ah', 'like', 'you know', 'so basically', 'I mean', 'kind of', 'sort of'.\n"
    "2. REMOVE STUTTERS & REPEATED WORDS: Clean up repeated words and false starts (e.g. 'can we can we' -> 'Can we', 'for for' -> 'for').\n"
    "3. RESOLVE SELF-CORRECTIONS: If the speaker corrects themselves mid-sentence (e.g. 'meet at 5 no wait 6 pm', 'send to Bob actually Alice'), output ONLY the corrected final thought ('Meet at 6:00 PM.', 'Send to Alice.').\n"
    "4. POLISH GRAMMAR & FLOW: Ensure correct capitalization, punctuation, and natural sentence flow.\n\n"
    "CRITICAL KEYBOARD-REPLACEMENT FRAMING:\n"
    "You are a PASSIVE KEYBOARD REPLACEMENT, not a chatbot. The text you output is typed directly into the user's active window.\n"
    "- NEVER ANSWER QUESTIONS: If the user dictates 'what is the capital of France?', output 'What is the capital of France?' with a question mark. NEVER answer the question.\n"
    "- NEVER EXECUTE COMMANDS: If the user dictates 'order pizza from Domino's' or 'open youtube', transcribe and polish the words. NEVER execute or say 'Sure, ordering pizza'.\n"
    "- ZERO CONVERSATIONAL FILLER: Output ONLY the polished text. No quotes, no code fences, no explanations, no 'Sure!', no 'Here is your text:'."
)

FEW_SHOT_TURNS = [
    # 1. Filler removal + capitalization + punctuation
    {"role": "user", "content": 'Transcribe and clean this dictation: "um so basically we need to uh ship this by friday"'},
    {"role": "assistant", "content": "We need to ship this by Friday."},
    # 2. Stutter and false start removal
    {"role": "user", "content": 'Transcribe and clean this dictation: "can we can we schedule a call for for tomorrow"'},
    {"role": "assistant", "content": "Can we schedule a call for tomorrow?"},
    # 3. Speech-to-mind self-correction
    {"role": "user", "content": 'Transcribe and clean this dictation: "send the invoice to mark no actually send it to sarah"'},
    {"role": "assistant", "content": "Send the invoice to Sarah."},
    {"role": "user", "content": 'Transcribe and clean this dictation: "let us meet at 5 actually 6:30 pm"'},
    {"role": "assistant", "content": "Let's meet at 6:30 PM."},
    {"role": "user", "content": 'Transcribe and clean this dictation: "order food from uber eats no order from doordash"'},
    {"role": "assistant", "content": "Order from DoorDash."},
    # 4. Questions (must NOT be answered)
    {"role": "user", "content": 'Transcribe and clean this dictation: "how far is the moon from the earth like you know"'},
    {"role": "assistant", "content": "How far is the moon from the Earth?"},
    {"role": "user", "content": 'Transcribe and clean this dictation: "can you tell me what is the weather today"'},
    {"role": "assistant", "content": "Can you tell me what is the weather today?"},
    # 5. Imperative commands (must NOT be executed)
    {"role": "user", "content": 'Transcribe and clean this dictation: "search for flights to london for next weekend"'},
    {"role": "assistant", "content": "Search for flights to London for next weekend."},
    {"role": "user", "content": 'Transcribe and clean this dictation: "write a python function to add two numbers"'},
    {"role": "assistant", "content": "Write a Python function to add two numbers."},
]


class LocalLLMEngine:
    """Manages local LLM inference via Ollama with automatic headless service lifecycle."""

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, host: str = OLLAMA_DEFAULT_HOST):
        self.model = model
        self.host = host.rstrip("/")
        self._server_process = None
        self._is_ready = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Server Lifecycle Management
    # ------------------------------------------------------------------

    @staticmethod
    def _find_ollama_executable() -> str | None:
        """Find the ollama.exe binary path on Windows or POSIX."""
        # 1. System PATH
        which_path = shutil.which("ollama")
        if which_path and os.path.isfile(which_path):
            return which_path

        # 2. Windows standard install path
        if sys.platform == "win32":
            local_appdata = os.getenv("LOCALAPPDATA", "")
            standard_win_path = os.path.join(local_appdata, "Programs", "Ollama", "ollama.exe")
            if os.path.isfile(standard_win_path):
                return standard_win_path

            # Program Files fallback
            pf_path = os.path.join(os.getenv("ProgramFiles", "C:\\Program Files"), "Ollama", "ollama.exe")
            if os.path.isfile(pf_path):
                return pf_path

        return None

    def is_server_running(self) -> bool:
        """Check if Ollama server responds to HTTP ping."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=1.2)
            return resp.status_code == 200
        except Exception:
            return False

    def ensure_server_running(self, timeout_seconds: float = 15.0) -> bool:
        """Ensure Ollama is running. If not, auto-launch it in background without window."""
        if self.is_server_running():
            self._is_ready = True
            return True

        binary = self._find_ollama_executable()
        if not binary:
            logging.error("[LocalLLM] Ollama executable not found on system PATH or standard directories.")
            return False

        logging.info(f"[LocalLLM] Ollama server not responding. Auto-starting headless: {binary} serve")
        try:
            creation_flags = 0
            if sys.platform == "win32":
                # DETACHED_PROCESS = 0x00000008, CREATE_NO_WINDOW = 0x08000000
                creation_flags = 0x08000000

            self._server_process = subprocess.Popen(
                [binary, "serve"],
                creationflags=creation_flags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logging.error(f"[LocalLLM] Failed to start Ollama background process: {e}")
            return False

        # Poll until responsive
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            time.sleep(0.4)
            if self.is_server_running():
                logging.info("[LocalLLM] Ollama background server is now healthy and ready.")
                self._is_ready = True
                return True

        logging.warning("[LocalLLM] Ollama server start timed out.")
        return False

    def is_model_installed(self, model_name: str | None = None) -> bool:
        """Check if target model is present in Ollama's local registry."""
        target = model_name or self.model
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                models = [m.get("name", "").split(":")[0] for m in resp.json().get("models", [])]
                full_names = [m.get("name", "") for m in resp.json().get("models", [])]
                base_target = target.split(":")[0]
                return target in full_names or base_target in models
        except Exception:
            pass
        return False

    def warm_up_in_background(self) -> None:
        """Load model weights into memory asynchronously to eliminate first-token latency."""
        def _warmup_task():
            if not self.ensure_server_running():
                return
            logging.info(f"[LocalLLM] Pre-warming model '{self.model}' in background...")
            try:
                payload = {
                    "model": self.model,
                    "prompt": "",
                    "keep_alive": "1h",
                }
                requests.post(f"{self.host}/api/generate", json=payload, timeout=20)
                logging.info(f"[LocalLLM] Model '{self.model}' pre-warmed and resident in RAM.")
            except Exception as e:
                logging.warning(f"[LocalLLM] Model pre-warm notice: {e}")

        threading.Thread(target=_warmup_task, daemon=True).start()

    # ------------------------------------------------------------------
    # Polish / Speech-to-Mind Inference
    # ------------------------------------------------------------------

    def polish(
        self,
        raw_text: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout: float = 25.0,
    ) -> str | None:
        """
        Polish raw spoken text locally using the loaded model.
        Returns polished text or None on failure.
        """
        if not raw_text or not raw_text.strip():
            return raw_text

        if not self.ensure_server_running():
            logging.error("[LocalLLM] Cannot run polish: Ollama server is unavailable.")
            return None

        if system_prompt and system_prompt.strip():
            if "PASSIVE KEYBOARD REPLACEMENT" in system_prompt or "Wispr Flow" in system_prompt or "EDITOR_SYSTEM_PROMPT" in system_prompt:
                effective_system = system_prompt
            else:
                effective_system = f"{LOCAL_SYSTEM_PROMPT}\n\n{system_prompt}"
        else:
            effective_system = LOCAL_SYSTEM_PROMPT

        messages = [{"role": "system", "content": effective_system}]
        messages.extend(FEW_SHOT_TURNS)
        messages.append({
            "role": "user",
            "content": f'Transcribe and clean this dictation: "{raw_text.strip()}"'
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 160,
            },
        }

        try:
            t0 = time.time()
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=timeout,
            )
            elapsed_ms = int((time.time() - t0) * 1000)

            if resp.status_code != 200:
                logging.error(f"[LocalLLM] Ollama returned HTTP {resp.status_code}: {resp.text[:120]}")
                return None

            data = resp.json()
            raw_output = data.get("message", {}).get("content", "").strip()

            if not raw_output:
                logging.warning("[LocalLLM] Received empty response from local model.")
                return None

            # Clean any stray wrapping quotes or common model chatter
            cleaned = self._clean_model_output(raw_output, raw_text=raw_text)
            logging.info(f"[LocalLLM] Polish succeeded in {elapsed_ms}ms with '{self.model}': {repr(cleaned)}")
            return cleaned

        except requests.Timeout:
            logging.error(f"[LocalLLM] Request timed out after {timeout}s on '{self.model}'.")
            return None
        except Exception as e:
            logging.error(f"[LocalLLM] Inference error on '{self.model}': {e}")
            return None

    @staticmethod
    def _clean_model_output(text: str, raw_text: str = "") -> str:
        """
        Strip model-introduced wrapping, conversational preambles, and assistant
        suffixes from the raw output so only the clean dictation transcription remains.

        Guards applied (in order):
          1. Outer quote wrapper  (" ... " or ' ... ')
          2. Markdown code-fence wrapper
          3. Conversational preamble prefixes  (Sure!, Here is your text:, etc.)
          4. Informational / explanation prefixes  (Here is the corrected text:, etc.)
          5. Anti-chatbot refusal guard  (reroutes to safe raw transcription)
          6. Trailing explanatory suffix guard  (strips '(Note: ...)' appended text)
        """
        cleaned = text.strip()

        # ── 1. Outer quote wrapper ───────────────────────────────────────────
        if len(cleaned) >= 2:
            if (cleaned.startswith('"') and cleaned.endswith('"')) or \
               (cleaned.startswith("'") and cleaned.endswith("'")):
                cleaned = cleaned[1:-1].strip()

        # ── 2. Markdown code fence ───────────────────────────────────────────
        if cleaned.startswith("`") and cleaned.endswith("`"):
            cleaned = cleaned.strip("`").strip()

        lower = cleaned.lower()

        # ── 3. Conversational preamble prefixes (assistant chatter) ──────────
        # These fire FIRST because they are single-word/short and must be stripped
        # before the longer informational prefixes are checked.
        conversational_preambles = [
            "sure!",
            "sure,",
            "certainly!",
            "certainly,",
            "of course!",
            "of course,",
            "absolutely!",
            "absolutely,",
            "great!",
            "great,",
            "great question!",
            "no problem!",
            "happy to help!",
            "got it!",
            "understood!",
            "noted!",
        ]
        for preamble in conversational_preambles:
            if lower.startswith(preamble):
                cleaned = cleaned[len(preamble):].lstrip(" ,!\n")
                lower = cleaned.lower()
                break  # only strip one preamble per call

        # ── 3.5. Chain-of-thought / reasoning-leak guard ─────────────────────
        # a. Strip labeled <think>...</think> or <reasoning>...</reasoning> blocks first
        cleaned = re.sub(r'(?is)<(think|reasoning)>.*?(?:</\1>|\Z)', '', cleaned).strip()
        lower = cleaned.lower()

        # b. Detect unlabeled reasoning narration or excessive length
        reasoning_markers = [
            "the user wants",
            "the user is dictating",
            "the user is asking",
            "the user asks",
            "let's apply",
            "let me parse",
            "okay, the user",
            "as a speech-to-text",
            "as an ai transcriber",
            "transcription process:",
            "thinking process:",
            "internal thought:",
        ]
        start_sample = lower[:200]
        raw_lower = raw_text.strip().lower()
        has_reasoning_marker = any(
            m in start_sample and m not in raw_lower
            for m in reasoning_markers
        )
        is_length_leak = bool(
            raw_text
            and len(cleaned) > 200
            and len(cleaned) > 4 * len(raw_text.strip())
        )

        if has_reasoning_marker or is_length_leak:
            # c. On a detected leak, try to recover the actual clean dictation
            recovered = None

            # Pattern 1: Output markers (e.g., Final output:, Cleaned text:, Transcription:, Result:)
            matches = list(re.finditer(
                r'(?i)\b(?:final\s+)?(?:intended\s+thought|answer|output|transcript|transcription|cleaned\s+text|polished\s+text|cleaned\s+version|clean\s+transcript|result)\s*:\s*(?:(["\'])(.*?)\1|([^\r\n]+))',
                cleaned
            ))
            if matches:
                last = matches[-1]
                if last.group(2) is not None:
                    cand = last.group(2).strip()
                else:
                    cand = last.group(3).strip()
                    if len(cand) >= 2 and (
                        (cand.startswith('"') and cand.endswith('"')) or
                        (cand.startswith("'") and cand.endswith("'"))
                    ):
                        cand = cand[1:-1].strip()
                    cand = cand.strip('"\'').strip()

                if 0 < len(cand) < 300 and not any(rm in cand.lower() for rm in reasoning_markers):
                    recovered = cand

            # Pattern 2: Quoted phrase at the end of the reasoning block
            if not recovered:
                quoted = re.findall(r'"([^"\n\r]{2,250})"', cleaned)
                if quoted:
                    for q_cand in reversed(quoted):
                        q_cand = q_cand.strip()
                        if not any(rm in q_cand.lower() for rm in reasoning_markers):
                            recovered = q_cand
                            break

            # Pattern 3: Last non-empty line if it doesn't contain reasoning markers
            if not recovered:
                lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
                if lines:
                    last_line = lines[-1].strip('"\' ')
                    if 0 < len(last_line) < 250 and not any(rm in last_line.lower() for rm in reasoning_markers):
                        recovered = last_line

            if recovered:
                cleaned = recovered
                lower = cleaned.lower()
            else:
                # d. No recoverable final line: DO NOT type reasoning text.
                # Fall back to lightly-punctuated raw_text and log warning.
                logging.warning(
                    f"[LocalLLM] Reasoning leak detected and discarded ({len(cleaned)} chars). "
                    "Falling back to lightly-punctuated raw text."
                )
                raw_fallback = raw_text.strip()
                if raw_fallback:
                    first_words = raw_fallback.lower().split()[:2]
                    is_q = any(
                        w in first_words
                        for w in ["what", "how", "who", "where", "when", "why",
                                  "can", "could", "is", "are", "does", "did", "will", "would"]
                    )
                    if is_q and not raw_fallback.endswith("?"):
                        return raw_fallback + "?"
                    if not raw_fallback.endswith((".", "!", "?")):
                        return raw_fallback + "."
                    return raw_fallback
                return ""

        # ── 3.6 Word-overlap ratio guard (chatbot divergence detector) ───────
        # A polish should be ~the same words, cleaned up. If the output shares
        # less than 25% of its words with the input, the model has almost
        # certainly gone into chatbot/answer mode rather than polishing.
        if raw_text and not is_length_leak:
            raw_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', raw_text.lower()))
            out_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', cleaned.lower()))
            if raw_words and out_words and len(out_words) > 3:
                overlap = len(raw_words & out_words) / max(len(out_words), 1)
                if overlap < 0.25:
                    logging.warning(
                        f"[LocalLLM] Word-overlap guard triggered: "
                        f"overlap={overlap:.0%} ({len(raw_words & out_words)}/{len(out_words)} words). "
                        f"Output looks like a chatbot response, not a polish. "
                        f"Falling back to raw text."
                    )
                    raw_fallback = raw_text.strip()
                    if raw_fallback:
                        first_words = raw_fallback.lower().split()[:2]
                        is_q = any(
                            w in first_words
                            for w in ["what", "how", "who", "where", "when", "why",
                                      "can", "could", "is", "are", "does", "did", "will", "would"]
                        )
                        if is_q and not raw_fallback.endswith("?"):
                            return raw_fallback + "?"
                        if not raw_fallback.endswith((".", "!", "?")):
                            return raw_fallback + "."
                        return raw_fallback
                    return ""

        # ── 4. Informational / explanation prefixes ──────────────────────────

        informational_prefixes = [
            "here is the polished text:",
            "here's the polished text:",
            "here is the corrected text:",
            "here's the corrected text:",
            "here is the cleaned text:",
            "here's the cleaned text:",
            "here is the transcription:",
            "here's the transcription:",
            "here is your text:",
            "here's your text:",
            "corrected text:",
            "cleaned text:",
            "polished text:",
            "transcription:",
            "clean polished transcript:",
            "output:",
            "result:",
        ]
        for p in informational_prefixes:
            if lower.startswith(p):
                cleaned = cleaned[len(p):].lstrip(" \n")
                lower = cleaned.lower()
                break

        # ── 5. Anti-chatbot refusal guard ────────────────────────────────────
        # If the model slips into refusal / assistant mode, recover by returning
        # the raw dictation as a clean transcription.
        bot_refusal_prefixes = [
            "as an ai,",
            "as an ai language model",
            "as a language model",
            "as a large language model",
            "i cannot",
            "i can't",
            "i am unable to",
            "i'm unable to",
            "i'm not capable",
            "i am not capable",
            "i don't have access",
            "i do not have access",
            "i'm not able to",
            "i am not able to",
            "i'm sorry, but",
            "i apologize, but",
        ]
        if raw_text and any(lower.startswith(bp) for bp in bot_refusal_prefixes):
            first_words = raw_text.strip().lower().split()[:2]
            is_q = any(
                w in first_words
                for w in ["what", "how", "who", "where", "when", "why",
                          "can", "could", "is", "are", "does", "did", "will"]
            )
            logging.warning(
                f"[LocalLLM] Guarded against assistant refusal: {repr(cleaned[:60])}. "
                "Returning safe transcription of raw text."
            )
            raw_stripped = raw_text.strip()
            if is_q and not raw_stripped.endswith("?"):
                return raw_stripped + "?"
            if not raw_stripped.endswith((".", "!", "?")):
                return raw_stripped + "."
            return raw_stripped

        # ── 5.5 Chatbot-answer mode guard ────────────────────────────────
        # Unlike refusals (step 5), this catches cases where the model
        # helpfully tries to ANSWER or EXECUTE the dictated content instead
        # of polishing it. E.g., user says "search for flights to London"
        # and the model generates "I'd be happy to help you search...".
        chatbot_answer_patterns = [
            "i'd be happy to",
            "i would be happy to",
            "i'd love to help",
            "i can help you",
            "let me help you",
            "here are some",
            "here is a",
            "here's a",
            "here is how",
            "here's how",
            "the answer is",
            "the distance is",
            "the time is",
            "to do this,",
            "you can ",
            "you could ",
            "you should ",
            "to answer your question",
            "based on your request",
            "in response to",
            "this is a great question",
            "that's a great question",
            "to help you with",
            "here is what you need",
            "here's what you need",
            "i found",
            "i'll help",
            "i will help",
            "let me ",
            "first, ",
            "step 1",
            "1.",
        ]
        if raw_text:
            raw_first_5 = " ".join(raw_text.strip().lower().split()[:5])
            matched_pattern = None
            for bp in chatbot_answer_patterns:
                if bp == "1.":
                    if re.match(r'^1\.(?!\d)\s*', lower):
                        matched_pattern = "1."
                        break
                elif lower.startswith(bp):
                    matched_pattern = bp.strip()
                    break

            if matched_pattern:
                # Check if raw text contains the same phrase anywhere in its first ~5 words
                # (e.g. user dictated "1.5 million" or "you can do that" or "first, clean this up")
                pattern_words = matched_pattern.lower().rstrip(".,!?").split()
                has_match = (
                    matched_pattern.lower() in raw_first_5
                    or (pattern_words and " ".join(pattern_words) in raw_first_5)
                    or (len(pattern_words) >= 2 and any(" ".join(pattern_words[i:i+2]) in raw_first_5 for i in range(len(pattern_words)-1)))
                )
                if not has_match:
                    logging.warning(
                        f"[LocalLLM] Chatbot-answer guard triggered: {repr(cleaned[:80])}. "
                        "Model is answering/executing instead of polishing. "
                        "Returning safe transcription of raw text."
                    )
                    return LocalLLMEngine.format_lightly_punctuated_raw(raw_text)

        # ── 6. Trailing explanatory suffix guard ─────────────────────────────
        # Some models append "Note: ..." or "(As an AI, ...)" after the transcription.
        # Strip everything from the first occurrence of these patterns onward.
        trailing_patterns = [
            "\n\nnote:",
            "\nnote:",
            "\n\n(note:",
            "\n(as an ai",
            "\n\n(as an ai",
            "\n\nplease note",
            "\nplease note",
        ]
        lower_full = cleaned.lower()
        for pat in trailing_patterns:
            idx = lower_full.find(pat)
            if idx != -1:
                cleaned = cleaned[:idx].strip()
                lower_full = cleaned.lower()
                break

        # Re-strip quotes if outer quotes were wrapped around the body after preamble
        cleaned = cleaned.strip()
        if len(cleaned) >= 2:
            if (cleaned.startswith('"') and cleaned.endswith('"')) or \
               (cleaned.startswith("'") and cleaned.endswith("'")):
                cleaned = cleaned[1:-1].strip()

        return cleaned


