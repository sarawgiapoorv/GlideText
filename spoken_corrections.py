"""
spoken_corrections.py -- Context-Aware Spoken Self-Correction Resolution for GlideText.

This module resolves natural spoken self-corrections inside a DictationSession
transcript while preserving the speaker's intended meaning, technical vocabulary,
and normal uses of words like "actually", "sorry", "rather", and "instead".

===========================================================================
SUPPORTED CORRECTION PATTERNS
===========================================================================

1. Explicit Replacement Directives (`make that`, `make it`, `change that to`, `correction`):
   - Trigger phrases:
     * "actually make that <Y>" / "actually, make that <Y>"
     * "no, make that <Y>" / "no make that <Y>" / "no, make it <Y>" / "no make it <Y>"
     * "wait, make that <Y>" / "wait make that <Y>" / "wait, make it <Y>"
     * "sorry, make that <Y>" / "sorry make that <Y>"
     * "change that to <Y>" / "change it to <Y>"
     * "correction, <Y>" / "correction: <Y>"
   - Examples:
     * "Let's meet at five PM, actually make that six thirty."
       -> "Let's meet at six thirty."
     * "Create a three-column table—no, make it four columns."
       -> "Create a four-column table."
     * "Order 5 boxes, change that to 10."
       -> "Order 10 boxes."

2. Full-Clause / Sentence Retractions (`scratch that`, `forget that`):
   - Trigger phrases:
     * "scratch that"
     * "forget that"
   - Replaces the preceding clause or sentence segment with the new statement:
     * "We should deploy on Friday afternoon, scratch that, let's wait until Monday morning."
       -> "Let's wait until Monday morning."

3. Parallel Verb / Preposition Re-framing (`actually <verb>`, `let's <verb>`, `instead <verb>`):
   - When a correction marker ("actually", "sorry", "rather", "instead", "I mean", "no wait")
     is followed by a phrase that repeats the verb or preposition from the antecedent
     (or re-frames the clause with "let's <verb>" / "we should <verb>" / "I want to <verb>"):
     * "Use Redis—actually use PostgreSQL for persistence."
       -> "Use PostgreSQL for persistence."
     * "I want to use React—actually, let's use Vue."
       -> "Let's use Vue."
     * "Send the invoice to Mark, no actually send it to Sarah."
       -> "Send the invoice to Sarah."

4. Same-Category Atomic Substitutions (Numbers, Dates/Times, Names, Technical Terms):
   - When a pause/punctuation boundary + correction marker ("sorry", "I mean", "actually",
     "rather", "no wait", "or rather") is followed by a concise replacement of the same
     semantic category as the trailing element of the antecedent:
     * Date/Time: "Send it tomorrow—sorry, Thursday." -> "Send it Thursday."
     * Number: "Set retries to 3, I mean 5." -> "Set retries to 5."
     * Name: "Email John—I mean David—about the release." -> "Email David about the release."
     * Tech term: "Deploy to Kubernetes, sorry Docker Swarm." -> "Deploy to Docker Swarm."

5. Chained / Multiple Corrections in a Single Session:
   - Corrections are resolved iteratively left-to-right within each sentence and across
     sentences in a DictationSession:
     * "Let's meet on Monday, sorry Tuesday, actually make that Wednesday."
       -> "Let's meet on Wednesday."

6. Non-Correction & Ambiguity Safeguards:
   - Normal grammatical uses of "actually", "sorry", "instead", and "rather" are NEVER deleted:
     * "I actually really like the new architecture." (preserved)
     * "I am sorry for the delay in responding." (preserved)
     * "We used PostgreSQL instead of Redis." (preserved)
   - Ambiguous reflections or commentary clauses after "actually" or "sorry" are preserved
     verbatim whenever there is no clear structural replacement target:
     * "I was looking at the logs, actually I'm not sure what caused the timeout." (preserved)
"""

from __future__ import annotations

import re
from typing import Optional


SUPPORTED_CORRECTION_PATTERNS: list[dict[str, str]] = [
    {
        "name": "explicit_directive",
        "triggers": (
            "actually make that, no make that, no make it, wait make that, "
            "sorry make that, change that to, change it to, correction"
        ),
        "example": "Let's meet at five PM, actually make that six thirty. -> Let's meet at six thirty.",
    },
    {
        "name": "clause_retraction",
        "triggers": "scratch that, forget that",
        "example": "Deploy on Friday, scratch that, deploy on Monday. -> Deploy on Monday.",
    },
    {
        "name": "parallel_verb_or_clause_reframing",
        "triggers": "actually <verb>, actually let's <verb>, instead <verb>, sorry <verb>",
        "example": "Use Redis—actually use PostgreSQL for persistence. -> Use PostgreSQL for persistence.",
    },
    {
        "name": "atomic_category_substitution",
        "triggers": "sorry, I mean, actually, rather, or rather, no wait",
        "example": "Send it tomorrow—sorry, Thursday. -> Send it Thursday.",
    },
]


# ---------------------------------------------------------------------------
# Lexical & Semantic Category Helpers
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "once", "twice", "half", "quarter",
}

_TIME_WORDS = {
    "am", "pm", "a.m.", "p.m.", "o'clock", "oclock", "noon", "midnight",
    "morning", "afternoon", "evening", "tonight",
}

_DATE_DAY_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "tomorrow", "yesterday", "weekend",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

_COMMON_VERBS = {
    "use", "using", "make", "create", "build", "send", "email", "message", "call",
    "meet", "schedule", "book", "deploy", "run", "install", "configure", "set",
    "update", "change", "write", "read", "fetch", "query", "store", "save",
    "switch", "migrate", "connect", "route", "assign", "add", "remove", "delete",
    "open", "close", "start", "stop", "launch", "push", "pull", "merge", "test",
    "order", "buy", "pick", "choose", "select", "go", "move", "wait", "ship",
}

_PREPOSITIONS = {
    "at", "on", "in", "to", "for", "with", "from", "by", "into", "using", "via", "until", "after", "before",
}

# Starters that indicate an independent conversational/commentary clause rather than an entity replacement
_COMMENTARY_CLAUSE_STARTERS = (
    "i am ", "i'm ", "i was ", "i have ", "i've ", "i had ", "i don't ", "i do not ",
    "i didn't ", "i cannot ", "i can't ", "i think ", "i thought ", "i wonder ",
    "i forgot ", "i remember ", "i guess ", "i suppose ", "i believe ", "i really ",
    "we are ", "we're ", "we were ", "we have ", "we've ", "we had ", "we don't ",
    "we didn't ", "we already ", "it is ", "it's ", "it was ", "there is ", "there's ",
    "that is ", "that's ", "this is ", "you know", "to be honest", "by the way",
    "never mind that", "what ", "why ", "how ", "when ", "where ", "who ",
)


def _singularize(word: str) -> str:
    """Simple singularizer for matching plural units ('columns' -> 'column')."""
    w = word.lower().strip()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes") or w.endswith("zes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        return w[:-1]
    return w


def _is_number_token(tok: str) -> bool:
    t = tok.lower().strip(".,!?-:")
    if not t:
        return False
    if re.match(r"^\d+(?:\.\d+)?%?$", t):
        return True
    if re.match(r"^\d+(?:st|nd|rd|th)$", t):
        return True
    return t in _NUMBER_WORDS


def _is_time_phrase(phrase: str) -> bool:
    """Return True if `phrase` looks like a spoken or numeric time expression."""
    tokens = [t.lower().strip(".,!?") for t in phrase.split() if t.strip(".,!?")]
    if not tokens or len(tokens) > 5:
        return False
    # e.g. "5:30", "6:30 pm", "five PM", "six thirty", "noon"
    if any(re.match(r"^\d{1,2}:\d{2}(?:am|pm)?$", t) for t in tokens):
        return True
    if any(t in _TIME_WORDS for t in tokens) and any(_is_number_token(t) or t in _TIME_WORDS for t in tokens):
        return True
    # Two consecutive number words representing clock time like "six thirty", "five forty five"
    if len(tokens) >= 2 and all(_is_number_token(t) for t in tokens):
        return True
    return False


def _is_date_or_day_phrase(phrase: str) -> bool:
    """Return True if `phrase` looks like a day of the week or date expression."""
    tokens = [t.lower().strip(".,!?") for t in phrase.split() if t.strip(".,!?")]
    if not tokens or len(tokens) > 4:
        return False
    if any(t in _DATE_DAY_WORDS for t in tokens):
        return True
    return False


def _is_numeric_phrase(phrase: str) -> bool:
    """Return True if `phrase` starts with or consists of a number/quantity."""
    tokens = [t.lower().strip(".,!?") for t in phrase.split() if t.strip(".,!?")]
    if not tokens or len(tokens) > 4:
        return False
    first_parts = tokens[0].split("-")
    return _is_number_token(first_parts[0])


def _is_concise_entity_or_term(phrase: str) -> bool:
    """Return True if `phrase` is a 1-3 word entity, name, or technical term (not a commentary clause or verb phrase)."""
    cleaned = phrase.strip().strip(".,!?")
    if not cleaned:
        return False
    lower = cleaned.lower()
    if any(lower.startswith(starter) for starter in _COMMENTARY_CLAUSE_STARTERS):
        return False
    tokens = cleaned.split()
    if len(tokens) > 3:
        return False
    first_low = tokens[0].lower().strip(".,!?")
    if len(tokens) > 1 and first_low in _COMMON_VERBS:
        return False
    # Avoid treating pure filler/pronoun/conjunction fragments as entity replacements
    non_entity_singletons = {
        "i", "we", "you", "they", "he", "she", "it", "that", "this", "there",
        "yes", "no", "maybe", "well", "so", "and", "but", "or", "if", "because",
    }
    if len(tokens) == 1 and first_low in non_entity_singletons:
        return False
    return True


def _preserve_initial_case(reference: str, replacement: str) -> str:
    """Capitalize the first character of `replacement` if `reference` started with an uppercase letter."""
    if not replacement:
        return replacement
    ref_stripped = reference.lstrip()
    rep_stripped = replacement.lstrip()
    if ref_stripped and ref_stripped[0].isupper() and rep_stripped and rep_stripped[0].islower():
        return rep_stripped[0].upper() + rep_stripped[1:]
    return rep_stripped


# ---------------------------------------------------------------------------
# Target Replacement Logic inside an Antecedent Clause
# ---------------------------------------------------------------------------

# Time suffix pattern at the end of an antecedent: e.g. "five PM", "5:30 pm", "six thirty", "noon"
_TRAILING_TIME_RE = re.compile(
    r"(?P<prefix>.*?\b(?:at|around|by|for|until|from|to)\s+)"
    r"(?P<target>"
    r"(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)"
    r"|(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"(?:\s+(?:ten|fifteen|twenty|thirty|forty|forty-five|fifty|o'clock|oclock))?"
    r"(?:\s*(?:a\.?m\.?|p\.?m\.?|in the morning|in the afternoon|in the evening))?)"
    r"|(?:noon|midnight)"
    r")$",
    re.IGNORECASE,
)

# Date/day suffix pattern at the end of an antecedent: e.g. "tomorrow", "on Monday", "Tuesday"
_TRAILING_DATE_RE = re.compile(
    r"(?P<prefix>.*?)"
    r"(?P<target>"
    r"(?:next\s+|this\s+)?"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|yesterday)"
    r"(?:\s+(?:morning|afternoon|evening|night))?"
    r"|(?:(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?)"
    r")$",
    re.IGNORECASE,
)

# Numeric modifier + unit + optional head noun:
# e.g. "Create a three-column table" or "Create a three column table"
_NUM_UNIT_MODIFIER_RE = re.compile(
    r"^(?P<prefix>.*?\b)"
    r"(?P<num>(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))"
    r"(?P<sep>[-\s]+)"
    r"(?P<unit>[a-zA-Z]+)"
    r"(?P<suffix>\s+[a-zA-Z0-9_-]+.*)$",
    re.IGNORECASE,
)

# Standard numeric phrase in antecedent: e.g. "Order 5 boxes", "timeout to 30 seconds"
_NUMERIC_TARGET_RE = re.compile(
    r"^(?P<prefix>.*?\b)"
    r"(?P<num>(?:\d+(?:\.\d+)?%?|"
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand))"
    r"(?P<unit_suffix>(?:\s+[a-zA-Z]+)?)$",
    re.IGNORECASE,
)


def _apply_replacement_to_antecedent(
    antecedent: str,
    replacement: str,
    is_explicit_directive: bool,
) -> Optional[str]:
    """Attempt to apply `replacement` to `antecedent`.

    Returns the merged corrected string if a clear, unambiguous target is found;
    otherwise returns None so the caller preserves the original text.
    """
    ant = antecedent.strip().rstrip(",;:-—–.")
    rep = replacement.strip()
    if not ant or not rep:
        return None

    # Save trailing punctuation from replacement if any
    trailing_punct = ""
    if rep and rep[-1] in ".!?":
        trailing_punct = rep[-1]
        rep_core = rep[:-1].strip()
    else:
        rep_core = rep

    if not rep_core:
        return None

    ant_words = ant.split()
    rep_words = rep_core.split()
    rep_first_lower = rep_words[0].lower().strip(".,!?")

    # 1. Clause-level re-framing: e.g. "I want to use React" + "let's use Vue" / "we should use Vue"
    clause_reframe_Match = re.match(
        r"^(?:let's|lets|let us|we should|we can|we will|we'll|i want to|i'd like to|i would like to)\s+([a-zA-Z]+)\b",
        rep_core,
        re.IGNORECASE,
    )
    if clause_reframe_Match:
        reframe_verb = clause_reframe_Match.group(1).lower()
        ant_lower_words = [w.lower().strip(".,!?") for w in ant_words]
        if reframe_verb in ant_lower_words or is_explicit_directive:
            prefix_match = re.match(
                r"^(?P<lead>.*?\b(?:and|but|so|then)\s+)?(?P<clause>(?:i\s+want\s+to|i'd\s+like\s+to|we\s+should|let's|lets|let\s+us)\s+.*)$",
                ant,
                re.IGNORECASE,
            )
            if prefix_match and prefix_match.group("lead"):
                lead = prefix_match.group("lead")
                return f"{lead}{rep_core}{trailing_punct}"
            return f"{_preserve_initial_case(ant, rep_core)}{trailing_punct}"

    # 2. Shared Verb or Preposition Parallelism:
    #    e.g. "Use Redis" + "use PostgreSQL for persistence"
    #    e.g. "Send the invoice to Mark" + "send it to Sarah"
    if rep_first_lower in _COMMON_VERBS or rep_first_lower in _PREPOSITIONS:
        pronoun_bridge = re.match(
            r"^(?P<verb>[a-zA-Z]+)\s+(?:it|them|that|this)\s+(?P<prep>to|for|in|on|with|at|into|from)\s+(?P<new_target>.+)$",
            rep_core,
            re.IGNORECASE,
        )
        if pronoun_bridge:
            v_low = pronoun_bridge.group("verb").lower()
            p_low = pronoun_bridge.group("prep").lower()
            new_t = pronoun_bridge.group("new_target")
            ant_pattern = re.compile(
                rf"^(?P<head>.*?\b{re.escape(v_low)}\b.+?\b{re.escape(p_low)}\s+)(?P<old_target>[^,;]+)$",
                re.IGNORECASE,
            )
            m_bridge = ant_pattern.match(ant)
            if m_bridge:
                return f"{m_bridge.group('head')}{new_t}{trailing_punct}"

        matches = list(re.finditer(rf"\b{re.escape(rep_first_lower)}\b", ant, re.IGNORECASE))
        if matches:
            last_m = matches[-1]
            prefix = ant[:last_m.start()]
            matched_word = ant[last_m.start():last_m.end()]
            adjusted_rep = _preserve_initial_case(matched_word, rep_core)
            return f"{prefix}{adjusted_rep}{trailing_punct}"

        # Instrumental synonym: "with <X>" / "using <X>" / "via <X>" / "into <X>" / "in <X>" / "to <X>" / "for <X>" corrected by "use <Y>" / "using <Y>"
        if rep_first_lower in ("use", "using") and len(rep_words) >= 2:
            new_tool = " ".join(rep_words[1:])
            m_inst = re.match(
                r"^(?P<prefix>.*?\b(?:with|using|via|into|in|to|for|on)\s+)(?P<old_tool>[A-Za-z0-9_.\-+/#]+(?:\s+[A-Za-z0-9_.\-+/#]+){0,2})$",
                ant,
                re.IGNORECASE,
            )
            if m_inst:
                return f"{m_inst.group('prefix')}{new_tool}{trailing_punct}"

    # 3. Numeric modifier + unit + head noun:
    #    e.g. "Create a three-column table" + "four columns" -> "Create a four-column table"
    m_hyph = _NUM_UNIT_MODIFIER_RE.match(ant)
    if m_hyph and _is_numeric_phrase(rep_core):
        unit = m_hyph.group("unit")
        sep = "-" if "-" in m_hyph.group("sep") else " "
        rep_tokens = rep_core.split()
        new_num = rep_tokens[0]
        if len(rep_tokens) == 1:
            return f"{m_hyph.group('prefix')}{new_num}{sep}{unit}{m_hyph.group('suffix')}{trailing_punct}"
        if len(rep_tokens) == 2 and _singularize(rep_tokens[1]) == _singularize(unit):
            return f"{m_hyph.group('prefix')}{new_num}{sep}{unit}{m_hyph.group('suffix')}{trailing_punct}"

    # 4. Time expression replacement:
    #    e.g. "Let's meet at five PM" + "six thirty" -> "Let's meet at six thirty"
    if _is_time_phrase(rep_core):
        m_time = _TRAILING_TIME_RE.match(ant)
        if m_time:
            return f"{m_time.group('prefix')}{rep_core}{trailing_punct}"

    # 5. Date / Day expression replacement:
    #    e.g. "Send it tomorrow" + "Thursday" -> "Send it Thursday"
    if _is_date_or_day_phrase(rep_core):
        m_date = _TRAILING_DATE_RE.match(ant)
        if m_date:
            return f"{m_date.group('prefix')}{rep_core}{trailing_punct}"

    # 6. Numeric / Quantity replacement:
    #    e.g. "Order 5 boxes" + "10" -> "Order 10 boxes"
    #    e.g. "Set timeout to 30 seconds" + "60 seconds" -> "Set timeout to 60 seconds"
    if _is_numeric_phrase(rep_core):
        m_num = _NUMERIC_TARGET_RE.match(ant)
        if m_num:
            prefix = m_num.group("prefix")
            unit_suffix = m_num.group("unit_suffix") or ""
            rep_tokens = rep_core.split()
            if len(rep_tokens) == 1 and unit_suffix:
                return f"{prefix}{rep_core}{unit_suffix}{trailing_punct}"
            return f"{prefix}{rep_core}{trailing_punct}"

    # 7. Concise Entity / Name / Technical Term Substitution:
    #    e.g. "Email John" + "David" -> "Email David"
    #    e.g. "Deploy to Kubernetes" + "Docker Swarm" -> "Deploy to Docker Swarm"
    if _is_concise_entity_or_term(rep_core):
        m_tail = re.match(
            r"^(?P<prefix>.*?\b(?:use|using|to|for|with|from|on|in|at|into|via|call|email|message|assign|ping|ask|tell|choose|select|pick|install|run|deploy)\s+)"
            r"(?P<target>[A-Za-z0-9_.\-+/#]+(?:\s+[A-Za-z0-9_.\-+/#]+){0,2})$",
            ant,
            re.IGNORECASE,
        )
        if m_tail:
            target = m_tail.group("target")
            if is_explicit_directive or _is_concise_entity_or_term(target):
                return f"{m_tail.group('prefix')}{rep_core}{trailing_punct}"

        if is_explicit_directive and len(ant_words) >= 2:
            n_replace = min(len(rep_words), max(1, len(ant_words) - 1))
            kept = " ".join(ant_words[:-n_replace])
            return f"{kept} {rep_core}{trailing_punct}"

    # 8. If explicit directive and replacement is a full sentence/clause, replace the antecedent clause
    if is_explicit_directive and len(rep_words) >= 4:
        return f"{_preserve_initial_case(ant, rep_core)}{trailing_punct}"

    return None


# ---------------------------------------------------------------------------
# Mid-Sentence Parenthetical Correction Pass
# ---------------------------------------------------------------------------

_MID_SENTENCE_PARENTHETICAL_RE = re.compile(
    r"\b(?P<lead_verb>email|message|call|ping|assign|ask|tell|send\s+to|talk\s+to|meet\s+with|use|using)\s+"
    r"(?P<old_entity>[A-Z][a-zA-Z0-9_.\-]*)"
    r"(?:\s*(?:—|–|--|-|,)\s*)"
    r"(?:no\s*,?\s*)?(?:i\s+mean|sorry|or\s+rather|actually\s+make\s+that|make\s+that)\s*,?\s*"
    r"(?P<new_entity>[A-Z][a-zA-Z0-9_.\-]*)"
    r"(?:\s*(?:—|–|--|-|,)\s*|\s+)"
    r"(?P<rest>(?:about|for|on|to|in|with|regarding|before|after|and|because)\b.+)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Clause-Level Correction Splitting & Iterative Resolution
# ---------------------------------------------------------------------------

# Detects a subsequent correction clause chained after the current replacement
_CHAINED_CORRECTION_TAIL_RE = re.compile(
    r"^(?P<head_rep>.+?)"
    r"(?P<tail>"
    r"(?:\s*(?:,|;|—|–|--|\.\.\.)\s*)"
    r"(?:(?:no|wait|oh)\s*,?\s*)?"
    r"(?:scratch\s+that|forget\s+that|actually|sorry|rather|instead|i\s+mean|make\s+that|make\s+it|change\s+(?:that|it)\s+to|correction)\b"
    r".*)$",
    re.IGNORECASE,
)

# 1. Clause retraction triggers ("scratch that", "forget that")
_RETRACTION_SPLIT_RE = re.compile(
    r"^(?P<antecedent>.+?)"
    r"(?:\s*(?:,|;|—|–|--|-|\.\.\.)\s*|\s+)"
    r"(?:no\s*,?\s*|wait\s*,?\s*|oh\s*,?\s*)?"
    r"(?:scratch\s+that|forget\s+that)"
    r"(?:\s*(?:,|;|—|–|--|-|:)\s*|\s+)"
    r"(?P<replacement>.+)$",
    re.IGNORECASE,
)

# 2. Explicit directive triggers ("make that", "make it", "change that to", "correction", "I mean")
_EXPLICIT_DIRECTIVE_RE = re.compile(
    r"^(?P<antecedent>.+?)"
    r"(?:\s*(?:,|;|—|–|--|-|\.\.\.)\s*|\s+)"
    r"(?P<marker>"
    r"(?:(?:no|wait|oh|um|uh)\s*,?\s*)?"
    r"(?:"
    r"(?:actually|sorry)\s*,?\s*(?:make\s+that|make\s+it|change\s+that\s+to|change\s+it\s+to)"
    r"|(?:no|wait)\s*,?\s*(?:make\s+that|make\s+it)"
    r"|make\s+that"
    r"|change\s+(?:that|it)\s+to"
    r"|correction\s*[:,]"
    r"|(?:no\s*,?\s*)?i\s+mean"
    r"|or\s+rather"
    r")"
    r")"
    r"\s*,?\s*"
    r"(?P<replacement>.+)$",
    re.IGNORECASE,
)

# 3. Soft context-dependent triggers ("actually", "sorry", "rather", "instead", "no wait")
_SOFT_CORRECTION_RE = re.compile(
    r"^(?P<antecedent>.+?)"
    r"(?P<sep>\s*(?:,|;|—|–|--|\.\.\.)\s*|\s+-\s+|\s+)"
    r"(?P<marker>"
    r"(?:no\s*,?\s*|oh\s*,?\s*|wait\s*,?\s*)?"
    r"(?:actually|sorry|rather|instead|no\s+wait)"
    r")"
    r"(?:\s*,\s*|\s+)"
    r"(?P<replacement>.+)$",
    re.IGNORECASE,
)


def _split_chained_tail(replacement: str) -> tuple[str, str]:
    """If `replacement` itself is followed by another chained correction clause, split it off."""
    m = _CHAINED_CORRECTION_TAIL_RE.match(replacement.strip())
    if m:
        return m.group("head_rep").strip(), m.group("tail")
    return replacement.strip(), ""


def _is_normal_non_correction_usage(antecedent: str, sep: str, marker: str, replacement: str) -> bool:
    """Return True if the marker is used in normal grammar rather than as a spoken self-correction."""
    ant_clean = antecedent.strip()
    rep_clean = replacement.strip()
    marker_low = marker.lower().strip()
    rep_low = rep_clean.lower()

    # 1. "instead of ..." is standard comparative grammar, never a self-correction
    if marker_low.endswith("instead") and rep_low.startswith("of "):
        return True

    # 2. "would rather ...", "had rather ...", "rather than ..."
    ant_last_word = ant_clean.split()[-1].lower().strip(".,!?") if ant_clean.split() else ""
    if marker_low == "rather":
        if ant_last_word in {"would", "id", "i'd", "we'd", "they'd", "he'd", "she'd", "had", "is", "was", "are", "were", "quite", "very"}:
            return True
        if rep_low.startswith("than "):
            return True

    # 3. "sorry for ...", "sorry about ...", "sorry to ...", "am sorry", "is sorry", "so sorry", "very sorry"
    if "sorry" in marker_low:
        if ant_last_word in {"am", "i'm", "im", "is", "are", "was", "were", "so", "very", "really", "truly", "feel"}:
            return True
        if rep_low.startswith(("for ", "about ", "to ", "that ", "if ")):
            return True

    # 4. "actually" preceded by subject pronoun or auxiliary verb without punctuation separator
    has_punct_sep = any(ch in sep for ch in (",", ";", "—", "–", "-", "."))
    if marker_low == "actually" and not has_punct_sep:
        if ant_last_word in {
            "i", "we", "you", "they", "he", "she", "it", "that", "this", "who",
            "am", "is", "are", "was", "were", "have", "has", "had",
            "do", "does", "did", "can", "could", "will", "would", "should", "may", "might",
            "not", "never", "always", "just", "really",
        }:
            return True

    # 5. Commentary / reflection clauses after "actually" or "sorry"
    if any(rep_low.startswith(starter) for starter in _COMMENTARY_CLAUSE_STARTERS):
        return True

    return False


def _resolve_single_sentence(sentence: str) -> str:
    """Iteratively resolve spoken corrections within a single sentence."""
    current = sentence.strip()
    if not current:
        return current

    # Pass 0: Mid-sentence parenthetical entity correction ("Email John—I mean David—about the contract.")
    m_mid = _MID_SENTENCE_PARENTHETICAL_RE.search(current)
    if m_mid:
        prefix = current[:m_mid.start()]
        reconstructed = (
            f"{prefix}{m_mid.group('lead_verb')} {m_mid.group('new_entity')} {m_mid.group('rest')}"
        )
        current = reconstructed.strip()

    # Iteratively resolve up to 5 chained corrections in the sentence
    for _ in range(5):
        changed = False

        # 1. Check full-clause retraction ("scratch that", "forget that")
        m_ret = _RETRACTION_SPLIT_RE.match(current)
        if m_ret:
            ant = m_ret.group("antecedent").strip()
            rep_raw = m_ret.group("replacement").strip()
            rep, tail = _split_chained_tail(rep_raw)
            if rep:
                targeted = _apply_replacement_to_antecedent(ant, rep, is_explicit_directive=True)
                if targeted:
                    current = f"{targeted}{tail}"
                else:
                    current = f"{_preserve_initial_case(ant, rep)}{tail}"
                changed = True
                continue

        # 2. Check soft or explicit correction in left-to-right order:
        #    If a soft correction appears BEFORE an explicit directive (e.g. "Meet Monday, sorry Tuesday, actually make that Wednesday"),
        #    resolve the earlier clause first.
        m_exp = _EXPLICIT_DIRECTIVE_RE.match(current)
        m_soft = _SOFT_CORRECTION_RE.match(current)

        # Try soft first if it starts earlier in the sentence than explicit
        if m_soft and (not m_exp or len(m_soft.group("antecedent")) < len(m_exp.group("antecedent"))):
            ant = m_soft.group("antecedent")
            sep = m_soft.group("sep")
            marker = m_soft.group("marker")
            rep_raw = m_soft.group("replacement")
            rep, tail = _split_chained_tail(rep_raw)
            if not _is_normal_non_correction_usage(ant, sep, marker, rep):
                resolved = _apply_replacement_to_antecedent(
                    ant, rep, is_explicit_directive=False
                )
                if resolved:
                    current = f"{resolved}{tail}"
                    changed = True
                    continue

        if m_exp:
            ant = m_exp.group("antecedent").strip()
            rep_raw = m_exp.group("replacement").strip()
            rep, tail = _split_chained_tail(rep_raw)
            resolved = _apply_replacement_to_antecedent(ant, rep, is_explicit_directive=True)
            if resolved:
                current = f"{resolved}{tail}"
                changed = True
                continue

        if m_soft:
            ant = m_soft.group("antecedent")
            sep = m_soft.group("sep")
            marker = m_soft.group("marker")
            rep_raw = m_soft.group("replacement")
            rep, tail = _split_chained_tail(rep_raw)
            if not _is_normal_non_correction_usage(ant, sep, marker, rep):
                resolved = _apply_replacement_to_antecedent(
                    ant, rep, is_explicit_directive=False
                )
                if resolved:
                    current = f"{resolved}{tail}"
                    changed = True
                    continue

        if not changed:
            break

    return current


def resolve_spoken_corrections(text: str) -> str:
    """Resolve natural spoken self-corrections across a complete session transcript.

    Preserves normal uses of "actually", "sorry", "rather", and "instead", and
    leaves ambiguous phrasing untouched.
    """
    if not text or not text.strip():
        return text

    # Split into sentence-level units while preserving paragraph breaks
    paragraphs = text.split("\n")
    resolved_paragraphs: list[str] = []

    for para in paragraphs:
        if not para.strip():
            resolved_paragraphs.append(para)
            continue

        # Split on sentence boundaries (. ! ?) followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        resolved_sentences = [_resolve_single_sentence(s) for s in sentences if s.strip()]
        resolved_paragraphs.append(" ".join(resolved_sentences))

    return "\n".join(resolved_paragraphs)
