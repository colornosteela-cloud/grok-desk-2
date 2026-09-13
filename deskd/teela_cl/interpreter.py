"""Goal interpretation. Default is record-backed; tests/production may inject Qwen."""

from __future__ import annotations

import re
from typing import Any, Callable, Protocol

from .records import CompiledSkill, InterpretedIntent
from .self_model import SelfModel


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the me my your you can could would will please to do at with that this little just try trying perform show".split()
)


def _norm(text: str) -> str:
    return " ".join(_WORD.findall((text or "").lower()))


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 1}


def phrase_score(request: str, goal: str) -> float:
    r, g = _norm(request), _norm(goal)
    if not g:
        return 0.0
    if g == r:
        return 1.0
    if re.search(rf"(?:^| ){re.escape(g)}(?:$| )", r):
        return 1.0
    rt, gt = _tokens(r), _tokens(g)
    if not gt:
        return 0.0
    overlap = len(rt & gt)
    if overlap == 0:
        return 0.0
    score = overlap / len(gt)
    if overlap >= 2:
        score = max(score, 0.5)
    return score


class GoalInterpreter(Protocol):
    def __call__(
        self,
        request: str,
        skills: list[CompiledSkill],
        self_model: SelfModel,
    ) -> InterpretedIntent: ...


def _affiliative_comment(request: str) -> bool:
    """Backchannel / evaluation of what Teela just said — not a body request."""
    t = _norm(request)
    if not t:
        return False
    if re.match(r"^(?:that|this|it) (?:sounds|looks|seems|feels)\b", t):
        return True
    if re.match(r"^(?:sounds|looks|seems|feels) (?:nice|good|great|fine|fun|cool|lovely|ok|okay)\b", t):
        return True
    if re.match(
        r"^(?:pretty |really |so |very |thats |that s )?(?:nice|cool|good|great|awesome|lovely|interesting|okay|ok|alright|wonderful)$",
        t,
    ):
        return True
    if t in {
        "got it",
        "makes sense",
        "good to hear",
        "glad to hear",
        "fair enough",
        "no worries",
        "sounds good",
        "sounds great",
        "sounds fun",
        "thats nice",
        "that s nice",
        "nice one",
    }:
        return True
    return False


def _talk(request: str) -> bool:
    t = _norm(request)
    if not t:
        return True
    if t in {"hi", "hello", "hey", "thanks", "thank you", "good morning", "good night"}:
        return True
    if t.startswith("how are you") or t.startswith("what model"):
        return True
    if _affiliative_comment(request):
        return True
    # Classify the requested output before looking for capability words in it.
    # Politeness does not turn a request for speech into a body request.
    speech = re.sub(
        r"^(?:(?:can|could|will|would) you |please |you (?:can|should|need to) |go ahead and )+",
        "", t,
    )
    if re.match(r"^(?:tell|explain|describe|discuss|summarize|answer|chat|talk)\b", speech):
        return True
    if re.match(r"^(?:say|greet)\b", speech):
        # A greeting can explicitly request an embodied means of expression.
        return not re.search(r"\b(?:physically|with (?:your|a|the) (?:hand|body|arm)|gesture)\b", speech)
    return False


def _information_request(text: str) -> bool:
    return bool(re.match(r"^(?:what|what s|why|how|when|where|who|which)\b", _norm(text)))


_REQUEST_PREFIX = (
    "can you",
    "could you",
    "will you",
    "would you",
    "please",
    "try ",
    "you can ",
    "you should ",
    "you need to ",
    "go ahead and",
)
_STATE_COMMENT_PREFIX = (
    "you are",
    "you're",
    "you currently",
    "i see you",
    "that is",
    "that's",
    "that sounds",
    "this sounds",
    "it sounds",
    "i noticed",
    "looks like you",
)


_GREETING_PREFIX = re.compile(
    r"^(?:hi|hello|hey|yo|thanks|thank you|good morning|good night|good evening)\b",
    re.I,
)
_REPEAT_RE = re.compile(
    r"\b(?:do (?:that|it)(?: movement)? again|do what you just did|"
    r"same thing as before|that movement again|do that movement again|"
    r"same as before|like before)\b",
    re.I,
)


def is_repeat_request(text: str) -> bool:
    """Speech-act: ask to repeat the prior outcome. No skill names."""
    t = (text or "").strip()
    if not t or _talk(t) or _information_request(t):
        return False
    return bool(_REPEAT_RE.search(t))


def is_performance_request(text: str) -> bool:
    """True when the utterance asks Teela to do something, not to chat or comment.

    Speech-act only. No user action names.
    """
    raw = (text or "").strip()
    if not raw or _talk(raw) or _information_request(raw):
        return False
    low = raw.lower()
    if _GREETING_PREFIX.match(low) and len(_WORD.findall(low)) <= 3:
        return False
    if any(low.startswith(p) for p in _STATE_COMMENT_PREFIX):
        return False
    if is_repeat_request(raw):
        return True
    if any(low.startswith(p) for p in _REQUEST_PREFIX):
        return True
    if re.search(r"\b(?:can|could|will|would)\s+you\b", low):
        return True
    if raw.endswith("?") and not low.startswith(("can you", "could you", "will you", "try")):
        return False
    try:
        import robot_sim

        if robot_sim.looks_like_motor(raw) or robot_sim.infer_command(raw):
            return True
    except Exception:
        pass
    try:
        import virtual_body

        if virtual_body.teela_args_from_intent(raw):
            return True
    except Exception:
        pass
    n = len(_WORD.findall(low))
    if 1 <= n <= 6 and not low.startswith(("i ", "we ", "it ", "they ")):
        return True
    return False


def record_backed_interpreter(
    request: str,
    skills: list[CompiledSkill],
    self_model: SelfModel,
    *,
    threshold: float = 0.85,
) -> InterpretedIntent:
    """Match the request against skill records and self-model capability descriptions.

    Contains no user-action special cases. Skill *data* supplies paraphrases.
    Unknown software vs embodied is scored against self-model facts, not action names.
    """
    text = (request or "").strip()
    if _talk(text) or (_GREETING_PREFIX.match(text) and len(_WORD.findall(text)) <= 3):
        return InterpretedIntent(
            goal="converse",
            required_capabilities=[],
            domain="talk",
            expected_outcome="spoken reply",
            confidence=1.0,
        )
    if _tokens(text) & {"why", "explain", "details"}:
        return InterpretedIntent(
            goal="converse",
            required_capabilities=[],
            domain="talk",
            expected_outcome="spoken explanation",
            confidence=0.8,
        )
    vis = _tokens("see look observe camera vision scene")
    _head_motion = re.search(
        r"\b(?:look|turn|face|pan|tilt)\b.{0,28}\b(?:left|right|up|down|straight|ahead|at me)\b|"
        r"\bhead\b.{0,16}\b(?:left|right|up|down)\b",
        text,
        re.I,
    )
    if "?" in text and (_tokens(text) & vis) and not _head_motion:
        return InterpretedIntent(
            goal="observe_scene",
            required_capabilities=[],
            domain="perception",
            expected_outcome="structured observation",
            confidence=0.7,
        )
    # Never retrieve an executable body skill from words mentioned in a
    # question or state report. Perception questions above remain observable.
    if _information_request(text) or any(text.lower().startswith(p) for p in _STATE_COMMENT_PREFIX):
        return InterpretedIntent(
            goal="converse", required_capabilities=[], domain="talk",
            expected_outcome="spoken reply", confidence=0.8,
        )
    toks = _tokens(text)
    if toks <= {"again", "it", "that", "more", "once", "retry", "thing", "stuff", "this", "do"} or not toks:
        return InterpretedIntent(
            goal="unclear",
            required_capabilities=[],
            domain="software",
            expected_outcome="clarification",
            confidence=0.2,
            notes="Goal is too underspecified to execute or compose.",
        )
    best: CompiledSkill | None = None
    score = 0.0
    for sk in skills:
        if sk.kind == "primitive":
            continue
        local = 0.0
        for g in list(sk.supported_goals) + [sk.semantic_description, sk.name]:
            local = max(local, phrase_score(text, g))
        if local > score:
            best, score = sk, local
    if best is not None and score >= threshold:
        return InterpretedIntent(
            goal=best.skill_id,
            required_capabilities=list(best.required_capabilities or [best.skill_id]),
            domain="embodied" if best.tool else "software",
            expected_outcome=best.expected_outcome or best.semantic_description,
            confidence=score,
        )
    sw_score, sw_id = 0.0, ""
    for fact in list(self_model.minios_capabilities) + list(self_model.software_tools):
        s = phrase_score(text, fact.description + " " + fact.cap_id.replace(".", " "))
        if s > sw_score:
            sw_score, sw_id = s, fact.cap_id
    body_score, body_ids = 0.0, []
    for fact in self_model.primitives:
        if not fact.available:
            continue
        s = phrase_score(text, fact.description + " " + fact.cap_id.replace(".", " "))
        if s >= 0.2:
            body_ids.append(fact.cap_id)
        if s > body_score:
            body_score = s
    if sw_score >= 0.3 and sw_score >= body_score:
        return InterpretedIntent(
            goal=sw_id or "software.task",
            required_capabilities=[sw_id] if sw_id else [],
            domain="software",
            expected_outcome="computer/desktop/service outcome",
            confidence=sw_score,
        )
    if body_score >= 0.2 and body_ids:
        return InterpretedIntent(
            goal="composed_outcome",
            required_capabilities=body_ids,
            domain="embodied",
            expected_outcome=text,
            confidence=body_score,
            notes="No validated skill; compose from matching primitives.",
        )
    if sw_score >= 0.15:
        return InterpretedIntent(
            goal="unknown_outcome",
            required_capabilities=[],
            domain="software",
            expected_outcome=text,
            confidence=sw_score,
            notes="No matching software procedure; research or ask.",
        )
    low = text.lower()
    asking = text.endswith("?") and not low.startswith(("can you", "could you", "will you", "try"))
    if asking:
        # Information questions are not unmatched motor skills.
        return InterpretedIntent(
            goal="unknown_outcome",
            required_capabilities=[],
            domain="software",
            expected_outcome=text,
            confidence=max(sw_score, 0.2),
            notes="Question is not an embodied skill request.",
        )
    if not is_performance_request(text):
        return InterpretedIntent(
            goal="converse",
            required_capabilities=[],
            domain="talk",
            expected_outcome="spoken reply",
            confidence=0.8,
            notes="Commentary or greeting, not a request to act.",
        )
    if is_repeat_request(text):
        return InterpretedIntent(
            goal="repeat_last",
            required_capabilities=[],
            domain="embodied",
            expected_outcome="repeat the prior outcome",
            confidence=0.8,
            notes="Speech-act: repeat the last successful skill.",
        )
    # Unmatched: do not dump every locomotion primitive. Missing knowledge → learn/ask.
    return InterpretedIntent(
        goal="unknown_outcome",
        required_capabilities=[],
        domain="embodied",
        expected_outcome=text,
        confidence=0.1,
        notes="No skill or primitive description matched; learning required.",
    )


def make_injected_interpreter(mapping: dict[str, InterpretedIntent]) -> GoalInterpreter:
    """Test/prod injection: map exact request strings (or a default) to intent."""

    def _inner(request: str, skills: list[CompiledSkill], self_model: SelfModel) -> InterpretedIntent:
        if request in mapping:
            return mapping[request]
        key = _norm(request)
        for k, v in mapping.items():
            if _norm(k) == key:
                return v
        return record_backed_interpreter(request, skills, self_model)

    return _inner
