"""
evidence_scope_guard.py — Task 69 Deterministic Evidence-Scope Guard Prototype

An isolated, deterministic, explainable filter for two persistent false-positive patterns:
1. Query-echo questions incorrectly retained as answers.
2. Explicit development/staging/local/test evidence incorrectly retained for production queries
   (and vice-versa).

EVALUATION & PROTOTYPE ONLY:
This module does NOT modify the production LangGraph state machine, rag_graph.py,
or any production grader/generator prompts.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class GuardDecision(str, Enum):
    REJECT_QUERY_ECHO = "REJECT_QUERY_ECHO"
    REJECT_ENVIRONMENT_MISMATCH = "REJECT_ENVIRONMENT_MISMATCH"
    PASS = "PASS"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class GuardResult:
    decision: GuardDecision
    rule: str
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "rule": self.rule,
            "explanation": self.explanation,
            "details": self.details,
        }


# Environment recognition patterns
ENV_PATTERNS: Dict[str, List[str]] = {
    "production": [
        r"\b(?:production|prod)\b",
        r"secrets[\._-]prod[\._-]json",
        r"production\.(?:yaml|yml|json|env|conf)",
        r"\.env\.prod(?:uction)?\b",
    ],
    "staging": [
        r"\b(?:staging|stage)\b",
        r"staging\.env",
        r"secrets[\._-]staging[\._-]json",
        r"staging\.(?:yaml|yml|json|env|conf)",
        r"\.env\.stag(?:e|ing)\b",
    ],
    "development": [
        r"\bdevelopment\b",
        r"\bdev\s+(?:env(?:ironment)?|cluster|setup|tier|secrets?|branch|conf(?:ig(?:uration)?)?|db|database|server|host)\b",
        r"\b(?:env(?:ironment)?|cluster|setup|tier|branch|config(?:uration)?)\s*[:=]?\s*dev\b",
        r"\bdev\s+(?:and|or|vs|versus|to)\s+(?:prod(?:uction)?|staging|test|local)\b",
        r"\b(?:prod(?:uction)?|staging|test|local)\s+(?:and|or|vs|versus|to)\s+dev\b",
        r"secrets[\._-]dev[\._-]json",
        r"dev\.(?:yaml|yml|json|env|conf)",
        r"\.env\.dev(?:elopment)?\b",
    ],
    "local": [
        r"\b(?:local|localhost)\b",
        r"local\.env",
        r"secrets[\._-]local[\._-]json",
        r"\.env\.local\b",
    ],
    "test": [
        r"\b(?:testing|test)\b",
        r"test\.env",
        r"secrets[\._-]test[\._-]json",
        r"\.env\.test\b",
    ],
}

STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "for",
    "to", "of", "did", "we", "you", "i", "our", "us", "do", "does", "it",
    "this", "that", "with", "by", "from", "can", "could", "would", "should",
    "have", "has", "had", "be", "been", "being", "my", "your", "their", "his",
    "her", "its", "and", "or", "but", "if", "as", "so", "then", "there", "what",
    "who", "where", "when", "which", "how", "why",
}


def extract_content_keywords(text: str) -> Set[str]:
    """Extracts lowercase alphabetic content words (length > 1, non-stopword)."""
    if not text:
        return set()
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def extract_environments(text: str, is_candidate: bool = False) -> Set[str]:
    """
    Extracts explicit environment tokens from text.
    When is_candidate=True, strips sender prefixes (e.g., 'Dev_Security:', 'Dev_Ops:')
    so that developer role usernames do NOT trigger false 'development' detection.
    """
    if not text:
        return set()

    clean_text = text
    if is_candidate:
        # Strip sender prefixes like "Dev_Security:" or "[ID 123] ... Dev_Test:"
        clean_text = re.sub(
            r"(?:\[ID\s+\d+\]\s*)?(?:\[[^\]]+\]\s*)?[A-Za-z0-9_]+:\s*", " ", text
        )
        # Strip header brackets like [RETRIEVED ANCHOR], [CHRONOLOGICAL CONTEXT], [Candidate 1]
        clean_text = re.sub(
            r"\[(?:RETRIEVED ANCHOR|CHRONOLOGICAL CONTEXT|Candidate \d+)\]",
            " ",
            clean_text,
        )

    found: Set[str] = set()
    for env_name, patterns in ENV_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, clean_text, re.IGNORECASE):
                found.add(env_name)
                break
    return found


def parse_candidate_messages(candidate_text: str) -> List[Dict[str, Any]]:
    """
    Parses dialogue-unit text into structured messages.
    Preserves message order and extracts sender, id, and clean text.
    """
    if not candidate_text:
        return []

    lines = candidate_text.strip().splitlines()
    context_lines: List[str] = []
    has_context_section = any("[CHRONOLOGICAL CONTEXT]" in line for line in lines)

    if has_context_section:
        in_context = False
        for line in lines:
            if "[CHRONOLOGICAL CONTEXT]" in line:
                in_context = True
                continue
            if in_context:
                if (
                    line.startswith("[")
                    and not line.startswith("[ID ")
                    and not line.startswith("[Candidate")
                ):
                    in_context = False
                else:
                    context_lines.append(line)
    else:
        # Standalone anchor or raw candidate without [CHRONOLOGICAL CONTEXT]
        has_anchor_section = any("[RETRIEVED ANCHOR]" in line for line in lines)
        if has_anchor_section:
            in_anchor = False
            for line in lines:
                if "[RETRIEVED ANCHOR]" in line:
                    in_anchor = True
                    continue
                if in_anchor:
                    # Stop if another major section begins (e.g. metadata or other bracket header)
                    if (
                        line.startswith("[")
                        and not line.startswith("[ID ")
                        and not line.startswith("[Candidate")
                    ):
                        in_anchor = False
                    else:
                        context_lines.append(line)
        else:
            context_lines = [
                l for l in lines
                if not l.strip().startswith("---") and not l.strip().startswith("[Candidate")
            ]

    msg_pattern = re.compile(
        r"^(?:\[ID\s+(?P<id>\d+)\]\s*)?"
        r"(?:\[[^\]]+\]\s*)?"
        r"(?:(?P<sender>[A-Za-z0-9_\s]+?):\s+)?"
        r"(?P<body>.*)$"
    )

    messages: List[Dict[str, Any]] = []
    current_msg: Optional[Dict[str, Any]] = None

    for line in context_lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("---") or line_s.startswith("[Candidate"):
            continue
        m = msg_pattern.match(line_s)
        if m and (m.group("id") is not None or m.group("sender") is not None):
            msg_id = int(m.group("id")) if m.group("id") is not None else None
            sender = (
                m.group("sender").strip() if m.group("sender") is not None else None
            )
            body = m.group("body").strip() if m.group("body") is not None else ""
            current_msg = {
                "id": msg_id,
                "sender": sender,
                "text": body,
                "raw_line": line_s,
            }
            messages.append(current_msg)
        else:
            if current_msg is not None:
                current_msg["text"] += " " + line_s
            else:
                messages.append(
                    {
                        "id": None,
                        "sender": None,
                        "text": line_s,
                        "raw_line": line_s,
                    }
                )

    return messages


def is_interrogative_message(text: str) -> bool:
    """Detects if a message is an inquiry / question."""
    if not text:
        return False
    t = text.strip()
    if "?" in t:
        return True
    lower = t.lower()
    words = lower.split()
    if not words:
        return False
    first = words[0]
    first_two = " ".join(words[:2]) if len(words) >= 2 else ""
    starters = {
        "who", "what", "where", "when", "which", "how", "why",
        "did", "does", "do", "is", "are", "was", "were",
        "can", "could", "would", "should", "has", "have",
    }
    if first in starters:
        return True
    if first_two in {
        "has anyone", "can someone", "anyone know", "any idea", "does anyone"
    }:
        return True
    return False


def is_plausible_answering_reply(reply_text: str, query_keywords: Set[str]) -> bool:
    """
    Evaluates whether a subsequent message in dialogue context could plausibly be
    an answer or answer support for an earlier question.
    """
    if not reply_text:
        return False
    text = reply_text.strip()
    lower = text.lower()

    # 1. Affirmations, negatives, completion status
    if re.search(
        r"\b(?:yes|no|yep|nope|yeah|sure|done|clean|confirmed|true|false|succeeded|failed|passed|maybe|probably|completed)\b",
        lower,
    ):
        return True

    # 2. Time expressions
    if re.search(
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}\s*pm\b|\b\d{1,2}\s*am\b|\b(?:tonight|tomorrow|today|yesterday)\b|\b\d{1,2}\s*(?:hours|mins|minutes)\b",
        lower,
    ):
        return True

    # 3. Location / storage / container patterns
    if re.match(r"^(?:in|at|on|under|inside|by|from|to|near)\b", lower):
        return True
    if re.match(
        r"^(?:it|that|they|this)\s+(?:is|was|are|were|\'s)\s+(?:in|at|on|under|inside|located|stored|saved|placed|found)\b",
        lower,
    ):
        return True
    if re.search(
        r"\b(?:folder|drawer|vault|counter|rack|server|cluster|repo|bucket|directory|file|path|disk|database|db)\b",
        lower,
    ):
        return True
    if re.search(
        r"\b(?:stored|saved|placed|left|moved|shared|put|found|located)\b", lower
    ):
        return True

    # 4. Identification / technical entities
    if re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", lower):  # IP address
        return True
    if re.search(r"secrets[\._-][a-z0-9_-]+\.[a-z0-9]+", lower):
        return True

    # 5. Overlap with query topic keywords
    reply_kw = extract_content_keywords(text)
    if reply_kw.intersection(query_keywords):
        return True

    return False


def evaluate_evidence_scope(
    query: str,
    candidate_text: str,
    anchor_message_id: Any = None,
    included_message_ids: Optional[List[int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> GuardResult:
    """
    Evaluates candidate against deterministic scope guard:
    Returns GuardResult with decision, rule, explanation, and details.
    """
    details: Dict[str, Any] = {
        "anchor_message_id": anchor_message_id,
        "included_message_ids": included_message_ids or [],
        "query_envs": [],
        "cand_envs": [],
        "has_query_echo": False,
        "has_answer_reply": False,
    }

    if not candidate_text or not candidate_text.strip():
        return GuardResult(
            decision=GuardDecision.PASS,
            rule="rule_empty_candidate",
            explanation="Candidate text is empty; passed without deterministic rejection.",
            details=details,
        )

    # -------------------------------------------------------------
    # 1. Environment-Scope Mismatch Guard
    # -------------------------------------------------------------
    q_envs = extract_environments(query, is_candidate=False)
    c_envs = extract_environments(candidate_text, is_candidate=True)
    details["query_envs"] = sorted(list(q_envs))
    details["cand_envs"] = sorted(list(c_envs))

    if q_envs:
        # Check if candidate mentions environments
        if c_envs:
            # Rule 3: If candidate explicitly establishes requested environment, do not reject
            intersection = q_envs.intersection(c_envs)
            if intersection:
                # Requested environment is present in candidate
                # Check for ambiguous conflicting wording (Rule 4)
                if re.search(
                    r"\b(?:between|or|vs|versus|whether|migrating|transitioning)\s+[a-z0-9_\s]+\b(?:and|or)\b",
                    candidate_text,
                    re.IGNORECASE,
                ) and len(c_envs) > 1:
                    return GuardResult(
                        decision=GuardDecision.NEEDS_REVIEW,
                        rule="rule_env_ambiguous_wording",
                        explanation=(
                            f"Candidate mentions multiple environments {sorted(list(c_envs))} "
                            f"with ambiguous relationship to query environment {sorted(list(q_envs))}."
                        ),
                        details=details,
                    )
                # Otherwise, requested environment is established
                pass
            else:
                # Candidate has environment mentions, but NONE match the query
                non_prod = {"development", "staging", "local", "test"}
                if "production" in q_envs and c_envs.issubset(non_prod):
                    return GuardResult(
                        decision=GuardDecision.REJECT_ENVIRONMENT_MISMATCH,
                        rule="rule_env_mismatch_prod_vs_nonprod",
                        explanation=(
                            f"Query explicitly requested 'production' environment, but candidate provides "
                            f"only {sorted(list(c_envs))} evidence."
                        ),
                        details=details,
                    )
                elif "development" in q_envs and c_envs.issubset({"production", "staging"}):
                    return GuardResult(
                        decision=GuardDecision.REJECT_ENVIRONMENT_MISMATCH,
                        rule="rule_env_mismatch_dev_vs_prod",
                        explanation=(
                            f"Query explicitly requested 'development' environment, but candidate provides "
                            f"only {sorted(list(c_envs))} evidence."
                        ),
                        details=details,
                    )
                elif "staging" in q_envs and c_envs.issubset({"production", "development", "local"}):
                    return GuardResult(
                        decision=GuardDecision.REJECT_ENVIRONMENT_MISMATCH,
                        rule="rule_env_mismatch_staging_vs_other",
                        explanation=(
                            f"Query explicitly requested 'staging' environment, but candidate provides "
                            f"only {sorted(list(c_envs))} evidence."
                        ),
                        details=details,
                    )
                else:
                    return GuardResult(
                        decision=GuardDecision.NEEDS_REVIEW,
                        rule="rule_env_unclear_relationship",
                        explanation=(
                            f"Candidate mentions environments {sorted(list(c_envs))} "
                            f"conflicting with query {sorted(list(q_envs))}."
                        ),
                        details=details,
                    )

    # -------------------------------------------------------------
    # 2. Query-Echo Detection Guard
    # -------------------------------------------------------------
    msgs = parse_candidate_messages(candidate_text)
    q_kw = extract_content_keywords(query)

    if not msgs or not q_kw:
        return GuardResult(
            decision=GuardDecision.PASS,
            rule="rule_default_pass",
            explanation="Candidate passed all deterministic guards.",
            details=details,
        )

    echo_msgs: List[Tuple[int, Dict[str, Any]]] = []
    other_msgs: List[Tuple[int, Dict[str, Any]]] = []

    for idx, m in enumerate(msgs):
        m_text = m["text"]
        m_kw = extract_content_keywords(m_text)
        overlap = q_kw.intersection(m_kw)
        overlap_ratio = len(overlap) / len(q_kw) if q_kw else 0.0

        # An echo message is an interrogative message with strong query overlap
        if is_interrogative_message(m_text) and (
            overlap_ratio >= 0.70 or (len(overlap) >= 3 and overlap_ratio >= 0.5)
        ):
            echo_msgs.append((idx, m))
        else:
            other_msgs.append((idx, m))

    if echo_msgs:
        details["has_query_echo"] = True
        has_plausible_answer = False
        unrelated_declarative_count = 0

        for echo_idx, echo_m in echo_msgs:
            subsequent_msgs = [m for idx, m in other_msgs if idx > echo_idx]
            if not subsequent_msgs:
                continue

            for sub_m in subsequent_msgs:
                sub_text = sub_m["text"].strip()
                if is_plausible_answering_reply(sub_text, q_kw):
                    has_plausible_answer = True
                    break
                else:
                    unrelated_declarative_count += 1

        if has_plausible_answer:
            details["has_answer_reply"] = True
            return GuardResult(
                decision=GuardDecision.PASS,
                rule="rule_query_echo_with_answer",
                explanation="Candidate contains query-echo question but also includes a plausible answering reply.",
                details=details,
            )

        if unrelated_declarative_count > 0 and len(echo_msgs) >= 1 and not has_plausible_answer:
            return GuardResult(
                decision=GuardDecision.REJECT_QUERY_ECHO,
                rule="rule_query_echo_unanswered_banter",
                explanation=(
                    f"Candidate asks the query question '{echo_msgs[0][1]['text']}' "
                    f"without receiving an answer (subsequent message(s) are unrelated chatter)."
                ),
                details=details,
            )
        elif not any(idx > echo_msgs[0][0] for idx, m in other_msgs):
            return GuardResult(
                decision=GuardDecision.REJECT_QUERY_ECHO,
                rule="rule_query_echo_terminal",
                explanation=f"Candidate ends with unanswered query question '{echo_msgs[0][1]['text']}'.",
                details=details,
            )
        else:
            return GuardResult(
                decision=GuardDecision.NEEDS_REVIEW,
                rule="rule_query_echo_ambiguous_context",
                explanation="Candidate contains an echoed question with ambiguous subsequent messages.",
                details=details,
            )

    return GuardResult(
        decision=GuardDecision.PASS,
        rule="rule_default_pass",
        explanation="Candidate passed all deterministic guards.",
        details=details,
    )
