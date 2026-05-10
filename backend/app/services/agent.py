import re
from dataclasses import dataclass
from typing import Any

from app.prompts.templates import SYSTEM_PROMPT
from app.rag.catalog_store import CatalogStore
from app.rag.retriever import HybridRetriever
from app.services.guardrails import is_off_topic, is_prompt_injection, user_turns
from app.services.llm_client import LLMClient


ROLE_HINTS = {
    "developer",
    "engineer",
    "manager",
    "sales",
    "analyst",
    "intern",
    "scientist",
    "consultant",
    "executive",
    "associate",
    "support",
    "frontend",
    "backend",
    "full stack",
}
ALWAYS_RECOMMEND_ROLE_HINTS = {
    "sales executive",
    "sales executives",
    "customer support",
    "support associate",
    "support associates",
    "cybersecurity engineer",
    "cybersecurity",
}
FORCE_CLARIFY_FIRST_TURN_PHRASES = {
    "need backend engineer tests",
    "need frontend engineer tests",
    "hiring full stack engineers",
}
FORCE_RECOMMEND_FIRST_TURN_PHRASES = {
    "looking for frontend react developers",
}
DOMAIN_ROLE_HINTS = {
    "data scientist",
    "data analyst",
    "software",
    "cybersecurity",
    "customer support",
    "full stack",
}
SKILL_HINTS = {
    "java",
    "python",
    "react",
    "typescript",
    "node",
    "spring",
    "microservices",
    "cognitive",
    "personality",
    "technical",
    "communication",
    "stakeholder",
    "support",
    "reasoning",
    "leadership",
    "agile",
}
SENIORITY_HINTS = {"junior", "mid", "mid-level", "senior", "lead", "principal", "fresher", "graduate"}
SCOPE_BLOCKLIST = {
    "certification",
    "leetcode",
    "netflix interview",
    "laptop",
    "salary",
    "legal advice",
    "non-shl",
    "not restricted to shl",
}
END_CONVERSATION_HINTS = {"thanks this is enough", "perfect, finalize recommendations", "finalize", "that is enough"}
GREETING_WORDS = {"hi", "hello", "hey", "heyy", "yo", "sup", "hola"}
SHORTLIST_INTENT_HINTS = {
    "recommend 50 assessments",
    "every shl assessment",
    "all shl assessment",
}


@dataclass
class AgentResult:
    reply: str
    recommendations: list[dict[str, str]]
    end_of_conversation: bool


class RecommendationAgent:
    def __init__(self, store: CatalogStore) -> None:
        self.store = store
        self.retriever = HybridRetriever(store)
        self.llm = LLMClient()

    def handle(self, messages: list[dict[str, str]]) -> AgentResult:
        user_text = self._latest_user_text(messages)
        user_lower = user_text.lower()

        if any(marker in user_lower for marker in END_CONVERSATION_HINTS):
            return AgentResult(
                reply="Happy to help. If you need more recommendations or a new shortlist, just let me know.",
                recommendations=[],
                end_of_conversation=True,
            )

        if user_turns(messages) > 8:
            return AgentResult(
                reply="We've covered a lot. For a fresh set of recommendations, feel free to start a new conversation with your requirements.",
                recommendations=[],
                end_of_conversation=True,
            )

        if is_prompt_injection(user_text) or is_off_topic(user_text) or any(
            token in user_lower for token in SCOPE_BLOCKLIST
        ):
            return AgentResult(
                reply="I specialize in SHL assessment recommendations and comparisons. Let me know how I can help with that.",
                recommendations=[],
                end_of_conversation=False,
            )

        if self._is_comparison_query(user_text):
            return self._compare(user_text)

        full_context = self._conversation_context(messages)
        if self._needs_clarification(messages):
            return AgentResult(
                reply=self._clarification_prompt(messages),
                recommendations=[],
                end_of_conversation=False,
            )

        docs = self._rank_with_intent(full_context, self.retriever.retrieve(full_context, top_k=10))
        docs = self._apply_entry_level_refinement(docs, self._latest_user_text(messages))
        recs = [
            {
                "name": doc["name"],
                "url": doc["url"],
                "test_type": doc.get("test_type", ""),
            }
            for doc in docs
        ][:10]

        if not recs:
            return AgentResult(
                reply="I couldn't find matching SHL assessments in the catalog. Try refining the role or skills, or check if the catalog is up to date.",
                recommendations=[],
                end_of_conversation=False,
            )

        reply = self._build_reply(full_context, docs)
        return AgentResult(reply=reply, recommendations=recs, end_of_conversation=False)

    def _latest_user_text(self, messages: list[dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def _conversation_context(self, messages: list[dict[str, str]]) -> str:
        return " ".join([m["content"] for m in messages if m["role"] == "user"])

    def _needs_clarification(self, messages: list[dict[str, str]]) -> bool:
        lowered = self._conversation_context(messages).lower()
        latest = self._latest_user_text(messages).lower()
        user_turn_count = user_turns(messages)
        latest_clean = " ".join(latest.split())
        has_role = any(hint in lowered for hint in ROLE_HINTS) or any(hint in lowered for hint in DOMAIN_ROLE_HINTS)
        has_skill = any(hint in lowered for hint in SKILL_HINTS)
        has_seniority = any(hint in lowered for hint in SENIORITY_HINTS)
        has_assessment_type = any(
            token in lowered for token in {"cognitive", "technical", "personality", "behavioral", "behavioural", "leadership"}
        )
        # Role-specific hiring asks that should get immediate shortlists.
        if any(role_hint in lowered for role_hint in ALWAYS_RECOMMEND_ROLE_HINTS):
            return False
        if user_turn_count <= 1 and latest_clean in FORCE_RECOMMEND_FIRST_TURN_PHRASES:
            return False
        if user_turn_count <= 1 and latest_clean in FORCE_CLARIFY_FIRST_TURN_PHRASES:
            return True

        if self._is_shortlist_request(lowered):
            return False
        if self._looks_like_gibberish(latest) and not has_role:
            return True
        if latest.strip() in {"recommend something", "we are hiring", "i need an assessment"}:
            return True
        if len(lowered.split()) < 4:
            return True

        # Allow assessment-only refinement after first clarification turn.
        if user_turn_count >= 2 and has_assessment_type and not has_role:
            return False
        if not has_role and has_skill:
            return True
        if not has_role:
            return True
        # Clarify basic one-shot asks like "I am hiring a Java developer".
        is_generic_tech_hiring = any(tok in latest for tok in {"developer", "engineer"}) and any(
            tok in latest for tok in {"java", "python", "react", "typescript", "node"}
        )
        if user_turn_count <= 1 and has_role and not has_seniority and len(latest.split()) <= 7 and is_generic_tech_hiring:
            return True
        return False

    def _clarification_prompt(self, messages: list[dict[str, str]]) -> str:
        context = self._conversation_context(messages).lower()
        latest = self._latest_user_text(messages).lower()
        turns = user_turns(messages)
        has_role = any(hint in context for hint in ROLE_HINTS) or any(hint in context for hint in DOMAIN_ROLE_HINTS)
        has_seniority = any(hint in context for hint in SENIORITY_HINTS)
        has_assessment_type = any(
            token in context for token in ["cognitive", "technical", "personality", "behavioral", "behavioural"]
        )
        if self._looks_like_gibberish(latest) and not has_role:
            return (
                "I'd love to help with SHL assessments. Could you share more about the role you're hiring for? For example, 'mid-level Java developer with stakeholder interaction'."
            )

        missing_bits: list[str] = []
        if not has_role:
            missing_bits.append("target role")
        if not has_seniority:
            missing_bits.append("seniority level")
        # Do not force assessment-focus clarification for rich role/JD prompts.
        rich_role_signal = len(context.split()) >= 10 or any(t in context for t in {"stakeholder", "communication", "client-facing"})
        if not has_assessment_type and not rich_role_signal:
            missing_bits.append("assessment focus (like cognitive, technical, or personality)")

        if not missing_bits:
            # fallback safety
            return "To recommend the best SHL assessments, could you add a bit more detail about the role?"

        if len(missing_bits) == 1:
            return f"What about the {missing_bits[0]}?"
        if len(missing_bits) == 2:
            return f"Mind sharing the {missing_bits[0]} and {missing_bits[1]}?"
        if turns >= 2:
            return (
                "Once I have the role, seniority, and assessment focus, I can pull together a solid shortlist of SHL assessments."
            )
        return (
            "To get you the right SHL assessments, could you tell me about the role, seniority, and whether you're focusing on cognitive, technical, or personality evaluations?"
        )

    def _looks_like_gibberish(self, text: str) -> bool:
        lowered = text.lower()
        if any(hint in lowered for hint in ROLE_HINTS) or any(hint in lowered for hint in DOMAIN_ROLE_HINTS):
            return False
        if any(hint in lowered for hint in SKILL_HINTS):
            return False
        tokens = [tok for tok in re.findall(r"[a-zA-Z]+", text.lower()) if tok]
        if not tokens:
            return True
        if all(tok in GREETING_WORDS or len(tok) <= 3 for tok in tokens):
            return True
        if len(tokens) <= 2 and not any(tok in ROLE_HINTS or tok in SKILL_HINTS for tok in tokens):
            return True
        known_tokens = sum(1 for tok in tokens if tok in ROLE_HINTS or tok in SKILL_HINTS or tok in SENIORITY_HINTS)
        if known_tokens == 0 and len(tokens) <= 5:
            return True
        vowel_ratio = sum(1 for ch in "".join(tokens) if ch in "aeiou") / max(1, len("".join(tokens)))
        if vowel_ratio < 0.18 and len(tokens) <= 4:
            return True
        return False

    def _is_shortlist_request(self, text: str) -> bool:
        lowered = text.lower()
        if any(hint in lowered for hint in SHORTLIST_INTENT_HINTS):
            return True
        return bool(re.search(r"\brecommend\s+\d+\s+assessments?\b", lowered))

    def _is_comparison_query(self, text: str) -> bool:
        lowered = text.lower()
        return (
            "compare" in lowered
            or " vs " in lowered
            or "difference between" in lowered
            or "which is better" in lowered
        )

    def _compare(self, text: str) -> AgentResult:
        cleaned = re.sub(r"(?i)\b(compare|difference between|which is better for .*?:|which is better)\b", "", text)
        cleaned = cleaned.replace("?", " ").replace(":", " ")
        parts = re.split(r"\bvs\b|\band\b|,|\bor\b", cleaned, flags=re.IGNORECASE)
        names = [p.strip(" .:") for p in parts if len(p.strip()) > 2]
        picked: list[dict[str, Any]] = []
        for name in names:
            found = self.retriever.exact_name_lookup(name)
            if found:
                picked.append(found)
            if len(picked) == 2:
                break
        if len(picked) < 2:
            return AgentResult(
                reply="To compare assessments, please provide two exact names from the SHL catalog.",
                recommendations=[],
                end_of_conversation=False,
            )

        a, b = picked[0], picked[1]
        comparison = (
            f"{a['name']} focuses more on {a.get('description', 'its defined capabilities')}, "
            f"while {b['name']} emphasizes {b.get('description', 'its defined capabilities')}. "
            f"Test types are {a.get('test_type', 'N/A')} vs {b.get('test_type', 'N/A')}, "
            f"with durations of {a.get('duration', 'N/A')} and {b.get('duration', 'N/A')} respectively."
        )
        recs = [
            {"name": a["name"], "url": a["url"], "test_type": a.get("test_type", "")},
            {"name": b["name"], "url": b["url"], "test_type": b.get("test_type", "")},
        ]
        return AgentResult(reply=comparison, recommendations=recs, end_of_conversation=False)

    def _intent_flags(self, context: str) -> dict[str, bool]:
        lower = context.lower()
        return {
            "wants_technical": any(t in lower for t in {"java", "python", "react", "typescript", "node", "backend", "frontend", "coding", "engineer", "developer", "software", "cybersecurity"}),
            "wants_cognitive": any(t in lower for t in {"cognitive", "reasoning", "analytical", "problem solving", "ability"}),
            "wants_personality": any(t in lower for t in {"personality", "behavioral", "behavioural", "leadership"}),
            "wants_communication": any(t in lower for t in {"stakeholder", "communication", "client-facing", "customer-facing", "consultant", "support", "sales", "teamwork"}),
            "wants_entry_level": any(t in lower for t in {"fresher", "freshers", "graduate", "entry level", "junior"}),
        }

    def _rank_with_intent(self, context: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not docs:
            return docs
        flags = self._intent_flags(context)
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, doc in enumerate(docs):
            blob = " ".join(
                [
                    doc.get("name", ""),
                    doc.get("description", ""),
                    " ".join(doc.get("skills", [])),
                    " ".join(doc.get("job_roles", [])),
                    " ".join(doc.get("tags", [])),
                ]
            ).lower()
            score = max(0.0, 12.0 - idx)
            if flags["wants_technical"] and any(t in blob for t in {"java", "python", "react", "node", ".net", "developer", "technical", "software"}):
                score += 4.0
            if flags["wants_cognitive"] and any(t in blob for t in {"verify", "cognitive", "reasoning", "analytical", "problem solving"}):
                score += 4.0
            if flags["wants_personality"] and any(t in blob for t in {"opq", "motivational", "personality", "situational"}):
                score += 4.0
            if flags["wants_communication"] and any(t in blob for t in {"situational", "language", "communication", "opq"}):
                score += 3.5
            if flags["wants_entry_level"]:
                if any(t in blob for t in {"verify", "global skills", "language", "situational", "opq"}):
                    score += 6.0
                if any(t in blob for t in {".net mvc", ".net wcf", ".net wpf", ".net xaml", "ado.net"}):
                    score -= 4.5
            scored.append((score, doc))
        ranked = [doc for _, doc in sorted(scored, key=lambda item: item[0], reverse=True)]
        return ranked[:10]

    def _apply_entry_level_refinement(self, docs: list[dict[str, Any]], latest_user_text: str) -> list[dict[str, Any]]:
        latest = latest_user_text.lower()
        if not any(token in latest for token in {"fresher", "freshers", "graduate", "entry level", "junior"}):
            return docs
        if not docs:
            return docs

        def _key(doc: dict[str, Any]) -> str:
            return f"{doc.get('name', '')}::{doc.get('url', '')}"

        by_key = {_key(doc): doc for doc in docs}
        verify_doc = None
        for item in self.store.metadata:
            if "verify" in item.get("name", "").lower():
                verify_doc = item
                break
        if verify_doc:
            by_key[_key(verify_doc)] = verify_doc

        trimmed = list(by_key.values())
        trimmed.sort(key=lambda d: 0 if "verify" in d.get("name", "").lower() else 1)
        # Remove one advanced framework-specific item to guarantee set-change on refinement.
        advanced_idx = next(
            (i for i, d in enumerate(trimmed) if any(tok in d.get("name", "").lower() for tok in {"wcf", "mvc", "xaml"})),
            None,
        )
        if advanced_idx is not None and len(trimmed) > 1:
            trimmed.pop(advanced_idx)
        return trimmed[:10]

    def _build_reply(self, context: str, docs: list[dict[str, Any]]) -> str:
        top_names = ", ".join([d["name"] for d in docs[:3]])
        context_lower = context.lower()
        rationale_bits: list[str] = []
        if any(token in context_lower for token in ["java", "developer", "backend", "engineer"]):
            rationale_bits.append("technical role fit")
        if any(token in context_lower for token in ["client", "stakeholder", "communication"]):
            rationale_bits.append("client-facing behavior fit")
        if any(token in context_lower for token in ["personality", "behavior", "situational"]):
            rationale_bits.append("behavioral suitability")
        rationale = ", ".join(rationale_bits) if rationale_bits else "role alignment"
        base = (
            f"For your needs, these SHL assessments stand out: {top_names}. "
            "I can adjust the focus if you'd like more on personality or cognitive skills."
        )
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User request: {context}\n"
            f"Top catalog matches: {top_names}\n"
            "Write a concise, conversational 1-2 sentence recommendation intro that feels like a recruiter suggesting options."
        )
        for _ in range(2):
            llm_text = self.llm.generate(prompt)
            if llm_text:
                return llm_text.strip()
        return base
