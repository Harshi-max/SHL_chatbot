from app.rag.catalog_store import CatalogStore
from app.services.agent import RecommendationAgent


def _agent() -> RecommendationAgent:
    store = CatalogStore()
    store.records = []
    store.metadata = []
    agent = RecommendationAgent(store)
    agent.retriever.retrieve = lambda _q, top_k=10: []
    return agent


def test_gibberish_prompts_clarification() -> None:
    agent = _agent()
    result = agent.handle([{"role": "user", "content": "heyy hello abcc"}])
    assert result.recommendations == []
    assert "shl" in result.reply.lower()
    assert "role" in result.reply.lower()


def test_skill_only_prompts_role_clarification() -> None:
    agent = _agent()
    result = agent.handle([{"role": "user", "content": "cognitive"}])
    assert result.recommendations == []
    assert "clarify" in result.reply.lower() or "role" in result.reply.lower()


def test_repeated_clarification_is_more_guided() -> None:
    agent = _agent()
    result = agent.handle(
        [
            {"role": "user", "content": "I need an assessment"},
            {"role": "assistant", "content": "Could you clarify the target role, seniority, and whether you want cognitive, technical, or personality assessments?"},
            {"role": "user", "content": "hello"},
        ]
    )
    assert result.recommendations == []
    assert "share role + seniority + skills" in result.reply.lower()
