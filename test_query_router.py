"""
test_query_router.py

Focused regression test suite for query intent classification and scope routing.

Tests coverage:
1. Factual questions using analytical vocabulary ('frequency', 'period', 'compare', 'between ... and')
   in a non-analytical informational sense -> must route to specific retrieval (semantic_rag / specific).
2. Genuine requests for time analysis, activity patterns, trends, and message frequency
   -> must route to analytical profile (time_analysis / broad).
3. Genuine comparative requests regarding communication dynamics or activity across periods
   -> must route to analytical comparison (comparison / broad).
4. Contextual questions requiring surrounding conversational messages
   -> must route to specific retrieval with context expansion (context / specific).
5. Existing analytical intents (statistics, behavior, sentiment, anomaly detection)
   -> must remain unchanged.
"""

import pytest

from query_router import (
    QueryIntent,
    build_query_spec,
    detect_query_intent,
    route_query,
)
from rag_graph import classify_scope


def assert_routing(
    question: str,
    expected_intent: QueryIntent,
    expected_scope: str,
    expected_answer_type: str = None,
):
    """
    Asserts both the intent from query_router and the resulting scope from classify_scope.
    """
    route = route_query(question)
    spec = build_query_spec(question)

    state = {
        "question": question,
        "primary_intent": route.primary_intent.value,
    }
    scoped_state = classify_scope(state)

    actual_intent = route.primary_intent
    actual_scope = scoped_state["scope"]

    assert actual_intent == expected_intent, (
        f"Query: '{question}'\n"
        f"Expected intent: {expected_intent.value}, got: {actual_intent.value}"
    )
    assert actual_scope == expected_scope, (
        f"Query: '{question}'\n"
        f"Expected scope: {expected_scope}, got: {actual_scope}"
    )

    if expected_answer_type is not None:
        assert spec.answer_type == expected_answer_type, (
            f"Query: '{question}'\n"
            f"Expected answer_type: {expected_answer_type}, got: {spec.answer_type}"
        )


# =====================================================================
# 1. Factual Questions with Analytical Keywords (Frequency, Period, etc.)
# =====================================================================

@pytest.mark.parametrize(
    "question",
    [
        "What is the calibration frequency of the telescope?",
        "What radio frequency do they use?",
        "Which transmission frequency was selected for the broadcast?",
        "What was the frequency mentioned by Dr. Aris?",
        "Did they discuss the vibration frequency of the engine?",
        "Tell me the operating frequency of the sensor.",
    ],
)
def test_factual_questions_with_frequency_route_to_specific_retrieval(question: str):
    """
    Factual inquiries asking for a frequency fact in the archive must route
    to semantic_rag with specific scope, not time_analysis.
    """
    assert_routing(
        question=question,
        expected_intent=QueryIntent.SEMANTIC_RAG,
        expected_scope="specific",
    )


@pytest.mark.parametrize(
    "question",
    [
        "What is the period of the experiment?",
        "What are the trial periods for the new software?",
        "Did anyone mention the incubation period?",
        "What is the warranty period for the laptop?",
        "What was the observation period discussed yesterday?",
        "Tell me what trial period was agreed upon.",
    ],
)
def test_factual_questions_with_period_route_to_specific_retrieval(question: str):
    """
    Factual inquiries using 'period' or 'periods' to denote a duration or timeline
    must route to semantic_rag with specific scope, not time_analysis.
    """
    assert_routing(
        question=question,
        expected_intent=QueryIntent.SEMANTIC_RAG,
        expected_scope="specific",
    )


@pytest.mark.parametrize(
    "question",
    [
        "Did they compare the two laptops?",
        "What did Alice compare in her review?",
        "Where in the notes did they compare the prices?",
        "Did anyone compare the battery life of both models?",
        "What did the team compare during yesterday's discussion?",
    ],
)
def test_factual_questions_about_discussion_comparison_route_to_specific_retrieval(question: str):
    """
    Questions asking what participants discussed or compared in the chat
    must route to semantic_rag with specific retrieval, not broad chat comparison.
    """
    assert_routing(
        question=question,
        expected_intent=QueryIntent.SEMANTIC_RAG,
        expected_scope="specific",
    )


@pytest.mark.parametrize(
    "question",
    [
        "What was the score between Arsenal and Chelsea?",
        "What happened between Alice and Bob during the meeting?",
        "What did they discuss between 2 PM and 3 PM?",
        "What was decided between Option A and Option B?",
    ],
)
def test_factual_questions_with_between_and_route_to_specific_retrieval(question: str):
    """
    Factual questions containing 'between X and Y' (time, participants, or options)
    must route to semantic_rag with specific retrieval.
    """
    assert_routing(
        question=question,
        expected_intent=QueryIntent.SEMANTIC_RAG,
        expected_scope="specific",
    )


# =====================================================================
# 2. Genuine Time Analysis and Message Frequency Requests
# =====================================================================

@pytest.mark.parametrize(
    "question",
    [
        "Analyze message frequency over time",
        "How has our messaging frequency changed?",
        "How has message frequency changed over time?",
        "When are we most active?",
        "What day of the week do we chat the most?",
        "What time do we usually talk?",
        "When did we talk the most?",
        "How has our activity changed over time?",
        "How does the tone of the conversation change over time?",
    ],
)
def test_genuine_time_analysis_requests_route_to_broad_profile(question: str):
    """
    Requests genuinely asking for temporal patterns or messaging frequency trends
    must route to time_analysis with broad scope.
    """
    assert_routing(
        question=question,
        expected_intent=QueryIntent.TIME_ANALYSIS,
        expected_scope="broad",
    )


# =====================================================================
# 3. Genuine Comparative Requests Across Chat Dimensions
# =====================================================================

@pytest.mark.parametrize(
    "question",
    [
        "Compare activity across periods",
        "Compare activity between January and June",
        "Compare the messaging volume of Alice and Bob",
        "How did our communication change between January and June?",
        "What topics have increased over time?",
        "What topics have decreased over time?",
    ],
)
def test_genuine_comparison_requests_route_to_broad_profile(question: str):
    """
    Requests genuinely asking to compare communication volume, trends, or activity
    across participants or periods must route to comparison with broad scope.
    """
    assert_routing(
        question=question,
        expected_intent=QueryIntent.COMPARISON,
        expected_scope="broad",
    )


# =====================================================================
# 4. Contextual Questions
# =====================================================================

@pytest.mark.parametrize(
    "question",
    [
        "What was the context behind this message?",
        "What was happening when the server crashed?",
        "What happened around the time of the announcement?",
        "What did Bob mean by that remark?",
        "Can you give context for what happened around lunch?",
    ],
)
def test_contextual_questions_route_to_context_specific(question: str):
    """
    Questions asking for surrounding context around a message or event
    must route to context with specific scope.
    """
    assert_routing(
        question=question,
        expected_intent=QueryIntent.CONTEXT,
        expected_scope="specific",
    )


# =====================================================================
# 5. Standard Semantic Retrieval (Existing Baseline)
# =====================================================================

@pytest.mark.parametrize(
    "question, expected_target, expected_entities",
    [
        ("What did Udit say about money?", "money", ["udit"]),
        ("What did we discuss about travel?", "travel", []),
        ("What laptop was EXODIA considering buying?", "laptop", []),
        ("What sports are explicitly mentioned in the conversation?", "sports", []),
    ],
)
def test_standard_semantic_questions_remain_specific(
    question: str,
    expected_target: str,
    expected_entities: list,
):
    """
    Ensures standard factual retrieval questions remain semantic_rag with specific scope,
    and preserves correct QuerySpec entity/target extraction.
    """
    assert_routing(
        question=question,
        expected_intent=QueryIntent.SEMANTIC_RAG,
        expected_scope="specific",
    )
    spec = build_query_spec(question)
    assert spec.target == expected_target
    assert spec.entities == expected_entities


# =====================================================================
# 6. Other Analytical Categories (Existing Baseline Unchanged)
# =====================================================================

@pytest.mark.parametrize(
    "question, expected_intent",
    [
        ("Who sends more messages?", QueryIntent.STATISTICS),
        ("Who sends the most messages?", QueryIntent.STATISTICS),
        ("How many messages did each person send?", QueryIntent.STATISTICS),
        ("What is the average number of messages per day?", QueryIntent.STATISTICS),
        ("Who uses more emojis?", QueryIntent.BEHAVIOR_ANALYSIS),
        ("Who initiates conversations more often?", QueryIntent.BEHAVIOR_ANALYSIS),
        ("Who tends to write longer messages?", QueryIntent.BEHAVIOR_ANALYSIS),
        ("What is the overall sentiment of the conversation?", QueryIntent.SENTIMENT_ANALYSIS),
        ("Who uses more positive language?", QueryIntent.SENTIMENT_ANALYSIS),
        ("Are there unusual spikes in messaging activity?", QueryIntent.ANOMALY_DETECTION),
        ("Were there any unusually active days?", QueryIntent.ANOMALY_DETECTION),
    ],
)
def test_existing_analytical_categories_remain_broad(
    question: str,
    expected_intent: QueryIntent,
):
    """
    Ensures that existing statistics, behavior analysis, sentiment analysis,
    and anomaly detection intents remain properly classified as broad scope.
    """
    assert_routing(
        question=question,
        expected_intent=expected_intent,
        expected_scope="broad",
    )
