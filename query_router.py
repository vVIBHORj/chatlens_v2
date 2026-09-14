import re

from dataclasses import dataclass, field

from enum import Enum

from typing import List

class QueryIntent(str, Enum):
    SEMANTIC_RAG = "semantic_rag"
    STATISTICS = "statistics"
    TIME_ANALYSIS = "time_analysis"
    COMPARISON = "comparison"
    BEHAVIOR_ANALYSIS = "behavior_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    CONTEXT = "context"


@dataclass
class QueryRoute:
    primary_intent: QueryIntent
    secondary_intents: List[QueryIntent]
    
@dataclass
class QuerySpec:

    intent: QueryIntent

    target: str | None = None

    entities: List[str] = field(
        default_factory=list
    )

    answer_type: str = "semantic"

    constraints: List[str] = field(
        default_factory=list
    )

def build_query_spec(query: str) -> QuerySpec:
    """
    Build a deterministic structured representation of a user query.

    The function identifies:
    - analytical intent
    - likely target phrase
    - explicitly named entities
    - expected answer type
    - retrieval constraints

    It does not answer the question and contains no
    domain-specific topic vocabulary.
    """

    route = route_query(query)
    text = normalize_query(query)

    constraints = []
    entities = []
    target = None

    if not text:
        return QuerySpec(
            intent=route.primary_intent,
            target=None,
            entities=[],
            answer_type="semantic",
            constraints=[],
        )

    # --------------------------------------------------------
    # Explicit-information constraint
    # --------------------------------------------------------

    if re.search(
        r"\bexplicit(?:ly)?\b",
        text,
    ):
        constraints.append("explicit")

    # --------------------------------------------------------
    # Speaker-specific constraint
    #
    # Only treat a word as a speaker when the grammatical
    # structure indicates a named subject.
    # --------------------------------------------------------

    speaker_match = re.search(
        r"\bwhat did\s+([a-z0-9_]+)\s+(?:say|mention|tell|write|talk)\b",
        text,
    )

    if speaker_match:
        entities.append(
            speaker_match.group(1)
        )
        constraints.append("speaker_specific")

    # --------------------------------------------------------
    # Answer type
    # --------------------------------------------------------

    answer_type = "semantic"

    if re.search(
        r"\bwho\b",
        text,
    ):
        answer_type = "person"

    elif re.search(
        r"\bhow many\b|\bhow much\b|\bpercentage\b|\baverage\b",
        text,
    ):
        answer_type = "numeric"

    elif re.search(
        r"\bwhat .* mentioned\b"
        r"|\bwhat .* discussed\b"
        r"|\bwhich .* mentioned\b",
        text,
    ):
        answer_type = "list"

    elif route.primary_intent in {
        QueryIntent.STATISTICS,
        QueryIntent.TIME_ANALYSIS,
        QueryIntent.BEHAVIOR_ANALYSIS,
        QueryIntent.SENTIMENT_ANALYSIS,
        QueryIntent.COMPARISON,
        QueryIntent.ANOMALY_DETECTION,
    }:
        answer_type = "analytic"

    # --------------------------------------------------------
    # Generic target extraction
    #
    # For now we only extract the object of common semantic
    # question constructions. No topic vocabulary is used.
    # --------------------------------------------------------

    target_patterns = [
        r"\bwhat did (?:we|you|they)\s+(?:discuss|talk about)\s+(.+)$",
        r"\bwhat did\s+[a-z0-9_]+\s+(?:say|mention|tell|write)\s+about\s+(.+)$",
        r"\bwhat .*?\s+about\s+(.+)$",
        r"\bwhat .*?\s+(?:mentioned|discussed)\s+(.+)$",
    ]

    for pattern in target_patterns:

        match = re.search(
            pattern,
            text,
        )

        if match:

            candidate = match.group(1).strip()

            if candidate:
                target = candidate
                break

    # --------------------------------------------------------
    # Remove entity from target when the query explicitly
    # identifies a speaker.
    # --------------------------------------------------------

    if target and entities:

        for entity in entities:

            target = re.sub(
                rf"\b{re.escape(entity)}\b",
                "",
                target,
            ).strip()

    if target:
        target = re.sub(
            r"\s+",
            " ",
            target,
        ).strip()

    # --------------------------------------------------------
    # Aggregation constraint
    # --------------------------------------------------------

    if route.primary_intent in {
        QueryIntent.STATISTICS,
        QueryIntent.COMPARISON,
    }:
        constraints.append("aggregated")

    return QuerySpec(
        intent=route.primary_intent,
        target=target,
        entities=entities,
        answer_type=answer_type,
        constraints=constraints,
    )



def normalize_query(query: str) -> str:
    if not query:
        return ""

    return re.sub(
        r"\s+",
        " ",
        query.lower(),
    ).strip()
    
def extract_query_terms(query: str) -> List[str]:
    """
    Extract likely content-bearing terms from a query.

    This is domain-agnostic. It does not contain knowledge about
    specific topics such as sports, travel, money, laptops, etc.

    Analytical/statistical queries are handled separately by the
    query router, so their structural words are not treated as
    semantic retrieval targets.
    """

    text = normalize_query(query)

    if not text:
        return []

    tokens = re.findall(
        r"\b[a-z0-9']+\b",
        text,
    )

    structural_words = {
        "what", "whats", "what's",
        "who", "whom", "whose", "which",
        "where", "when", "why", "how",

        "is", "are", "am", "was", "were",
        "be", "been", "being",

        "do", "does", "did",
        "can", "could", "would", "should",
        "will", "shall", "may", "might",

        "have", "has", "had",

        "i", "me", "my",
        "we", "our",
        "you", "your",
        "they", "their",

        "it", "its",
        "this", "that", "these", "those",

        "the", "a", "an",
        "of", "to", "for", "from",
        "with", "about", "in", "on", "at",
        "by", "and", "or", "but", "as",
        "into", "over", "between",
        "through", "up", "down",

        "more", "most", "less", "least",
        "fewer", "many", "much",
        "often", "usually",

        "really", "just",
        "main", "overall", "general",
        "common", "mainly",
        "each", "person", "people",

        "chat", "chats",
        "conversation", "conversations",
        "message", "messages",

        "discuss", "discussed", "discussing",
        "talk", "talked", "talking",
        "say", "said",
        "tell", "told",
        "decide", "decided", "decision",

        "behind", "context",
        "explicitly", "explicit",
        "mentioned",
    }

    analytical_terms = {
        QueryIntent.STATISTICS: {
            "send", "sends", "sent",
            "sending",
            "count", "number",
            "average", "percentage",
            "total", "highest", "lowest",
            "longest", "shortest",
            "frequency",
        },

        QueryIntent.TIME_ANALYSIS: {
            "time", "times",
            "day", "days",
            "week", "weeks",
            "month", "months",
            "year", "years",
            "active", "activity",
            "frequency",
            "change", "changed",
            "over",
        },

        QueryIntent.BEHAVIOR_ANALYSIS: {
            "initiate", "initiates",
            "starts", "start",
            "responds", "respond",
            "replies", "reply",
            "asks", "ask",
            "uses", "use",
            "writes", "write",
            "longer",
            "style", "pattern",
            "communication",
            "conversation",
            "emojis", "emoji",
        },

        QueryIntent.SENTIMENT_ANALYSIS: {
            "sentiment",
            "tone",
            "positive", "negative",
            "happy", "sad",
            "angry", "frustrated",
            "excited",
            "emotional",
        },

        QueryIntent.COMPARISON: {
            "compare", "comparison",
            "versus", "vs",
            "increased", "decreased",
            "change",
        },

        QueryIntent.ANOMALY_DETECTION: {
            "unusual", "unusually",
            "anomaly", "anomalies",
            "abnormal",
            "spike", "spikes",
            "outlier", "outliers",
        },
    }

    route = route_query(query)

    excluded_terms = set(structural_words)

    excluded_terms.update(
        analytical_terms.get(
            route.primary_intent,
            set(),
        )
    )

    terms = []

    for token in tokens:

        if token in excluded_terms:
            continue

        if len(token) <= 2:
            continue

        if token not in terms:
            terms.append(token)

    return terms

def detect_secondary_intents(query: str) -> List[QueryIntent]:
    """
    Detect additional analytical dimensions present in a query.

    Secondary intents do not replace the primary intent.
    They describe additional work that may be required.
    """

    text = normalize_query(query)

    secondary = []

    def add(intent: QueryIntent):
        if intent not in secondary:
            secondary.append(intent)

    # --------------------------------------------------------
    # Temporal dimension
    # --------------------------------------------------------

    temporal_patterns = [
        r"\bwhen\b",
        r"\bwhat time\b",
        r"\bover time\b",
        r"\blast year\b",
        r"\blast month\b",
        r"\blast week\b",
        r"\bthis month\b",
        r"\bthis week\b",
        r"\bbetween january\b",
        r"\bbetween february\b",
        r"\bbetween march\b",
        r"\bbetween april\b",
        r"\bbetween may\b",
        r"\bbetween june\b",
        r"\bbetween july\b",
        r"\bbetween august\b",
        r"\bbetween september\b",
        r"\bbetween october\b",
        r"\bbetween november\b",
        r"\bbetween december\b",
        r"\bchange\b.*\btime\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in temporal_patterns
    ):
        add(QueryIntent.TIME_ANALYSIS)

    # --------------------------------------------------------
    # Statistical dimension
    # --------------------------------------------------------

    statistical_patterns = [
        r"\bhow many\b",
        r"\bhow much\b",
        r"\baverage\b",
        r"\bcount\b",
        r"\btotal\b",
        r"\bpercentage\b",
        r"\bmore\b",
        r"\bless\b",
        r"\bfewer\b",
        r"\bmost\b",
        r"\bfewest\b",
        r"\blongest\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in statistical_patterns
    ):
        add(QueryIntent.STATISTICS)

    # --------------------------------------------------------
    # Behavior dimension
    # --------------------------------------------------------

    behavior_patterns = [
        r"\bwho initiates\b",
        r"\bwho starts\b",
        r"\bwho sends\b",
        r"\bwho asks\b",
        r"\bwho responds\b",
        r"\bwho uses\b",
        r"\bwho tends to\b",
        r"\bcommunication style\b",
        r"\bcommunication pattern\b",
        r"\bcommunication\b",
        r"\bconversation flow\b",
        r"\bkeep conversations going\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in behavior_patterns
    ):
        add(QueryIntent.BEHAVIOR_ANALYSIS)

    # --------------------------------------------------------
    # Sentiment / emotional language dimension
    # --------------------------------------------------------

    sentiment_patterns = [
        r"\bsentiment\b",
        r"\btone\b",
        r"\bpositive language\b",
        r"\bnegative language\b",
        r"\bpositive\b",
        r"\bnegative\b",
        r"\bhappy\b",
        r"\bsad\b",
        r"\bangry\b",
        r"\bfrustrated\b",
        r"\bexcited\b",
        r"\bemotional language\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in sentiment_patterns
    ):
        add(QueryIntent.SENTIMENT_ANALYSIS)

    # --------------------------------------------------------
    # Semantic / topic dimension
    # --------------------------------------------------------

    semantic_patterns = [
        r"\btopic\b",
        r"\btopics\b",
        r"\bdiscuss\b",
        r"\btalked about\b",
        r"\bwhat did\b",
        r"\bwhat were\b",
        r"\bwhat was\b",
        r"\bdecide\b",
        r"\bdecision\b",
        r"\bdiscussed\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in semantic_patterns
    ):
        add(QueryIntent.SEMANTIC_RAG)

    # --------------------------------------------------------
    # Anomaly dimension
    # --------------------------------------------------------

    anomaly_patterns = [
        r"\bunusual\b",
        r"\banomal",
        r"\bspike\b",
        r"\babnormal\b",
        r"\boutlier\b",
        r"\bunusually\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in anomaly_patterns
    ):
        add(QueryIntent.ANOMALY_DETECTION)

    return secondary


    
    

def detect_query_intent(query: str) -> QueryIntent:
    """
    Detect the broad analytical intent of a user query.

    This is intentionally deterministic and conservative.
    It does not attempt to answer the question.
    """

    text = normalize_query(query)

    if not text:
        return QueryIntent.SEMANTIC_RAG

    # --------------------------------------------------------
    # Context / specific-message questions
    # --------------------------------------------------------

    context_patterns = [
        r"\bcontext\b",
        r"\bwhat happened around\b",
        r"\bwhat was happening when\b",
        r"\bwhat did .* mean\b",
        r"\bwhat was the context behind\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in context_patterns
    ):
        return QueryIntent.CONTEXT

    # --------------------------------------------------------
    # Comparison questions
    # --------------------------------------------------------

    comparison_patterns = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bbetween .* and\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bmore than\b",
        r"\bless than\b",
        r"\bincreased\b",
        r"\bdecreased\b",
        r"\bchange between\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in comparison_patterns
    ):
        return QueryIntent.COMPARISON

    # --------------------------------------------------------
    # Anomaly / unusual-pattern questions
    # --------------------------------------------------------

    anomaly_patterns = [
        r"\bunusual\b",
        r"\banomal",
        r"\bspike\b",
        r"\babnormal\b",
        r"\bsignificant change\b",
        r"\bdiffer significantly\b",
        r"\bunusually\b",
        r"\boutlier\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in anomaly_patterns
    ):
        return QueryIntent.ANOMALY_DETECTION

    # --------------------------------------------------------
    # Time / temporal analysis
    # --------------------------------------------------------

    time_patterns = [
        r"\bover time\b",
        r"\blast month\b",
        r"\bthis month\b",
        r"\blast week\b",
        r"\bthis week\b",
        r"\bwhen are\b",
        r"\bwhen did\b",
        r"\bwhat time\b",
        r"\bday of the week\b",
        r"\bfrequency\b",
        r"\bperiods\b",
        r"\bhow has .* changed\b",
        r"\bbetween january\b",
        r"\bbetween february\b",
        r"\bbetween march\b",
        r"\bbetween april\b",
        r"\bbetween may\b",
        r"\bbetween june\b",
        r"\bbetween july\b",
        r"\bbetween august\b",
        r"\bbetween september\b",
        r"\bbetween october\b",
        r"\bbetween november\b",
        r"\bbetween december\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in time_patterns
    ):
        return QueryIntent.TIME_ANALYSIS

    # --------------------------------------------------------
    # Behavior / communication analysis
    # --------------------------------------------------------

    behavior_patterns = [
        # Conversation initiation
        r"\bwho initiates\b",
        r"\bwho usually starts\b",
        r"\bwho starts conversations\b",
        r"\bwho starts the conversation\b",

        # Participation / messaging behavior
        r"\bwho sends longer\b",
        r"\bwho writes longer\b",
        r"\bwho asks more\b",
        r"\bwho responds faster\b",
        r"\bwho responds quicker\b",
        r"\bwho replies faster\b",

        # Conversation control
        r"\bwho tends to\b",
        r"\bwho usually\b.*\btopic\b",
        r"\bwho changes the topic\b",
        r"\bwho changes the subject\b",
        r"\bwho keeps conversations going\b",

        # Communication style
        r"\bcommunication pattern\b",
        r"\bcommunication style\b",
        r"\bconversation style\b",
        r"\bconversation flow\b",
        r"\bcasual language\b",
        r"\bformal language\b",

        # Messaging characteristics
        r"\bemoji\b",
        r"\bemojis\b",
        r"\bresponse time\b",
        r"\breply time\b",
        r"\bhow quickly do we usually respond\b",
        r"\bhow quickly do we respond\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in behavior_patterns
    ):
        return QueryIntent.BEHAVIOR_ANALYSIS

    # --------------------------------------------------------
    # Sentiment / emotional language analysis
    # --------------------------------------------------------

    sentiment_patterns = [
        r"\bsentiment\b",
        r"\btone\b",
        r"\bpositive language\b",
        r"\bnegative language\b",
        r"\bemotional language\b",
        r"\bhow positive\b",
        r"\bhow negative\b",
        r"\bmost positive\b",
        r"\bmost negative\b",
        r"\bmore positive\b",
        r"\bmore negative\b",
        r"\bless positive\b",
        r"\bless negative\b",
        r"\bwhen .* positive\b",
        r"\bwhen .* negative\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in sentiment_patterns
    ):
        return QueryIntent.SENTIMENT_ANALYSIS

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics_patterns = [
        r"\bhow many\b",
        r"\bhow much\b",
        r"\bpercentage\b",
        r"\baverage\b",
        r"\bmost common\b",
        r"\bmost frequently\b",
        r"\bcount\b",
        r"\btotal\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\blongest\b",
        r"\bnumber of messages\b",

    # Message-count / participation questions
        r"\bwho sends more\b",
        r"\bwho sends the most\b",
        r"\bwho sent more\b",
        r"\bwho sent the most\b",
        r"\bwho sends fewer\b",
        r"\bwho sent fewer\b",
        r"\bwho sends less\b",
        r"\bwho sent less\b",
        r"\bmost messages\b",
        r"\bfewest messages\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in statistics_patterns
    ):
        return QueryIntent.STATISTICS

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return QueryIntent.SEMANTIC_RAG


def route_query(query: str) -> QueryRoute:
    """
    Build a complete query route consisting of one primary
    intent and zero or more supporting intents.
    """

    primary = detect_query_intent(query)

    secondary = detect_secondary_intents(query)

    # The primary intent should never appear as a secondary intent.
    secondary = [
        intent
        for intent in secondary
        if intent != primary
    ]

    return QueryRoute(
        primary_intent=primary,
        secondary_intents=secondary,
    )


if __name__ == "__main__":

    TEST_QUERIES = [

        # --------------------------------------------------------
        # Conversation & Topic Analysis
        # --------------------------------------------------------
        "What are the main topics discussed in this chat?",
        "What did we discuss about travel?",
        "What were the main things we talked about last year?",
        "Which topics come up most often?",

        # --------------------------------------------------------
        # Individual Behavior
        # --------------------------------------------------------
        "Who sends more messages?",
        "Who initiates conversations more often?",
        "Who usually starts conversations?",
        "Who sends longer messages?",
        "Who asks more questions?",

        # --------------------------------------------------------
        # Time & Activity Patterns
        # --------------------------------------------------------
        "When are we most active?",
        "What day of the week do we chat the most?",
        "What time do we usually talk?",
        "When did we talk the most?",
        "How has our activity changed over time?",

        # --------------------------------------------------------
        # Communication Style
        # --------------------------------------------------------
        "How would you describe each person's communication style?",
        "Who uses more emojis?",
        "Who tends to write longer messages?",
        "Who uses more casual language?",
        "Who tends to change the subject?",

        # --------------------------------------------------------
        # Conversation Dynamics
        # --------------------------------------------------------
        "Who usually changes the topic?",
        "Who responds faster?",
        "How quickly do we usually respond to each other?",
        "Who tends to keep conversations going?",

        # --------------------------------------------------------
        # Sentiment / Emotional Language
        # --------------------------------------------------------
        "What is the overall sentiment of the conversation?",
        "Who uses more positive language?",
        "Who uses more negative language?",
        "How does the tone of the conversation change over time?",
        "When was the conversation most positive?",

        # --------------------------------------------------------
        # Semantic / Relationship Context
        # --------------------------------------------------------
        "What did Udit say about money?",
        "What did we discuss about relationships?",
        "What did we decide about the trip?",
        "What was the context behind this message?",

        # --------------------------------------------------------
        # Statistics & Insights
        # --------------------------------------------------------
        "How many messages did each person send?",
        "What is the average number of messages per day?",
        "What is the most common topic?",
        "Who sent the longest message?",
        "How many messages did we send in total?",

        # --------------------------------------------------------
        # Change Over Time
        # --------------------------------------------------------
        "What topics have increased over time?",
        "What topics have decreased over time?",
        "How did our communication change between January and June?",
        "Has our messaging frequency changed?",

        # --------------------------------------------------------
        # Anomaly / Pattern Detection
        # --------------------------------------------------------
        "Are there unusual spikes in messaging activity?",
        "Were there any unusually active days?",
        "Are there any unusual changes in our conversation pattern?",
    ]

    print("=" * 80)
    print("QUERY ROUTER TEST")
    print("=" * 80)

    for query in TEST_QUERIES:

        route = route_query(query)

        print(f"\nQuery: {query}")
        print(f"Primary: {route.primary_intent.value}")

        if route.secondary_intents:
            print(
                "Secondary: "
                + ", ".join(
                    intent.value
                    for intent in route.secondary_intents
                )
            )
        else:
            print("Secondary: none")
            
            
if __name__ == "__main__":

    test_queries = [
        "What sports are explicitly mentioned in the conversation?",
        "What did Udit say about money?",
        "What did we discuss about travel?",
        "What laptop was EXODIA considering buying?",
        "Who sends the most messages?",
        "When are we most active?",
        "Who uses more emojis?",
        "What is the overall sentiment of the conversation?",
    ]

    for query in test_queries:

        spec = build_query_spec(query)

        print()
        print("QUERY:", query)
        print("INTENT:", spec.intent.value)
        print("TARGET:", spec.target)
        print("ENTITIES:", spec.entities)
        print("ANSWER TYPE:", spec.answer_type)
        print("CONSTRAINTS:", spec.constraints)