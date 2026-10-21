"""Prometheus metrics, exposed on /metrics.

Recall@k and citation correctness need labelled data, so they are measured offline
by `scripts/evaluate_retrieval.py` rather than in production.
"""

from prometheus_client import Counter, Histogram

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30)

HTTP_REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "rag_http_request_duration_seconds", "HTTP request latency", ["method", "route"], buckets=LATENCY_BUCKETS
)

CHAT_REQUESTS = Counter("rag_chat_requests_total", "Chat questions by outcome", ["outcome"])
RESPONSE_LATENCY = Histogram("rag_response_latency_seconds", "End-to-end answer latency", buckets=LATENCY_BUCKETS)
RETRIEVAL_LATENCY = Histogram("rag_retrieval_latency_seconds", "Retrieval latency", buckets=LATENCY_BUCKETS)
RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks", "Authorized chunks returned per question", buckets=(0, 1, 2, 3, 5, 8, 13, 20)
)
LLM_TIME_TO_FIRST_TOKEN = Histogram(
    "rag_llm_time_to_first_token_seconds", "LLM time to first token", buckets=LATENCY_BUCKETS
)
LLM_TOKENS = Counter("rag_llm_tokens_total", "LLM tokens consumed", ["kind"])

INVALID_CITATIONS = Counter("rag_invalid_citations_total", "Citations to sources that were never provided")
UNCITED_ANSWERS = Counter("rag_uncited_answers_total", "Model answers rejected for having no valid citation")
UNAUTHORIZED_REQUESTS = Counter(
    "rag_unauthorized_requests_total", "Rejected authentication or authorization attempts", ["reason"]
)
